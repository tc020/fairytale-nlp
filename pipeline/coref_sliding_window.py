import os
import re
import warnings
from collections import defaultdict
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SEED = 20250101
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import spacy
from nltk.tokenize import word_tokenize
from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding
from maverick_de import Maverick

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"


# =====================================================
# Konfiguration Sliding Window
# =====================================================

MAX_WINDOW_TOKENS = 750   # maximale Fenstergröße (Kontext + Chunk), Modell-Limit: 1024
OVERLAP_TOKENS = 150      # Überlappung; dient dem nächsten Fenster nur als Kontext


# =====================================================
# Ersetzungsregeln nach Pronomentyp (STTS)
# =====================================================

#vollständig durch den Repräsentanten ersetzen
ERSETZEN_TAGS = {
    "PPER",   #Personalpronomen: "er", "sie", "ihm"
    "PDS",    #substituierendes Demonstrativum: "Der aber sprach ..."
}

#im Text stehen lassen, weil syntaktisch tragend oder generisch
BEHALTEN_TAGS = {
    "PRELS",  #Relativpronomen: "die Zauberin, DIE ..."
    "PRELAT", #attributives Relativpronomen: "DEREN Tochter"
    "PRF",    #Reflexivpronomen: "freute SICH"
    "PIS",    #Indefinitum: "MAN", "jemand", "alle"
    "PPOSS",  #substituierendes Possessivum: "das ist MEINER"
    "PWS",    #Interrogativum: "WER"
    "PWAT",   #attributives Interrogativum: "WESSEN"
    "PWAV",   #Pronominaladverb/Interrogativadverb: "WO", "WIE"
}

PROTOKOLL_TAGS = {
    "PPOSAT", #attributives Possessivum: "mit SEINEM Kahne"
}

PRONOMEN_TAGS = ERSETZEN_TAGS | BEHALTEN_TAGS | PROTOKOLL_TAGS

#zulässige Begleiter für Fall 3 (Zweitoken-Mention "DET + Nomen"):
#der Begleiter bleibt stehen, nur der Kopf wird vereinheitlicht
#("die Alte" -> "die Hexe", "seine Frau" -> "seine Königin")
ARTIKELWORT_TAGS = {"ART", "PDAT", "PPOSAT", "PIAT"}

#Ab dieser Tokenzahl gilt eine Erwaehnung als Langspanne
LANGSPANNE_AB = 5

#Punktwerte je Belegklasse. Der Abstand ist bewusst gross, damit ein
#einzelner starker Beleg viele schwache überwiegt
PUNKTE_BELEG = {
    "eigenstaendig": 100,   #die Erwähnung besteht ganz aus dem Namen
    "kopfposition": 50,     #der Name steht am Kopf einer kurzen Phrase
    "schwach": 5,           #der Name steht sonstwo in einer kurzen Phrase
    "langspanne": 1,        #nur innerhalb einer überlangen Spanne
}

#Rangfolge der Belegklassen, fuer Vergleiche und die Vererbung
RANG_BELEG = {"langspanne": 0, "schwach": 1,
              "kopfposition": 2, "eigenstaendig": 3}

#Als sicher gilt ein Cluster, dessen Repraesentant so belegt ist
BELEG_SICHER = {"eigenstaendig", "kopfposition"}

ERSETZE_UNSICHERE_CLUSTER = True

VEREINHEITLICHE_REPRAESENTANTEN = True

#Wörter, nach denen ein Name als Attribut und nicht als Kopf steht
GENITIV_MARKER = {"des", "von", "vom", "der"}


def finde_teilfolge(tokens: tuple, folge: tuple) -> list:
    """Startpositionen, an denen `folge` in `tokens` vorkommt."""
    if not folge or len(folge) > len(tokens):
        return []
    n = len(folge)
    return [i for i in range(len(tokens) - n + 1)
            if tokens[i:i + n] == folge]


def kopf_index(tokens: tuple) -> int:
    """
    Position des Kopftokens einer kurzen Nominalphrase.
    """
    kopf = len(tokens) - 1
    for i in range(1, len(tokens)):
        if tokens[i].lower() in GENITIV_MARKER:
            kopf = i - 1
            break
    #auf Satzzeichen darf der Kopf nicht fallen
    while kopf > 0 and not tokens[kopf][:1].isalnum():
        kopf -= 1
    return kopf


def bewerte_cluster(cluster: list, figuren_sequenzen: dict) -> tuple:
    """
    Bewertet die Erwaehnungen eines Clusters und waehlt den Repraesentanten.
    """
    punkte_stark = defaultdict(int)   #eigenstaendig, kopfposition, schwach
    punkte_schwach = defaultdict(int) #ausschliesslich Langspannen
    klasse_je_figur = {}

    for element in cluster:
        tokens = tuple(word_tokenize(element))
        ist_lang = len(tokens) >= LANGSPANNE_AB
        kopf = kopf_index(tokens) if tokens else -1

        for figur, folge in figuren_sequenzen.items():
            stellen = finde_teilfolge(tokens, folge)
            if not stellen:
                continue

            if tokens == folge:
                klasse = "eigenstaendig"
            elif ist_lang:
                klasse = "langspanne"
            elif any(stelle <= kopf < stelle + len(folge)
                     for stelle in stellen):
                klasse = "kopfposition"
            else:
                klasse = "schwach"

            if klasse == "langspanne":
                punkte_schwach[figur] += PUNKTE_BELEG[klasse]
            else:
                punkte_stark[figur] += PUNKTE_BELEG[klasse]

            #beste bislang gesehene Belegklasse dieser Figur merken
            bisher = klasse_je_figur.get(figur)
            if bisher is None or RANG_BELEG[klasse] > RANG_BELEG[bisher]:
                klasse_je_figur[figur] = klasse

    #Belege aus kurzen Erwaehnungen haben Vorrang
    punkte = punkte_stark or punkte_schwach
    if not punkte:
        return None, None, False, {}

    #Gleichstand alphabetisch aufloesen, damit reproduzierbar
    repraesentant = max(sorted(punkte), key=punkte.get)
    klasse = klasse_je_figur.get(repraesentant, "langspanne")
    return repraesentant, klasse, klasse in BELEG_SICHER, dict(punkte)


def vererbe_repraesentanten(cluster_daten: list, vorfenster: dict,
                            kontext_token_ende: int) -> int:
    """
    Verknuepft Cluster mit denen des Vorfensters und vereinheitlicht sie.
    """
    if not vorfenster or not vorfenster.get("cluster"):
        return 0

    versatz = vorfenster.get("tokens_gesamt", 0) - kontext_token_ende
    index_vor = {}
    for ci, c in enumerate(vorfenster["cluster"]):
        for m in c["mentions"]:
            index_vor[(m["start"], m["ende"])] = (m["text"], ci)

    uebernommen = 0
    for eintrag in cluster_daten:
        if eintrag is None:
            continue
        stimmen = defaultdict(int)
        for m in eintrag["mentions"]:
            if m["start"] >= kontext_token_ende:
                continue  #liegt im neuen Textteil, kein Gegenstueck moeglich
            treffer = index_vor.get((m["start"] + versatz,
                                     m["ende"] + versatz))
            if treffer and treffer[0] == m["text"]:
                stimmen[treffer[1]] += 1

        if not stimmen:
            continue

        #häufigste Verknüpfung entscheidet, bei Gleichstand die erste
        ci = max(sorted(stimmen), key=stimmen.get)
        vor_c = vorfenster["cluster"][ci]
        eintrag["identitaet"] = vor_c.get("identitaet", eintrag["identitaet"])
        eintrag["verknuepft_mit"] = ci

        rang_vor = RANG_BELEG.get(vor_c.get("belegklasse") or "langspanne", 0)
        rang_akt = RANG_BELEG.get(eintrag["belegklasse"] or "langspanne", 0)
        #Ein Cluster ohne eigenen Beleg erbt in jedem Fall
        ohne_eigenen_beleg = eintrag["repraesentant"] is None
        if vor_c["repraesentant"] and (
                ohne_eigenen_beleg
                or (vor_c["repraesentant"] != eintrag["repraesentant"]
                    and rang_vor >= rang_akt)):
            eintrag["repraesentant_eigen"] = eintrag["repraesentant"]
            eintrag["repraesentant"] = vor_c["repraesentant"]
            eintrag["belegklasse"] = vor_c.get("belegklasse")
            eintrag["sicher"] = bool(vor_c.get("sicher"))
            eintrag["vererbt"] = True
            uebernommen += 1

    return uebernommen


# =====================================================
# Modelle nur EINMAL laden
# =====================================================

SPRACHMODELL_POS = "de_core_news_md"

nlp_pos = spacy.load(SPRACHMODELL_POS)
nlp_ner = spacy.load("output/model-best")

model = Maverick(
    hf_name_or_path="uhhlt/droc-singletons",
    device="cuda:0"
)

encoder = model.model.encoder
encoder.config._attn_implementation = "eager"
encoder.config._attn_implementation_internal = "eager"

for layer in encoder.layers:
    attn = layer.attn
    config = attn.config
    dim = config.hidden_size // config.num_attention_heads
    base = config.rope_theta if hasattr(config, "rope_theta") else 10000

    attn.rotary_emb = ModernBertRotaryEmbedding(
        dim=dim,
        base=base,
        config=config
    )


# =====================================================
# Geräte-Fix: gesamtes Modell konsistent auf die GPU
# =====================================================

try:
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(model.model.encoder, recurse=True)
except Exception as e:
    print("Hinweis: accelerate-Hooks nicht entfernt:", e)

model.model.to("cuda:0")
model.device = "cuda:0"

# einmaliger Kontroll-Ausdruck beim Laden => beide Werte sollten 0 sein
_params_on_cpu = sum(p.device.type != "cuda" for p in model.model.parameters())
_buffers_on_cpu = sum(b.device.type != "cuda" for b in model.model.buffers())
print(f"[Geräte-Check] Parameter auf CPU: {_params_on_cpu}, Buffer auf CPU: {_buffers_on_cpu}")

model.model.eval()

_dropout_aktiv = sum(m.training for m in model.model.modules()
                     if isinstance(m, torch.nn.Dropout))
print(f"[Geräte-Check] Evaluationsmodus: {not model.model.training}, "
      f"aktive Dropout-Module: {_dropout_aktiv}")


# =====================================================
# Hilfsfunktionen
# =====================================================

def merge_figures(token_to_character: dict, figuren_in_satz: list) -> list:

    figuren_dict = {pos: text for text, pos in figuren_in_satz}

    bekannte_figuren = (
        set(token_to_character.values())
        | {text for text, pos in figuren_in_satz}
    )

    token_to_character_gefiltert = {
        pos: text
        for pos, text in token_to_character.items()
        if text in bekannte_figuren
    }

    alle_positionen = (
        set(figuren_dict.keys())
        | set(token_to_character_gefiltert.keys())
    )

    result = {}

    for pos in alle_positionen:

        in_figuren = pos in figuren_dict
        in_token = pos in token_to_character_gefiltert

        if in_figuren and in_token:
            result[pos] = token_to_character_gefiltert[pos]

        elif in_figuren:
            result[pos] = figuren_dict[pos]

        else:
            result[pos] = token_to_character_gefiltert[pos]

    return [
        (text, pos)
        for pos, text in sorted(result.items())
    ]


def build_replacement(span, representative):
    tokens = [t for t in span if not t.is_punct]

    if not tokens:
        return None, "keine_regel"

    #Fall 1: Einzeltoken-Mention
    if len(tokens) == 1:
        tag = tokens[0].tag_

        #Pronomen nach Typ differenzieren statt pauschal ersetzen
        if tag in ERSETZEN_TAGS:
            return representative, "pronomen_ersetzt"
        if tag in PROTOKOLL_TAGS:
            return None, "possessiv_protokolliert"
        if tag in BEHALTEN_TAGS:
            return None, "pronomen_behalten"

        #kein Pronomen (z. B. Rollennomen "Mädchen", Eigenname):
        #Altverhalten beibehalten, die Vereinheitlichung ist hier gewollt
        return representative, "einzeltoken_ersetzt"

    token_texts = [t.text for t in tokens]

    #Fall 2
    if representative in token_texts:
        return None, "fall2_uebersprungen"

    #Fall 3
    if len(tokens) == 2:

        det = None

        for token in tokens:
            #Begleiter über STTS statt pos_ erkennen -- docs_pos wird
            #nur mit tok2vec+tagger erzeugt, pos_ wäre dort nicht gesetzt
            if token.tag_ in ARTIKELWORT_TAGS:
                det = token.text

        if det:
            return f"{det} {representative}", "fall3_artikelwort"

    return None, "keine_regel"


# =====================================================
# Sliding Window
# =====================================================

def create_windows(text: str,
                   max_tokens: int = MAX_WINDOW_TOKENS,
                   overlap_tokens: int = OVERLAP_TOKENS) -> list:

    #Sicherheitsnetz für sehr lange Texte (Romane)
    if len(text) > nlp_pos.max_length:
        nlp_pos.max_length = len(text) + 1

    #nur Tokenizer + Parser (für Satzgrenzen) laufen lassen, spart Zeit
    with nlp_pos.select_pipes(enable=["tok2vec", "parser"]):
        doc = nlp_pos(text)

    #Segmente bilden: normalerweise ganze Sätze,
    #überlange Sätze werden notfalls hart gesplittet
    segmente = []  #Liste von (text_mit_whitespace, token_anzahl)

    for sent in doc.sents:

        if len(sent) <= max_tokens:
            segmente.append((sent.text_with_ws, len(sent)))

        else:
            for start in range(0, len(sent), max_tokens):
                teil = sent[start:start + max_tokens]
                segmente.append((teil.text_with_ws, len(teil)))

    fenster = []
    kontext_segmente = []
    i = 0

    while i < len(segmente):

        kontext_len = sum(n for _, n in kontext_segmente)

        #Kontext verkleinern, falls das nächste Segment sonst nicht passt
        while kontext_segmente and kontext_len + segmente[i][1] > max_tokens:
            kontext_len -= kontext_segmente[0][1]
            kontext_segmente = kontext_segmente[1:]

        #Chunk mit so vielen Segmenten wie möglich füllen
        chunk_segmente = []
        chunk_len = 0

        while (i < len(segmente)
               and kontext_len + chunk_len + segmente[i][1] <= max_tokens):
            chunk_segmente.append(segmente[i])
            chunk_len += segmente[i][1]
            i += 1

        #Sicherheitsnetz (sollte nie eintreten, da jedes Segment <= max_tokens)
        if not chunk_segmente:
            chunk_segmente.append(segmente[i])
            i += 1

        kontext = "".join(t for t, _ in kontext_segmente)
        chunk = "".join(t for t, _ in chunk_segmente)

        fenster.append((kontext, chunk))

        #Überlappung für das nächste Fenster:
        #die letzten ganzen Sätze des aktuellen Chunks (max. overlap_tokens)
        kontext_segmente = []
        acc = 0

        for seg in reversed(chunk_segmente):
            if acc + seg[1] > overlap_tokens:
                break
            kontext_segmente.insert(0, seg)
            acc += seg[1]

    return fenster


def _resolve_window(kontext: str, chunk: str,
                    fenster_nr: int = None,
                    protokoll: dict = None,
                    vorfenster: dict = None) -> str:
    """
    Löst Koreferenzen innerhalb EINES Fensters (kontext + chunk) auf
    und gibt nur den aufgelösten chunk zurück.
    """

    window_text = kontext + chunk
    grenze = len(kontext)  #ab diesem Zeichen-Offset beginnt der neue Teil

    docs = nlp_ner(window_text)

    with nlp_pos.select_pipes(enable=["tok2vec", "tagger"]):
        docs_pos = nlp_pos(window_text)

    tags_verfuegbar = len(docs_pos) == len(docs)
    if not tags_verfuegbar:
        print(f"[Warnung] Tokenzahl docs={len(docs)} != docs_pos="
              f"{len(docs_pos)} -- Tag-Weiche fuer dieses Fenster inaktiv")

    kontext_token_ende = sum(1 for w in docs if w.idx < grenze)

    if protokoll is not None:
        protokoll["fenster"] = fenster_nr
        protokoll["kontext_zeichen"] = len(kontext)
        protokoll["chunk_zeichen"] = len(chunk)
        protokoll["tokens_gesamt"] = len(docs)
        protokoll["kontext_token_ende"] = kontext_token_ende
        protokoll["cluster"] = []
        protokoll["cluster_ohne_figur"] = 0
        protokoll["ersetzungen"] = defaultdict(int)
        protokoll["possessiva_lokal"] = []

    figuren = {
        ent.text
        for ent in docs.ents
        if ent.label_ == "FIGUR"
    }

    if not figuren:
        if protokoll is not None:
            protokoll["anzahl_figuren"] = 0
            protokoll["ersetzungen"] = dict(protokoll["ersetzungen"])
        return chunk

    if protokoll is not None:
        protokoll["anzahl_figuren"] = len(figuren)

    coref_result = model.predict(window_text)

    figuren_sequenzen = {
        figur: tuple(word_tokenize(figur)) for figur in sorted(figuren)
    }

    cluster_daten = []

    for cluster_id, (text_cluster, offset_cluster) in enumerate(zip(
            coref_result["clusters_token_text"],
            coref_result["clusters_token_offsets"])):

        rep, klasse, sicher, punkte = bewerte_cluster(
            text_cluster, figuren_sequenzen)

        cluster_daten.append({
            "repraesentant": rep,
            "repraesentant_eigen": rep,
            "belegklasse": klasse,
            "sicher": sicher,
            "punkte": punkte,
            "vererbt": False,
            "verknuepft_mit": None,
            "identitaet": f"{fenster_nr}-{cluster_id}",
            "mentions": [
                {"text": mt, "start": s, "ende": e}
                for mt, (s, e) in zip(text_cluster, offset_cluster)
            ],
            "text_cluster": text_cluster,
            "offset_cluster": offset_cluster,
        })

    uebernommen = 0
    if VEREINHEITLICHE_REPRAESENTANTEN:
        uebernommen = vererbe_repraesentanten(cluster_daten, vorfenster,
                                              kontext_token_ende)
    if protokoll is not None:
        protokoll["repraesentanten_vererbt"] = uebernommen

    #erst jetzt verwerfen: was auch nach der Vererbung keinen Namen trägt
    cluster_daten = [e if e and e["repraesentant"] else None
                     for e in cluster_daten]

    token_to_character = {}
    possessiv_positionen = {}  #start_token -> Protokolleintrag

    for eintrag in cluster_daten:

        #Cluster ohne bekannte Figur überspringen
        if eintrag is None:
            if protokoll is not None:
                protokoll["cluster_ohne_figur"] += 1
            continue

        rep = eintrag["repraesentant"]
        text_cluster = eintrag["text_cluster"]
        offset_cluster = eintrag["offset_cluster"]

        if protokoll is not None:
            #Herkunftsfelder additiv: bestehende Auswertungen lesen weiterhin nur repräsentant und mentions
            protokoll["cluster"].append({
                "repraesentant": rep,
                "mentions": eintrag["mentions"],
                "identitaet": eintrag["identitaet"],
                "belegklasse": eintrag["belegklasse"],
                "sicher": eintrag["sicher"],
                "vererbt": eintrag["vererbt"],
                "repraesentant_eigen": eintrag["repraesentant_eigen"],
                "verknuepft_mit": eintrag["verknuepft_mit"],
            })

        if not eintrag["sicher"] and not ERSETZE_UNSICHERE_CLUSTER:
            if protokoll is not None:
                protokoll["ersetzungen"]["cluster_unsicher"] += len(
                    eintrag["mentions"])
            continue

        for mention_text, (start, end) in zip(
                text_cluster,
                offset_cluster):

            if not (0 <= start <= end < len(docs_pos)):
                if protokoll is not None:
                    protokoll["ersetzungen"]["offset_fehler"] += 1
                continue

            if tags_verfuegbar:
                replacement, aktion = build_replacement(
                    docs_pos[start:end + 1],
                    rep
                )
            else:
                #Rückfallpfad ohne Tags: Altverhalten (Einzeltoken ersetzen)
                span = docs_pos[start:end + 1]
                nicht_punkt = [t for t in span if not t.is_punct]
                if len(nicht_punkt) == 1:
                    replacement, aktion = rep, "einzeltoken_ersetzt"
                else:
                    replacement, aktion = None, "keine_regel"

            if protokoll is not None:
                protokoll["ersetzungen"][aktion] += 1

            #Possessiv-Bezug festhalten, Text bleibt unverändert
            if aktion == "possessiv_protokolliert":
                possessiv_positionen[start] = {
                    "text": mention_text,
                    "tag": docs_pos[start].tag_,
                    "repraesentant": rep,
                    "start_token": start,
                    "ende_token": end,
                    "im_kontext": docs[start].idx < grenze,
                    "zeichen_start_original_lokal": None,
                    "zeichen_start_aufgeloest_lokal": None,
                }
                continue

            #keine Ersetzung gewünscht
            if replacement is None:
                continue

            #kompletten Span durch replacement ersetzen
            for token_idx in range(start, end + 1):
                token_to_character[token_idx] = replacement

    #docs wiederverwenden statt den Text ein zweites Mal zu parsen
    figuren_in_satz = [
        (ent.text, ent.start)
        for ent in docs.ents
        if ent.label_ == "FIGUR"
    ]

    merged = merge_figures(
        token_to_character,
        figuren_in_satz
    )

    new_text = ""

    for tok, word in enumerate(docs):

        #Tokens im Kontextbereich überspringen --
        #sie wurden bereits im vorherigen Fenster ausgegeben
        if word.idx < grenze:
            continue

        #Zeichenoffsets nicht ersetzter Possessiva festhalten 
        #lokal je Fenster, die Hauptfunktion globalisiert sie anschliessend
        if tok in possessiv_positionen:
            possessiv_positionen[tok]["zeichen_start_original_lokal"] = (
                word.idx - grenze
            )
            possessiv_positionen[tok]["zeichen_start_aufgeloest_lokal"] = (
                len(new_text)
            )

        new_word = word.text

        for replacement, pos in merged:

            if tok == pos:
                new_word = replacement
                break

        new_text += new_word + word.whitespace_

    if protokoll is not None:
        protokoll["possessiva_lokal"] = list(possessiv_positionen.values())
        protokoll["ersetzungen"] = dict(protokoll["ersetzungen"])

    return new_text


# =====================================================
# Hauptfunktion
# =====================================================

def resolve_coreferences_protokolliert(
        text: str,
        max_tokens: int = MAX_WINDOW_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS):

    text = re.sub(r"\s+", " ", text).strip()

    protokoll = {
        "parameter": {
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "modell_pos": SPRACHMODELL_POS,
        },
        "fenster": [],
        "possessiva": [],
    }

    if not text:
        return text, protokoll

    fenster = create_windows(text, max_tokens, overlap_tokens)

    teile = []
    offset_original = 0    #Zeichenoffset im Eingabetext (Chunk-Summe)
    offset_aufgeloest = 0  #Zeichenoffset im Ergebnistext
    vorfenster = None      #Zustand des vorangehenden Fensters

    for nr, (kontext, chunk) in enumerate(fenster, start=1):
        print(f"--- Verarbeite Fenster {nr}/{len(fenster)} ---")

        p_fenster = {}
        #Zustand des Vorfensters durchreichen, damit die
        #Repräsentanten über die Überlappung vereinheitlicht werden
        teil = _resolve_window(kontext, chunk,
                               fenster_nr=nr, protokoll=p_fenster,
                               vorfenster=vorfenster)

        #Possessiv-Offsets von fensterlokal auf global umrechnen
        for pv in p_fenster.pop("possessiva_lokal", []):
            eintrag = dict(pv)
            eintrag["fenster"] = nr
            lok_o = eintrag.pop("zeichen_start_original_lokal")
            lok_a = eintrag.pop("zeichen_start_aufgeloest_lokal")
            eintrag["zeichen_start_original"] = (
                offset_original + lok_o if lok_o is not None else None
            )
            eintrag["zeichen_start_aufgeloest"] = (
                offset_aufgeloest + lok_a if lok_a is not None else None
            )
            protokoll["possessiva"].append(eintrag)

        protokoll["fenster"].append(p_fenster)
        vorfenster = p_fenster

        offset_original += len(chunk)
        offset_aufgeloest += len(teil)
        teile.append(teil)

    return "".join(teile), protokoll


def resolve_coreferences(text: str,
                         max_tokens: int = MAX_WINDOW_TOKENS,
                         overlap_tokens: int = OVERLAP_TOKENS) -> str:

    neuer_text, _ = resolve_coreferences_protokolliert(
        text, max_tokens, overlap_tokens
    )
    return neuer_text
