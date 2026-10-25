import unicodedata
from collections import Counter, defaultdict

VERSION = "V2 (AP4a, Ebene 1)"

ZAHLWORT_STAEMME = {
    "erst", "zweit", "dritt", "viert", "fuenft", "fünft", "sechst",
    "siebt", "siebent", "acht", "neunt", "zehnt", "elft", "zwoelft", "zwölft",
}
POSITIONS_STAEMME = {"letzt", "vorletzt", "mittler"}
ZAEHLORDINALIA = ZAHLWORT_STAEMME | POSITIONS_STAEMME

#Zurückgestellt: unterscheidet zwar Identität, trägt aber eine wertende
#Komponente ("die rechte Braut"). Nicht aktiv, der Durchsicht vorbehalten.
ZURUECKGESTELLT = {"recht"}

#Altersachse: nur im Komparativ und Superlativ ordnend. Im Positiv sind
#dieselben Lemmata qualitativ ("die alte Frau") und ausgeschlossen.
ALTERSACHSE = {"alt", "jung"}
ORDNENDE_GRADE = {"Cmp", "Sup"}

KONTRAST_PRAEFIX = "ander"

NOMEN_POS = {"NOUN", "PROPN"}
#TIGER: attributive Anbindung. 'ag' bleibt aussen vor
ATTRIBUTIV_DEPS = {"nk", "amod"}

METHODE_ORDINAL = "ordinal"
METHODE_NUMERAL = "numeral"
METHODE_KONTRAST = "kontrast"
METHODE_DEFINITHEIT = "definitheit"
METHODE_UNBESTIMMT = "unbestimmt"

METHODEN = (METHODE_ORDINAL, METHODE_NUMERAL, METHODE_KONTRAST,
            METHODE_DEFINITHEIT, METHODE_UNBESTIMMT)

INSTANZBILDENDE_KLASSEN = {"ordinal_alter", "ordinal_zahl", "kontrast"}

#Klasse -> Methodenwert
KLASSE_ZU_METHODE = {
    "ordinal_alter": METHODE_ORDINAL,
    "ordinal_zahl": METHODE_ORDINAL,
    "kontrast": METHODE_KONTRAST,
    "numeral": METHODE_NUMERAL,
}

MAX_KONTRAST_ABSTAND = 1

ZERFALL_ZAEHLT = False

SCHWELLE_APPELLATIV = 0.5


#===========================================================================
# Hilfsfunktionen
#===========================================================================
def normform(s):
    """Vereinheitlicht eine Form (wie in der Pipeline)."""
    return unicodedata.normalize("NFC", str(s)).strip().lower()


def ist_zaehlordinale(lemma):
    """
    Prüft ein Adjektivlemma gegen die geschlossene Klasse der Zählordinalia.
    Erwartet die stark flektierte Form auf '-er' ("zweiter"), deren Stamm in
    der Liste steht. Rückgabe: Stamm oder None.
    """
    l = normform(lemma)
    if not l.endswith("er"):
        return None
    stamm = l[:-2]
    return stamm if stamm in ZAEHLORDINALIA else None


def ist_zurueckgestellt(lemma):
    l = normform(lemma)
    return l.endswith("er") and l[:-2] in ZURUECKGESTELLT


def determinierer_merkmale(token):
    """
    Determinierer, Definitheit und Doppelartikel-Muster eines Nomens.
    """
    if token is None:
        return None, None, False

    artikel = [k for k in token.children
               if (k.pos_ == "DET" or k.tag_.startswith("ART")) and k.i < token.i]
    if not artikel:
        return None, None, False

    def definitheit(k):
        werte = k.morph.get("Definite")
        if werte:
            return werte[0]
        return "Def" if normform(k.lemma_) == "der" else None

    werte = [definitheit(k) for k in artikel]
    doppel = "Def" in werte and "Ind" in werte
    erster = min(artikel, key=lambda k: k.i)
    return erster.text, definitheit(erster), doppel


#===========================================================================
# Schritt 1 -- Diskriminatoren aus dem Dependenzbaum
#===========================================================================
def finde_diskriminatoren(token, versatz=0):
    """
    Diskriminatoren einer Figurenerwähnung.
    """
    treffer = []
    for k in token.children:
        if k.dep_ not in ATTRIBUTIV_DEPS:
            continue
        lemma = normform(k.lemma_)

        #--- listenfrei: Zahlwörter ---------------------------------------
        if k.pos_ == "NUM":
            treffer.append({"klasse": "numeral", "wert": lemma,
                            "beleg": k.text, "beleg_token_i": k.i - versatz})
            continue

        #--- listenminimal: Altersachse mit ordnendem Grad -----------------
        grad = k.morph.get("Degree")
        if lemma in ALTERSACHSE and grad and grad[0] in ORDNENDE_GRADE:
            treffer.append({"klasse": "ordinal_alter",
                            "wert": f"{lemma}_{grad[0].lower()}",
                            "beleg": k.text, "beleg_token_i": k.i - versatz})
            continue

        #--- listenfrei: Kontrastglied "ander" -----------------------------
        if lemma.startswith(KONTRAST_PRAEFIX):
            treffer.append({"klasse": "kontrast", "wert": "andere",
                            "beleg": k.text, "beleg_token_i": k.i - versatz})
            continue

        #--- geschlossene Klasse: Zählordinalia ---------------------------
        stamm = ist_zaehlordinale(k.lemma_) if k.tag_ == "ADJA" else None
        if stamm:
            treffer.append({"klasse": "ordinal_zahl", "wert": stamm,
                            "beleg": k.text, "beleg_token_i": k.i - versatz})

    #--- listenfrei: Doppelartikel-Muster "die eine ..." -------------------
    #Ein einfacher unbestimmter Artikel ("eine Frau") löst bewusst NICHT aus.
    _, _, doppel = determinierer_merkmale(token)
    if doppel:
        treffer.append({"klasse": "kontrast", "wert": "eine",
                        "beleg": "die eine", "beleg_token_i": token.i - versatz})

    return treffer


#===========================================================================
# Schritt 2 -- Erwähnungsfundament
#===========================================================================
def sammle_erwaehnungen(satz_docs, figurenindex, quelle=None):
    """
    Erwähnungen aller Figuren im ORIGINALTEXT.
    """
    zeilen = []
    for satz_id, doc in satz_docs:
        versatz = getattr(doc, "start", 0)   #Span: Offset im Doc; Doc: 0
        for token in doc:
            if token.pos_ not in NOMEN_POS:
                continue
            rolle = (figurenindex.get(normform(token.lemma_))
                     or figurenindex.get(normform(token.text)))
            if not rolle:
                continue

            det, definit, doppel = determinierer_merkmale(token)
            numerus = token.morph.get("Number")
            zeilen.append({
                "quelle": quelle,
                "satz_id": satz_id,
                "token_i": token.i - versatz,
                "rolle": rolle,
                "oberflaeche": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "determinierer": det,
                "definitheit": definit,
                "doppelartikel": bool(doppel),
                "numerus": numerus[0] if numerus else None,
                "diskriminatoren": finde_diskriminatoren(token, versatz),
                #werden in bilde_instanzen gesetzt
                "instanz": None,
                "instanz_methode": METHODE_UNBESTIMMT,
                "instanz_evidenz": None,
            })
    return zeilen


#===========================================================================
# Schritt 3 -- Instanzbildung
#===========================================================================
def _kontrast_bestaetigt(erwaehnungen):
    """
    Prüft die Nachbarschaftsbedingung der Kontrastformel.
    """
    kandidaten = defaultdict(list)
    for i, e in enumerate(erwaehnungen):
        for d in e["diskriminatoren"]:
            if d["klasse"] == "kontrast":
                kandidaten[e["rolle"]].append((i, e["satz_id"], d["wert"]))

    bestaetigt = set()
    for rolle, liste in kandidaten.items():
        for i, satz_i, wert_i in liste:
            for j, satz_j, wert_j in liste:
                if i == j or wert_i == wert_j:
                    continue
                if abs(satz_i - satz_j) <= MAX_KONTRAST_ABSTAND:
                    bestaetigt.add(i)
                    break
    return bestaetigt


def bilde_instanzen(erwaehnungen):
    """
    Setzt instanz, instanz_methode und instanz_evidenz je Erwähnung.
    """
    bestaetigt = _kontrast_bestaetigt(erwaehnungen)
    protokoll = {"kontrast_verworfen": [], "instanzen_je_rolle": defaultdict(set)}

    vorrang = ["ordinal_zahl", "ordinal_alter", "kontrast"]
    for i, e in enumerate(erwaehnungen):
        gewaehlt = None
        for klasse in vorrang:
            for d in e["diskriminatoren"]:
                if d["klasse"] != klasse:
                    continue
                if klasse == "kontrast" and i not in bestaetigt:
                    protokoll["kontrast_verworfen"].append({
                        "satz_id": e["satz_id"], "rolle": e["rolle"],
                        "beleg": d["beleg"],
                        "grund": "kein Gegenstueck in Nachbarschaft",
                    })
                    continue
                gewaehlt = (klasse, d)
                break
            if gewaehlt:
                break

        if not gewaehlt:
            continue  #bleibt unbestimmt -- keine stille Propagation

        klasse, d = gewaehlt
        e["instanz"] = f"{e['rolle']}#{d['wert']}"
        e["instanz_methode"] = KLASSE_ZU_METHODE[klasse]
        e["instanz_evidenz"] = {
            "klasse": klasse,
            "beleg": d["beleg"],
            "satz_id": e["satz_id"],
            "beleg_token_i": d["beleg_token_i"],
            "erwaehnung_token_i": e["token_i"],
        }
        protokoll["instanzen_je_rolle"][e["rolle"]].add(e["instanz"])

    protokoll["instanzen_je_rolle"] = {
        r: sorted(v) for r, v in protokoll["instanzen_je_rolle"].items()
    }
    return erwaehnungen, protokoll


#===========================================================================
# Schritt 4 -- Verdachtskennzeichnung 
#===========================================================================
def markiere_verdacht(erwaehnungen, zerfall_rollen=None,
                      zerfall_zaehlt=None):
    """
    Kennzeichnet je Rolle, ob sie ambiguitätsverdächtig ist.
    """
    if zerfall_zaehlt is None:
        zerfall_zaehlt = ZERFALL_ZAEHLT
    zerfall_rollen = zerfall_rollen or set()
    je_rolle = defaultdict(list)
    for e in erwaehnungen:
        je_rolle[e["rolle"]].append(e)

    befunde = {}
    for rolle, liste in je_rolle.items():
        n = len(liste)
        anteil_noun = sum(1 for e in liste if e["pos"] == "NOUN") / n
        anteil_det = sum(1 for e in liste if e["determinierer"]) / n
        appellativ = max(anteil_noun, anteil_det) >= SCHWELLE_APPELLATIV

        klassen = Counter()
        numeral_plural = 0
        for e in liste:
            for d in e["diskriminatoren"]:
                klassen[d["klasse"]] += 1
                if d["klasse"] == "numeral" and e["numerus"] == "Plur":
                    numeral_plural += 1

        stark = (numeral_plural > 0
                 or any(klassen.get(k, 0) > 0 for k in INSTANZBILDENDE_KLASSEN))
        zerfall = rolle in zerfall_rollen

        gruende = []
        if numeral_plural:
            gruende.append("numeral_plural")
        for k in sorted(INSTANZBILDENDE_KLASSEN):
            if klassen.get(k):
                gruende.append(k)
        if zerfall:
            gruende.append("coref_zerfall")

        #Der Zerfall trägt den Verdacht nur, wenn der Schalter es zulässt
        verdacht = appellativ and (stark or (zerfall and zerfall_zaehlt))

        befunde[rolle] = {
            "verdacht": bool(verdacht),
            "verdacht_zerfall": bool(appellativ and zerfall),
            "grund": "|".join(gruende) or None,
            "appellativ": bool(appellativ),
            "erwaehnungen": n,
            "anteil_noun": round(anteil_noun, 3),
            "anteil_determinierer": round(anteil_det, 3),
            "instanzen": sorted({e["instanz"] for e in liste if e["instanz"]}),
        }
    return befunde


#===========================================================================
# Schritt 5 -- Verknüpfung mit den Zuordnungszeilen
#===========================================================================
def verknuepfe_zuordnungen(zuordnungen, erwaehnungen, schluesselfeld="figur_schluessel"):
    """
    Trägt die Instanzfelder additiv in die Zuordnungszeilen ein.
    """
    nach_token = {}
    nach_text = defaultdict(list)
    for e in erwaehnungen:
        nach_token[(e["satz_id"], e["token_i"])] = e
        nach_text[(e["satz_id"], normform(e["oberflaeche"]))].append(e)

    stat = Counter()
    for z in zuordnungen:
        #Bestandsfeld unveraendert lassen, nur spiegeln
        z["figur_rolle"] = z.get("figur")
        z["figur_instanz"] = None
        z["figur_instanz_methode"] = METHODE_UNBESTIMMT
        z["figur_instanz_evidenz"] = None

        if z.get("status") != "zugeordnet":
            stat["nicht_zugeordnet"] += 1
            continue

        satz_id = z.get("satz_id")
        treffer = None

        #(1) Vorrang: Token-Anker
        tok = z.get("bezug_token_i")
        if tok is not None:
            treffer = nach_token.get((satz_id, tok))
            if treffer is not None:
                stat["ueber_token"] += 1

        #(2) Rueckfall: Textvergleich, nur wenn eindeutig
        if treffer is None:
            kandidaten = nach_text.get(
                (satz_id, normform(z.get("bezugswort") or "")), [])
            if len(kandidaten) == 1:
                treffer = kandidaten[0]
                stat["ueber_text"] += 1
            elif len(kandidaten) > 1:
                stat["mehrdeutig_unbestimmt"] += 1

        if treffer is None or not treffer["instanz"]:
            stat["unbestimmt"] += 1
            continue

        z["figur_instanz"] = treffer["instanz"]
        z["figur_instanz_methode"] = treffer["instanz_methode"]
        z["figur_instanz_evidenz"] = treffer["instanz_evidenz"]
        stat["zugewiesen"] += 1
        stat[f"methode_{treffer['instanz_methode']}"] += 1

    return dict(stat)


def markiere_zuordnungen_verdacht(zuordnungen, befunde,
                                  schluesselfeld="figur_schluessel"):
    """
    Trägt die Verdachtskennzeichnung additiv in die Zuordnungszeilen ein.
    """
    stat = Counter()
    for z in zuordnungen:
        rolle = z.get(schluesselfeld) or z.get("figur")
        befund = befunde.get(rolle)
        z["instanz_verdacht"] = bool(befund and befund["verdacht"])
        z["instanz_verdacht_zerfall"] = bool(
            befund and befund.get("verdacht_zerfall"))
        z["instanz_verdacht_grund"] = (
            befund["grund"]
            if befund and (befund["verdacht"] or befund.get("verdacht_zerfall"))
            else None)
        if z["instanz_verdacht"]:
            stat["verdacht"] += 1
        if z["instanz_verdacht_zerfall"]:
            stat["verdacht_zerfall"] += 1
        if z["instanz_verdacht_zerfall"] and not z["instanz_verdacht"]:
            stat["nur_zerfall"] += 1
    return dict(stat)


#===========================================================================
# Schritt 6 -- Plausibilitätskontrolle (Prüfung, keine Zuordnung)
#===========================================================================
def pruefe_plausibilitaet(erwaehnungen):
    """
    Stellt die Zahl der gebildeten Instanzen je Rolle der Einführungsevidenz
    gegenüber, etwa dem Numerale bei der Ersteinführung ("zwei Töchter").
    """
    je_rolle = defaultdict(lambda: {"instanzen": set(), "numerale": [],
                                    "erwaehnungen": 0})
    for e in erwaehnungen:
        eintrag = je_rolle[e["rolle"]]
        eintrag["erwaehnungen"] += 1
        if e["instanz"]:
            eintrag["instanzen"].add(e["instanz"])
        for d in e["diskriminatoren"]:
            if d["klasse"] == "numeral" and e["numerus"] == "Plur":
                eintrag["numerale"].append(
                    {"satz_id": e["satz_id"], "wert": d["wert"],
                     "beleg": d["beleg"]})

    widersprueche = []
    for rolle, eintrag in sorted(je_rolle.items()):
        if not eintrag["numerale"] or not eintrag["instanzen"]:
            continue
        #frueheste Nennung gilt als Einfuehrung
        einfuehrung = min(eintrag["numerale"], key=lambda x: x["satz_id"])
        n_instanzen = len(eintrag["instanzen"])
        widersprueche.append({
            "rolle": rolle,
            "instanzen_gebildet": n_instanzen,
            "einfuehrung_satz_id": einfuehrung["satz_id"],
            "einfuehrung_beleg": einfuehrung["beleg"],
            "instanzen": sorted(eintrag["instanzen"]),
        })
    return widersprueche


#===========================================================================
# Schritt 7 -- Instanzprofile und Abdeckungsbericht
#===========================================================================
def baue_instanzprofile(zuordnungen, lemmafeld="lemma_norm"):
    """
    Adjektivprofile je Instanz, MÄRCHENLOKAL.
    """
    profile = defaultdict(lambda: {"adjektive": Counter(), "vorkommen": [],
                                   "rolle": None, "gesamt_nennungen": 0})
    for z in zuordnungen:
        if z.get("status") != "zugeordnet" or not z.get("figur_instanz"):
            continue
        p = profile[z["figur_instanz"]]
        lemma = z.get(lemmafeld) or z.get("lemma")
        p["adjektive"][lemma] += 1
        p["rolle"] = z.get("figur_rolle") or z.get("figur")
        p["gesamt_nennungen"] += 1
        p["vorkommen"].append({
            "adjektiv": z.get("adjektiv"), "lemma": lemma,
            "satz_id": z.get("satz_id"), "methode": z.get("methode"),
            "instanz_methode": z.get("figur_instanz_methode"),
        })

    return {
        instanz: {
            "rolle": p["rolle"],
            "adjektive_haeufigkeit": [
                {"lemma": l, "anzahl": n} for l, n in p["adjektive"].most_common()
            ],
            "adjektive": sorted(p["adjektive"]),
            "gesamt_nennungen": p["gesamt_nennungen"],
            "vorkommen": p["vorkommen"],
        }
        for instanz, p in sorted(profile.items())
    }


def abdeckungsbericht(zuordnungen, schluesselfeld="figur_schluessel"):
    """
    Anteil zugewiesener gegenüber unbestimmten Zeilen, aufgeschlüsselt
    nach 'methode' sowie je Rolle und Sammlung.
    """
    zug = [z for z in zuordnungen if z.get("status") == "zugeordnet"]
    bericht = {
        "zeilen_zugeordnet": len(zug),
        "instanz_zugewiesen": sum(1 for z in zug if z.get("figur_instanz")),
        "nach_methode": {},
        "nach_instanz_methode": {},
        "je_rolle": {},
        "je_sammlung": {},
    }

    for methode in sorted({z.get("methode") for z in zug}):
        teil = [z for z in zug if z.get("methode") == methode]
        n_zu = sum(1 for z in teil if z.get("figur_instanz"))
        bericht["nach_methode"][methode] = {
            "zeilen": len(teil), "zugewiesen": n_zu,
            "unbestimmt": len(teil) - n_zu,
            "quote_prozent": round(100 * n_zu / max(len(teil), 1), 2),
            "erreichbar": methode == "original",
        }

    bericht["nach_instanz_methode"] = dict(Counter(
        z.get("figur_instanz_methode") or METHODE_UNBESTIMMT for z in zug))

    for feld, ziel in [(schluesselfeld, "je_rolle"), ("sammlung", "je_sammlung")]:
        gruppen = defaultdict(list)
        for z in zug:
            gruppen[z.get(feld) or "(ohne)"].append(z)
        bericht[ziel] = {
            str(k): {
                "zeilen": len(v),
                "zugewiesen": sum(1 for z in v if z.get("figur_instanz")),
                "quote_prozent": round(
                    100 * sum(1 for z in v if z.get("figur_instanz")) / max(len(v), 1), 2),
            }
            for k, v in sorted(gruppen.items())
        }
    return bericht


#===========================================================================
# Gesamtaufruf fuer die Pipeline
#===========================================================================
def verarbeite_instanzebene(satz_docs, figurenindex, zuordnungen,
                            quelle=None, zerfall_rollen=None,
                            schluesselfeld="figur_schluessel",
                            lemmafeld="lemma_norm",
                            zerfall_zaehlt=None):
    """
    Führt die gesamte Ebene 1 für EIN Märchen aus.
    """
    erwaehnungen = sammle_erwaehnungen(satz_docs, figurenindex, quelle)
    erwaehnungen, protokoll = bilde_instanzen(erwaehnungen)
    befunde = markiere_verdacht(erwaehnungen, zerfall_rollen,
                                zerfall_zaehlt)
    stat_verkn = verknuepfe_zuordnungen(zuordnungen, erwaehnungen, schluesselfeld)
    stat_verd = markiere_zuordnungen_verdacht(zuordnungen, befunde, schluesselfeld)

    return {
        "erwaehnungen": erwaehnungen,
        "instanz_protokoll": protokoll,
        "verdacht_befunde": befunde,
        "verknuepfung_statistik": stat_verkn,
        "verdacht_statistik": stat_verd,
        "plausibilitaet": pruefe_plausibilitaet(erwaehnungen),
        "instanzprofile": baue_instanzprofile(zuordnungen, lemmafeld),
        "abdeckung": abdeckungsbericht(zuordnungen, schluesselfeld),
    }
