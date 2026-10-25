from collections import Counter, defaultdict

ENDUNGS_MINDESTBELEGE = 5

UMLAUT_MAX_BELEGE = 3

_UMLAUTE = str.maketrans({"ä": "a", "ö": "o", "ü": "u"})


def _umlautfrei(wort):
    return wort.replace("äu", "au").translate(_UMLAUTE)


def _cistem_basis(wort, ge_strippen=True):
    if len(wort) == 0:
        return wort
    upper = wort[0].isupper()
    wort = wort.lower()
    wort = wort.replace("ü", "u").replace("ö", "o").replace("ä", "a")
    wort = wort.replace("ß", "ss")
    if ge_strippen and wort.startswith("ge") and len(wort) >= 6:
        wort = wort[2:]
    wort = wort.replace("sch", "$")
    wort = wort.replace("ei", "%")
    wort = wort.replace("ie", "&")
    wort = "".join(
        z if i < 2 or z != wort[i - 1] else "*" for i, z in enumerate(wort))
    while len(wort) > 3:
        if len(wort) > 5 and (wort.endswith("em") or wort.endswith("er")
                              or wort.endswith("nd")):
            wort = wort[:-2]
        elif (not upper and wort.endswith("t")) or wort.endswith("e") \
                or wort.endswith("s") or wort.endswith("n"):
            wort = wort[:-1]
        else:
            break
    wort = "".join(
        z if z != "*" else wort[i - 1] for i, z in enumerate(wort))
    wort = wort.replace("$", "sch").replace("%", "ei").replace("&", "ie")
    return wort


def _eigener_cistem(wort):
    return _cistem_basis(wort, ge_strippen=True)


def _cistem_ohne_ge(wort):
    return _cistem_basis(wort, ge_strippen=False)


try:
    from nltk.stem.cistem import Cistem as _NltkCistem
    _NLTK_STEMMER = _NltkCistem()
    CISTEM_QUELLE = "nltk"

    def _cistem(wort):
        return _NLTK_STEMMER.stem(wort)
except Exception:
    _NLTK_STEMMER = None
    CISTEM_QUELLE = "eingebaut"
    _cistem = _eigener_cistem


#---------------------------------------------------------------------------
#Signale und Sperren
#---------------------------------------------------------------------------
def leite_endungsinventar_ab(paare_textform_lemma, mindestbelege=ENDUNGS_MINDESTBELEGE):
    #Leitet das Flexionsendungsinventar empirisch aus (Textform, Lemma)-Paaren ab
    reste = Counter()
    for textform, lemma in paare_textform_lemma:
        t, l = textform.lower(), lemma.lower()
        if t != l and t.startswith(l):
            reste[t[len(l):]] += 1
    return {rest: anzahl for rest, anzahl in reste.items()
            if anzahl >= mindestbelege}


def _praefix_mit_endung(a, b, endungen):
    for x, y in ((a, b), (b, a)):
        if y.startswith(x) and y[len(x):] in endungen:
            return True
    return False


def _ist_flexion(form, kandidaten, endungen, e_ausnahme=False):
    #Laesst sich 'form' als anderes Element plus Flexionsendung darstellen?

    for anderer in kandidaten:
        if anderer == form or not form.startswith(anderer):
            continue
        rest = form[len(anderer):]
        if rest in endungen and not (e_ausnahme and rest == "e"):
            return True
    return False


def _gesperrt(a, b, stamm):
    #Grammatische Lexemgrenzen: ge-Praefixbildung und Partizip I auf -end
    for x, y in ((a, b), (b, a)):
        if y.startswith("ge") and not x.startswith("ge") \
                and (y[2:] == x or stamm(y[2:]) == stamm(x)):
            return "ge_praefix"
    for x, y in ((a, b), (b, a)):
        if y.endswith("end") and not x.startswith(y[:-1]) and not x.startswith(y):
            return "partizip_i"
    return None


def _signale(a, b, endungen, kollisionspaare, belege, stamm):
    s = set()
    if frozenset((a, b)) in kollisionspaare:
        s.add("A")
    if _praefix_mit_endung(a, b, endungen):
        s.add("B")
    if stamm(a) == stamm(b) or _cistem_ohne_ge(a) == _cistem_ohne_ge(b):
        s.add("C")
    ua, ub = _umlautfrei(a), _umlautfrei(b)
    if (a, b) != (ua, ub) and a != b:
        if ua == ub:
            if min(belege.get(a, 0), belege.get(b, 0)) <= UMLAUT_MAX_BELEGE:
                s.add("D")
        elif _praefix_mit_endung(ua, ub, endungen):
            s.add("D")
    return s


#---------------------------------------------------------------------------
#Hauptfunktion
#---------------------------------------------------------------------------
def normalisiere_adjektive(zuordnungen, status_filter="zugeordnet",
                           endungs_mindestbelege=ENDUNGS_MINDESTBELEGE):
    #Bildet die Normalisierungskarte aus den Zuordnungszeilen eines Laufs.
    zeilen = [z for z in zuordnungen
              if (status_filter is None or z.get("status") == status_filter)
              and z.get("lemma")]

    belege = Counter(z["lemma"].lower() for z in zeilen)
    formen_je_lemma = defaultdict(Counter)
    praedikativ_je_lemma = defaultdict(Counter)
    for z in zeilen:
        lemma = z["lemma"].lower()
        form = (z.get("adjektiv") or "").lower()
        if form:
            formen_je_lemma[lemma][form] += 1
            konstruktion = z.get("konstruktion") or ""
            if konstruktion.startswith("praedikativ"):
                praedikativ_je_lemma[lemma][form] += 1

    vokabular = sorted(belege)

    #Signal A: dieselbe Textform erzeugt verschiedene Lemmata
    lemmata_je_form = defaultdict(set)
    for lemma, formen in formen_je_lemma.items():
        for form in formen:
            lemmata_je_form[form].add(lemma)
    kollisionspaare = {
        frozenset((a, b))
        for form, lemmata in lemmata_je_form.items() if len(lemmata) > 1
        for a in lemmata for b in lemmata if a < b
    }

    #Signal B: Endungsinventar empirisch aus diesem Lauf
    endungen = leite_endungsinventar_ab(
        ((form, lemma) for lemma, formen in formen_je_lemma.items()
         for form, n in formen.items() for _ in range(n)),
        mindestbelege=endungs_mindestbelege)

    #Cistem-Staemme einmal berechnen
    stammkarte = {lemma: _cistem(lemma) for lemma in vokabular}

    def stamm(wort):
        ergebnis = stammkarte.get(wort)
        if ergebnis is None:
            ergebnis = stammkarte[wort] = _cistem(wort)
        return ergebnis

    je_stamm = defaultdict(list)
    for lemma in vokabular:
        je_stamm[stammkarte[lemma]].append(lemma)

    #Kandidatenpaare: Kollisionen, Stammklassen und Umlautgleiche verbinden
    umlaut_je_form = defaultdict(list)
    for lemma in vokabular:
        umlaut_je_form[_umlautfrei(lemma)].append(lemma)
    kandidaten = set(kollisionspaare)
    for gruppe in list(je_stamm.values()) + list(umlaut_je_form.values()):
        for i, a in enumerate(gruppe):
            for b in gruppe[i + 1:]:
                kandidaten.add(frozenset((a, b)))
    #Signal B kann auch ohne Stammgleichheit stehen ('anderer'/'anderr' hat denselben Stamm
    nach_anfang = defaultdict(list)
    for lemma in vokabular:
        nach_anfang[lemma[:3]].append(lemma)
    for gruppe in nach_anfang.values():
        for i, a in enumerate(gruppe):
            for b in gruppe[i + 1:]:
                if _praefix_mit_endung(a, b, endungen):
                    kandidaten.add(frozenset((a, b)))

    #Vereinigung per Union-Find
    eltern = {lemma: lemma for lemma in vokabular}

    def finde(x):
        while eltern[x] != x:
            eltern[x] = eltern[eltern[x]]
            x = eltern[x]
        return x

    paare_vereinigt = []
    paare_gesperrt = []
    for paar in sorted(kandidaten, key=sorted):
        a, b = sorted(paar)
        s = _signale(a, b, endungen, kollisionspaare, belege, stamm)
        if not (("A" in s) or len(s) >= 2):
            continue
        grund = _gesperrt(a, b, stamm)
        if grund:
            paare_gesperrt.append(
                {"a": a, "b": b, "signale": sorted(s), "sperre": grund})
            continue
        paare_vereinigt.append({"a": a, "b": b, "signale": sorted(s)})
        wa, wb = finde(a), finde(b)
        if wa != wb:
            eltern[max(wa, wb)] = min(wa, wb)

    klassen_mitglieder = defaultdict(list)
    for lemma in vokabular:
        klassen_mitglieder[finde(lemma)].append(lemma)

    #Normformwahl je Klasse
    def waehle_normform(mitglieder):
        #Deterministische Wahl der Normform
        praedikativ = Counter()
        formen = Counter()
        for lemma in mitglieder:
            praedikativ.update(praedikativ_je_lemma.get(lemma, {}))
            formen.update(formen_je_lemma.get(lemma, {}))

        stufe1 = {f: n for f, n in praedikativ.items()
                  if f in mitglieder
                  and not _ist_flexion(f, mitglieder, endungen, e_ausnahme=True)}
        if stufe1:
            beste = max(stufe1.items(), key=lambda p: (p[1], -len(p[0]), p[0]))
            return beste[0], "praedikative_grundform"

        basis = {m for m in mitglieder
                 if not _ist_flexion(m, mitglieder, endungen)} or set(mitglieder)
        stufe2 = {f: n for f, n in formen.items() if f in basis}
        if stufe2:
            beste = max(stufe2.items(), key=lambda p: (p[1], -len(p[0]), p[0]))
            return beste[0], "basisform_nach_haeufigkeit"

        beste = max(basis, key=lambda l: (belege[l], l))
        return beste, "belegstaerkste_basisform"

    karte = {}
    klassen = []
    signale_je_lemma = defaultdict(set)
    for eintrag in paare_vereinigt:
        signale_je_lemma[eintrag["a"]].update(eintrag["signale"])
        signale_je_lemma[eintrag["b"]].update(eintrag["signale"])
    for wurzel, mitglieder in sorted(klassen_mitglieder.items()):
        if len(mitglieder) == 1:
            continue
        normform, normform_regel = waehle_normform(mitglieder)
        for lemma in mitglieder:
            if lemma != normform:
                karte[lemma] = normform
        klassen.append({
            "normform": normform,
            "normform_regel": normform_regel,
            "mitglieder": [
                {"lemma": lemma, "belege": belege[lemma],
                 "signale": sorted(signale_je_lemma.get(lemma, set()))}
                for lemma in sorted(mitglieder, key=lambda l: (-belege[l], l))
            ],
            "belege": sum(belege[l] for l in mitglieder),
        })
    klassen.sort(key=lambda k: (-k["belege"], k["normform"]))

    meta = {
        "verfahren": "regelbasiert",
        "cistem_quelle": CISTEM_QUELLE,
        "parameter": {
            "status_filter": status_filter,
            "endungs_mindestbelege": endungs_mindestbelege,
            "umlaut_max_belege": UMLAUT_MAX_BELEGE,
            "vereinigung": "Signal A allein oder mindestens zwei aus B, C, D",
            "cistem_adjektivvariante": ("Stammvergleich zusaetzlich ohne "
                                        "ge-Stripping, da die ge-Abgrenzung "
                                        "die Sperre uebernimmt"),
            "sperren": ["ge_praefix", "partizip_i"],
        },
        "endungsinventar": dict(sorted(endungen.items(),
                                       key=lambda p: (-p[1], p[0]))),
        "vokabular": len(vokabular),
        "klassen": len(klassen),
        "regeln": len(karte),
        "paare_vereinigt": len(paare_vereinigt),
        "paare_gesperrt": len(paare_gesperrt),
    }
    return {
        "karte": karte,
        "klassen": klassen,
        "paare_vereinigt": paare_vereinigt,
        "paare_gesperrt": paare_gesperrt,
        "meta": meta,
    }
