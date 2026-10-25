# Aufteilungsplan — verbindliche Zuordnung vor Annotationsbeginn

## 1 Personen

| Kürzel | Name | Aufgabe | Eingewiesen am |
|---|---|---|---|
| A1 | [Charly] | Figuren, danach Adjudikation Pronomen | [27.07.2026] |
| A2 | [Alice] | ausschließlich Pronomen-IAA | [27.07.2026] |
| A3 | [Conny] | Figuren | [27.07.2026] |
| A4 | [Timucin] | Wortarten, danach Pronomen-Haupterhebung | [28.07.2026] |
| A5 | [Ned] | Wortarten, danach Pronomen-IAA | [28.07.2026] |
| A6 | [Geli] | Wortarten | [28.07.2026] |
| A7 | [Kevin] | Figuren | [27.07.2026] |
| A8 | [Laura] | Figuren | [27.07.2026] |


## 2 Figurenebene (NER)

Alle vier Personen bearbeiten **das vollständige Material** unabhängig;
eine gesonderte IAA-Teilmenge entfällt, weil die Übereinstimmung auf
allen Sätzen bestimmt wird.

| Tranche | Umfang | Bearbeitende | Ausgabe am | Abgabe bis |
|---|---|---|---|---|
| 1 | 75 Sätze, 1794 Token | A1, A3, A7, A8 | [27.07.2026] | [09.08.2026] |
| 2 | 75 Sätze, 2343 Token | A1, A3, A7, A8 | [13.08.2026] | [29.08.2026] |


## 3 Wortartenebene (POS)

Ebenfalls vollständige Dreifachannotation, blind ohne Vorbelegung.

| Tranche | Umfang | Bearbeitende | Ausgabe am | Abgabe bis |
|---|---|---|---|---|
| 1 | 1794 Token | A4, A5, A6 | [28.07.2026] | [17.08.2026] |
| 2 | 2343 Token | A4, A5, A6 | offen | offen |

**Ebenentrennung:** Keine Person der Wortartengruppe bearbeitet Sätze auf
der Figurenebene und umgekehrt.

## 4 Pronomenebene

| Rolle | Person | Umfang | Ausgabe am | Abgabe bis |
|---|---|---|---|---|
| Haupterhebung (= Gold) | A4 | 248 Einheiten | [18.08.2026] | [02.09.2026] |
| IAA-Zweitbearbeitung | A2 | 49 Einheiten | [27.07.2026] | [21.09.2026] |
| IAA-Drittbearbeitung | A5 | 49 Einheiten | [18.08.2026] | [19.08.2026] |

### IAA-Teilmenge — hier eintragen, vor Annotationsbeginn

**Woher die Werte stammen:** `pronomen_zu_excel.ipynb` gibt sie in
**Schritt 1** aus.

| Sammlung | Erzählung — Wert aus der Spalte `erzaehlung` | Excel-Zeilen | Einheiten |
|---|---|---|---|
| Andersen | [z. B. `2_Des Kaisers neue Kleider`] | [19]–[29] | [11] |
| Arabisch | [`0_Die Geschichte von Ali dem Kairenser und dem spukenden Hause in Bagdad`] | [85]–[96] | [12] |
| Grimm | [`1_Die drei Spinnerinnen`] | [159]–[172] | [14] |
| Japanisch | [`3_Das Mädchen mit dem Holznapfe`] | [187]–[198] | [12] |
| | **Summe** | | **[49]** |

Gemeint ist die vollständige Zeichenkette aus der Spalte `erzaehlung`,
also Präfixziffer, Unterstrich und Titel, etwa `1_Die drei Spinnerinnen`.
Sie ist der Schlüssel, über den sämtliche Werkzeuge die Erzählungen
zuordnen; Abkürzungen oder abweichende Schreibweisen führen dazu, dass
die Auswahl nicht wiedergefunden wird. Zu übernehmen ist der Wert genau
so, wie ihn `pronomen_zu_excel.ipynb` in Schritt 1 ausgibt.

Eingetragen am: [27.07.2026]