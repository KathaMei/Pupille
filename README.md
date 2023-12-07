# Einführung

Die Skripte dieses GitHub-Repositorys dienen der Aufbereitung und
Bereinigung von Pupillengrößen Daten, die mit dem Eye Trackern Pupil
Core von Pupil Labs aufgenommen wurden, während der Anwendung der
transkutanen aurikulären Vagusnervstimulation.

Ein Schwerpunkt liegt auf der Aufbereitung von Pupillendaten und der
Berechnung von Parametern des Pupillenlichtreflexes. Vorverarbeitet und
berechnet werden die Daten während und nach der Vagusnervstimulation. In
diesem Skript werden Daten von 3.4s sowie 30s langen
Stimulationssequenzen sowie Verum- und Shamstimulationen prozessiert.

# Voraussetzungen

Kurzfassung: 

```console
$ git clone https://github.com/KathaMei/Pupille.git
$ cd Pupille
$ ./run-jupyter.sh
```

Das `run-jupyter.sh` Script initialisiert eine virtuelle Python Umgebung mit JupyterLab sowie allen benötigten Paketen aus `requirements.txt` und startet dann JupyerLab im Browser. 

# [TODO] Wie kommt man an meine konkreten Messdaten? 

# Datenstruktur 

Die Skripte sind für die Datenstruktur des Pupil Core Eyetrackers (PCE) 
erstellt worden. Falls dieser nicht benutzt wurde, sollte die
Datendateien angepasst werden. Ein Stimulationsdurchgang für einen Probanden würde beispielsweise folgende Dateien produzieren:    

```
PJ10/PJ10_1_PLR1/
PJ10/PJ10_1_PLR1/exports
PJ10/PJ10_1_PLR1/exports/000
PJ10/PJ10_1_PLR1/exports/000/pupil_gaze_positions_info.txt
PJ10/PJ10_1_PLR1/exports/000/export_info.csv
PJ10/PJ10_1_PLR1/exports/000/gaze_positions.csv
PJ10/PJ10_1_PLR1/exports/000/pupil_positions.csv
PJ10/PJ10_1_PLR1/exports/000/annotations.csv
PJ10/PJ10_1_PLR1/annotation_timestamps.npy
```

Hierbei is PJ10_1_PLR1 ein Bezeichner, der für den Probanden (PJ10) und den Durchlauf (1_PLR) steht. Die Dateien in diesem Ordner werden vom PCE generiert. Dabei werden folgende Dateien verwendet: 

-   annotation_timestamps.npy: List mit Timestamps
-   pupil_positions.csv: verwendete Spalten namens
    pupil_timestamp (Zeit der Aufnahme), eye_id (Augenseite),
    confidence (Genauigkeit der Messung), diameter (Durchmesser
    der Pupille in 2D/Pixeln), diameter_3d (Durchmesser der
    Pupille in 3D/mm)

Die Skripte in diesem Projekt erhalten der Ort der Eingangs und Ausgangsdaten aus der Konfigurationsdatei `Skripte/pup_config.py`. Dort können folgende Variablen gesetzt werden: 

    `data_dir` - Ort an dem die Simulationsdurchläufe abgelegt wurden. 
    `obj_dir`  - Ort an dem die Berechnungsergebnisse abgelegt werden sollen. 
    
    -   Lichtreflex: In diesem Fall wurden vier verschiedene
        Lichtstärken dreimal hintereinander wiederholt. Dies ist für
        jeden Probanden pro Stimulationsdurchgang einmal erfolgt.

    -   Pupillendilation: Pro Probanden wurden pro Stimulationsdurchgang
        eine Datendatei erstellt.

# Vorbereitungen

-   Datenbenennung überprüfen, die Skripte namens „for-all-data.sh,
    run-validata.sh" in Datei Pupillendilation und Lichtreflex
    entsprechend anpassen oder folgende Dateibenennung nutzen.

    -   Lichtreflex

        -   Order: PJXX

        -   Datei: PJXX_X_PLR1, PJXX_X_PLR2, PJXX_X_PLR3

    -   Pupillendilation:

        -   Order: PJXX

        -   Datei: PJXX_X_Ruhe

    -   In dem Unterordnern wird annotations.npy und
        exports/000/pupil_positions.csv verwendet

        -   Annotations.npy: Liste mit timestamps

        -   Pupil_positions.csv: verwendete Spalten namens
            pupil_timestamp, eye_id, confidence, diameter, diameter_3d

[Idee dafür Beispiele hochladen zum Nachvollziehen?]{.mark}

-   Ordner Skripte:

    -   `pup_config.py`: Pfad zu den Datensätzen eingeben und Pfad, wo diese
        gespeichert werden sollen

    -   `zuordnungen.csv`: csv Datei erstellen, bei denen die
        Pseudoanonymisierung der Daten entschlüsselt wird
        [TODO] Erklärung, was die Zahlen bedeuten in der zuodnung.csv, z.B. `PJ01,2,3,1,4`

# Anwendungen

## Pupillendilation

-   anpassen der Zeitdauer, mit der die Stimulationen durchgeführt
    wurden im Skript `preprocessing.py`

-   testen der Vorverarbeitung mithilfe des Notebooks `smooth.ipynb`

-   Anpassung der Grenzwerte der Funktionen im Skript im
    `preprocessing.py`

-   Ausgabe der Anzahl an Datensätzen, die weiterhin valide waren
    nach der Vorverarbeitung mittels `validate_data.py` und
    `counts_fails.py`

-   Durchlauf und speichern der vorprozessierten Daten mithilfe des
    batch_pp_subject.py

## Lichtreflex

-   anpassen der Zeitdauer, mit der die Stimulationen durchgeführt
    wurden im Skript lr_preprocessing.py

-   testen der Vorverarbeitung mithilfe des Notebooks
    PLR-Blinkreconstruct.ipynb

-   Anpassung der Grenzwerte der Funktionen im Skript im
    lr_preprocessing.py

-   Durchlauf und speichern der vorprozessierten Daten mithilfe des
    batch_lr_pp_subject.py

# Dateien

-   Pupillendilation: enthält Skripte zur Aufbereitung der
    Pupillendaten, Notebooks zum Testen der Funktionen sowie Skripte,
    die die aufbereiteten Daten exportieren.

    -   Batch_pp_subject.py

        -   Speichert Daten als pickle und csv Dateien.

    -   Checkdata.ipynb

        -   Testet die Anwendung der Blinzelfunktion an einzelnen
            Datensätzen.

    -   Checkdata.py

        -   Die Klasse DataConfig für die Anwendung des Hauptskripts
            preprocessing wird definiert.

    -   Count_fails.py

        -   Errechnet für jeden Schritt der Datenvorverarbeitung wie
            viele Replikate / Datensätze noch enthalten sind.

    -   For-all-data.sh

        -   Bash-Skript zum Auffinden aller Datensätze.

    -   Plotting.py

        -   Definition zum Erstellen von Grafiken.

    -   Preprocessing.py

        -   Hauptskript: Datenvorverarbeitung und Berechnung.

    -   Run-validate.sh

        -   Bash-Skript, welches die Dateien für das validata_data.py
            Skript findet.

    -   Show_pickle.py

        -   Überprüfung, wie viele pickle Dateien als Resultat erstellt
            wurden und wie viele Replikate welche Stufe der
            Datenvorverarbeitung erreicht haben.

    -   Smooth.ipynb

        -   Notebook zur Visualisierung der verschiedenen Schritte der
            Datenvorverarbeitung des preprocessing Skripts.

    -   Validate_data.py

        -   CSV-Dateien werden erstellt, anhand dessen die am besten
            passenden Grenzwerte für die Funktionen des preprocessing
            Skriptes gefunden werden können.

-   Lichtreflex: enthält Skripte zur Aufbereitung der Pupillendaten, zur
    Errechnung von Parametern des Pupillenlichtreflexes, Notebooks zum
    Testen der Funktionen sowie Skripte, die die aufbereiteten Daten
    exportieren.

    -   Batch_lr_pp_subject.py

        -   Speichert Daten als pickle und csv Dateien.

    -   CheckdataPLR.py

        -   Die Klasse DataConfig für die Anwendung des Hauptskripts
            lr_preprocessing wird definiert.

    -   ClassPLRfromGitHub.py

        -   Die Klasse PLR zur Errechnung der Lichtreflexparameter wird
            erstellt.

    -   For-all-data.sh

        -   Bash-Skript zum Auffinden aller Datensätze.

    -   Lr_preprocessing.py

        -   Hauptskript: Datenvorverarbeitung und Berechnung.

    -   PLR-Blinkreconstruct.ipynb

        -   Notebook zur Visualisierung der verschiedenen Schritte der
            Datenvorverarbeitung des lr_preprocessing Skripts.

-   Skripte:

    -   Pup_config.py

        -   Pfade zu den Dateien, die eingelesen werden sollen und zu
            den Speicherorten.

    -   Pup_util.py

        -   Definitionen zur Zuteilung der Zuordnungen, speichern und
            laden von pickle Dateien.

    -   Zuordnungen

        -   Zuordnungen der Dateinamen zu den verschiedenen Konditionen
            der Vagusnervstimulation (3.4s oder 30s, Stimulation oder
            Placebo)
