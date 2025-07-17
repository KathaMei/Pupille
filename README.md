# Einführung

Die Skripte dieses GitHub-Repositoryies dienen der Aufbereitung und
Bereinigung von Pupillengrößendaten, die mit der Eyetracking Plattform Pupil
Core der Firma Pupil Labs während der Anwendung der
transkutanen aurikulären Vagusnervstimulation aufgenommen werden.

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

Das `run-jupyter.sh` Skript initialisiert eine virtuelle Python-Umgebung mit JupyterLab sowie allen benötigten Paketen aus `requirements.txt` und startet dann JupyterLab im Browser. 


# Datenablage 

`https://cloud.uol.de/s/kDQWeMb99riwwQf`

# Datenstruktur 

Die Skripte sind für die Datenstruktur der Eyetracking Plattform Pupil Core (EPPC) 
erstellt worden. Falls diese nicht für die Datenerhebung verwendet wird, sollte die
Datendateien angepasst werden. Ein Stimulationsdurchgang für einen Probanden produziert folgende Dateien:    

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

Hierbei ist PJ10_1_PLR1 die Kennung, die für den Probanden (PJ10) und den Durchlauf (1_PLR) steht. Die Dateien in diesem Ordner werden von der EPPC generiert. Dabei werden folgende Dateien verwendet: 

-   annotation_timestamps.npy: Liste mit timestamps
-   pupil_positions.csv: verwendete Spalten namens
    pupil_timestamp (Zeit der Aufnahme), eye_id (Augenseite),
    confidence (Genauigkeit der Messung), diameter (Durchmesser
    der Pupille in 2D/Pixeln), diameter_3d (Durchmesser der
    Pupille in 3D/mm)


In diesem Projekt sind insgesamt drei Ordner angelegt worden:
-   Skripte: beinhaltet drei Dateien, die für das gesamte Projekt gelten. 
Um die Eingangs- und Ausgangsdaten für alle Skripte zu konfigurieren, ist `Skripte/pup_config.py` erstellt worden. Dort können folgende Variablen gesetzt werden: 

    `data_dir` - Ort, an dem die Stimulationsdurchläufe abgelegt werden. 
    `obj_dir`  - Ort, an dem die Berechnungsergebnisse abgelegt werden. 

In der Datei `pupil_util.py` wird beschrieben, wie die Dateien zu den unterschiedlichen Stimulationsdurchgängen zugeordnet werden und wie Daten abgespeichert werden.

Das Skript `zuordnungen.csv` enthält die Zuordnung der pseudoanonymisierten Patientendaten zu den Stimulationsdurchgängen.

-   Lichtreflex: In der Studie wurden vier verschiedene Lichtstärken dreimal hintereinander wiederholt. Dies ist für jeden Probanden pro Stimulationsdurchgang einmal erfolgt.

-   Pupillendilation: Pro Probanden wurden pro Stimulationsdurchgang eine Datendatei erstellt.


# Vorbereitungen

-   Ordner Skripte:

    -   `pup_config.py`: Pfad zu den Datensätzen eingeben und Pfad, wo diese
        gespeichert werden sollen.

    -   `zuordnungen.csv`: csv-Datei erstellen, bei denen die Pseudoanonymisierung der Daten entschlüsselt wird.
            - Spalte 1 `proband`: Probanden-ID
            - Spalte 2 bis 5 `proto1` - `proto4`: die Reihenfolge der Stimulationssequenzen. Dabei bezeichnet `proto1` den ersten Durchgang, `proto2` den zweiten, usw. Eingesetzt in die Spalten pro Probanden wird eine Zahl zwischen 1 und 4, die das Stimulationsprotokoll angibt. `1 = 3.4s Stimulation`, `2 = 3.4s Placebo`, `3 = 30s Stimulation`, `4 = 30s Placebo`
       Beispielsweise bedeuten hierbei die Zahlen in der ersten Zeile des Skriptes von PJ01 2, 3, 1, 4, dass zuerst als Stimulationsprotokoll 3.4s Placebo, anschließend  30s Stimulation, dann eine 3.4s Stimulation und zuletzt eine 30s Placebo Messung durchgeführt wurden.



| Proband | Kennung | Proto1 | Proto2 | Proto3 | Proto4 |
|---------|---------|--------|--------|--------|--------|
| 1       | PJ01    | 2      | 3      | 1      | 4      |
| 2       | PJ02    | 1      | 2      | 4      | 3      |
  
Die Zuordnungen der Stimulationsdurchgänge zu den Zahlen wird in `pup_util.py` definiert. Dies Zuordnungen können folgendermaßen überprüft werden: 

```
$ python3
Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import pup_util
>>> pup_util.get_condition("PJ01_1")
('3.4', '3.4Placebo')
>>> pup_util.get_condition("PJ02_1")
('3.4', '3.4Stim')
>>> pup_util.get_condition("PJ02_2")
('3.4', '3.4Placebo')
>>> pup_util.get_condition("PJ02_3")
('30', '30Placebo')
>>> pup_util.get_condition("PJ02_4")
('30', '30Stim')
```

        
# Anwendungen

## Pupillendilation

- Mit dem Notebook namens `smooth.ipynb` können zuerst die Einstellungen des Hauptskripts `preprocessing.py` überprüft und die Schritte der Datenvorverarbeitung visualisiert werden. Die Probanden-ID, von der die Daten überprüft werden sollen, wird unter `Creation of variables` eingegeben. Ebenso wird unter dem Namen `field` angegeben, ob der `diameter` oder `diameter_3d` betrachtet werden soll. 


`subject_id="PJ21_4_Ruhe"
field="diameter"
ts="pupil_timestamp"
data_dir=pup_config.data_dir`

- Ebenso kann auch die Augenseite, dessen Daten betrachtet werden sollen, umgestellt werden. `0 - rechts`, `1 - links`

`config=preprocessing.create_process_config(0,field,subject_id,data_dir)`

- Die Parameter und Grenzwerte sämtlicher Funktionen im Skript `preprocessing.py` können verstellt werden und anschließend in dem Notebook `smooth.ipynb` für einzelne Probandendurchläufe überprüft werden. Erklärungen zu den einzelnen Parametern und Funktionen sind im Skript selbst enthalten.
  
- Wenn die Einstellungen der Datenvorverarbeitung für sämtliche Probanden passend erscheinen, kann für alle einzelnen Replikate über das Skript `validate_data.py` der `nan_score` abgebildet werden. Der `nan_score` gibt an, wie viele der Daten durch die Funktion `compute-and-reject-noise` ausgeschlossen und für die weitere Analyse nicht berücksicht werden.
`
 ./run-validate.sh /Users/Katharina/Desktop/Beispieldaten
`
- Anschließend können einzelne Datensätze im Notebook `smooth.ipynb` in ihrer Qualität beurteilt werden und der Grenzwert `noise-rejection-percent` so angepasst werden, dass die Datensätze mit dem nan-score über dem Grenzwert, also mit niedriger Qualität, von den weiteren Schritten ausgeschlossen werden. Die Datensätze, die den Grenzwert nicht überschreiten, durchlaufen die weiteren Vorverarbeitungsschritte.

- Wenn die Einstellungen der Funktionen überprüft worden sind, können die Ergebnisse aller Daten extrahiert werden. Dafür wird `batch_pp_subject.py` genutzt, worüber die Daten sowohl als `pickle` als auch als csv-Datei extern gespeichert werden.
`
./for-all-data.sh ~/Desktop/Beispieldaten ./batch_pp_subject.py 
`
- Im letzten Schritt kann dann überprüft werden, in welchen Abschnitten der Datenvorverarbeitung wie viele Replikate unter den Grenzwerten lagen und somit für die weiteren Schritte entfernt worden sind. Dafür wird das Skript `counts_fails.py` genutzt.
`
python3 /Users/Katharina/Desktop/Pupille/notebooks/count_fails.py /Users/Katharina/Desktop/Schreibtisch - MacBook Air/Forschungs- und Doktorarbeit Witt/Promotion/Datenauswertung/Pupillendilation/ErgebnissePreprocessing/*.pickle
`
Als Resultat erhält man zwei csv-Dateien, welche die Anzahl an Dateien pro Schritt angeben.
Beispielhaft wird folgende Tabelle dargestellt:

| Eyenum | Column   | Stage      | Data Count |
|--------|----------|------------|------------|
| 0      | diameter | finished   | 2241       |
| 0      | diameter | preprocess | 180        |


In der csv-Datei gibt `eyenum` die Augenseite (links oder rechts), `column` die Daten (diameter oder daimater_3d), `data count` die Anzahl der Daten und `stage` den Schritt, den die Daten als Letztes erreicht haben und anschließend ausgeschieden sind.


## Lichtreflex

- Mit dem Notebook namens `PLR-Blinkreconstruct.ipynb` können zuerst die Einstellungen des Hauptskripts `lr_preprocessing.py` überprüft und die Schritte der Datenvorverarbeitung visualisiert werden. Die Probanden-ID, von der die Daten überprüft werden sollen, wird unter `Check the application of the module checkdataPLR.py on specific subject data` oder `Check the application of the module checkdataPLR.py on specific subject data in a list` eingegeben.
- Anschließend wird entweder die Variable `diameter` oder `diameter_3d` unter `Use datamatrix from pydatamatrix.eu to detect and reconstruct blinks for diameter. Use the list of subjects and create plots of the different preprocessing steps.` eingefügt. Anschließend laden sich für alle Datensätze Grafiken, mit denen visuell die Anwendung der Vorverarbeitungsschritte überprüft werden können.
- Datensätze, bei denen durch die Anwendung keine Qualitätsverbesserung der Daten erzielt wird und die infolge von Messfehlern als invalide gelten, sollten dokumentiert und anschließend von den weiteren Analysen ausgeschlossen werden.
-  Die Funktionen, mit denen die Lichtreflexparameter errechnet werden, sind im Skript `classPLRfromGithub.py` enthalten. Um die Ergebnisse für alle Datensätze in Form von csv-Dateien zu erhalten, wird das Skript `batch_lr_pp_subject.py` genutzt und so die Dateien extern abgespeichert. Es werden automatisch vier Dateien pro Datensatz erstellt: jeweils für die Kombinationen der beiden Augenseiten und `diameter`/ `diameter_3d`. 
- `./for-all-data.sh ~/Desktop/Beispieldaten ./batch_lr_pp_subject.py`

