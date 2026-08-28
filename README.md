# Image to Patterned Roller Generator

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Trimesh](https://img.shields.io/badge/Trimesh-powerful_3D_mesh_library-f06529.svg?style=for-the-badge)

![Banner](https://via.placeholder.com/1200x300.png?text=Image+to+Patterned+Roller+Generator)

A Streamlit web application to convert any 2D image into a 3D-printable patterned roller. Use it to create custom textures for applying paint, stamping clay, or making impressions in other materials.

## ✨ Features

- **Image Upload:** Upload any pattern or texture image (`.png`, `.jpg`, `.bmp`).
- **Real-time 3D Preview:** See a live preview of your patterned roller as you adjust the settings.
- **Customizable Parameters:**
    - **Radius:** Control the overall size of the roller.
    - **Pattern Depth:** Adjust the depth of the pattern based on image brightness.
    - **Resolution (DPI):** Define the output resolution for the 3D model.
    - **Axis Hole:** Optionally create a central hole for an axle.
- **STL Export:** Download the final model as an `.stl` file, ready for any 3D slicer.
- **Watertight Mesh:** The generated mesh is processed to be watertight, ensuring high printability.

## 🚀 How to Use the Application

1.  **Upload an Image:** Use the file uploader in the sidebar to select an image.
2.  **Adjust Settings:** Use the sliders and checkboxes in the sidebar to configure the model to your liking.
3.  **Generate Model:** Click the "Generate 3D Model" button.
4.  **Preview:** Interact with the 3D model in the preview pane.
5.  **Download:** Once you are happy with the result, click the "Download Model as STL" button.

## 🛠️ How to Run Locally

To run this application on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    Make sure you have Python 3.8+ installed. Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    The application should now be open in your web browser.

## ⚙️ How it Works

### Einzel-Zylinder (Lithophane-Roller)

1.  **Image Processing:** The input image is converted to grayscale.
2.  **Vertex Mapping:** The application iterates through each pixel of the image. The pixel's position is mapped to a 3D coordinate on a cylinder's surface, and its brightness determines the radial displacement (pattern depth).
3.  **Mesh Generation:** The `trimesh` library is used to create a 3D mesh from these vertices.
4.  **STL Export:** The final `trimesh` object is exported to a binary STL format.

### Zweiteiler: Schneide + Ausstoesser (Voreinstellung)

Der Roller wird als **zwei ineinander gedruckte Koerper** erzeugt
(`gyroid_coexistence.py`): die dunklen Linien des Bildes werden zur
**Schneide**, die hellen Flaechen dazwischen zum **Ausstoesser**, der das
geschnittene Teil wieder herausdrueckt. Der Ausstoesser sitzt auf einer
exzentrischen Achse; verschiebt man ihn in der XY-Ebene, druecken seine
Platten auf der Seite nach aussen, die gerade aus dem Teig laeuft.

Der Bauraum wird dafuer **voxelisiert**, statt jedem Koerper wie frueher ein
Radiusfeld r(theta, z) zuzuordnen. Damit darf ein Ausstoesser-Bereich unter
einer Klinge hindurchlaufen -- und ein vollstaendig eingeschlossenes Segment
(Puzzleteil!) wird nicht mehr mit Stegen quer durch das Muster angebunden,
sondern in der Tiefe.

Zwei Prinzipien tragen den Aufbau:

1.  **45deg-Projektion.** Jedes Musterpixel wird radial nach innen projiziert
    und dabei um 45 Grad nach unten gekippt: ein Schritt nach innen ist ein
    Schritt nach unten. Der Zylinder wird stehend gedruckt, und so sitzt
    jedes Voxel auf dem Diagonalnachbarn darunter -- die Grenze, die FDM
    ohne Stuetzmaterial schafft.
2.  **Koexistenz durch ein Gyroid.** Tiefer im Bauteil zerlegt ein Gyroid
    (dieselbe Struktur wie in Waermetauschern) den Raum in zwei ineinander
    verschlungene, jeweils zusammenhaengende Netzwerke, die sich nirgends
    beruehren. Dort verbinden sich die eingeschlossenen Platten
    untereinander, ohne das Muster anzutasten.

Beides wird in einem einzigen Skalarfeld zusammengefuehrt, dessen Vorzeichen
den Bauraum teilt; der Bewegungsspalt entsteht anschliessend durch Erosion
und ist damit exakt statt geschaetzt. Gefordert ist dabei die HALBE
Hubstrecke: der Ausstoesser sitzt exzentrisch und wandert aus seiner
Mittellage um +-Hub/2.

**Das Muster geht vollstaendig ins Bauteil.** Jedes Klingenpixel taucht im
fertigen Koerper auf -- der Spalt kommt im Aussenband ausschliesslich vom
Ausstoesser, jede Klingenlinie bekommt eine durchgehende Saeule bis zu ihrem
eigenen Netzwerk, und die Musterzellen sind vor jeder Reparatur geschuetzt.
``pattern_completeness`` im Report misst es. Was danach noch offen
ist -- schwebendes Material, lose Fragmente, Kantenkontakte -- wird repariert
und **nachgemessen**: der Report nennt fuer jede Bedingung das Ergebnis
(Anzahl Koerper, schwebende Voxel, eingehaltener Hub, Wandstaerke, verlorene
Musterflaeche) statt sie nur zu behaupten.

**Der Ausstoesser wird zuerst versorgt, wo es eng wird.** Die Reihenfolge
ist: Muster und Saeulen der Schneide festlegen, dann die Platten des
Ausstoessers und ihre Verbindung nach innen, dann die Konflikte. Wo beide
denselben Weg brauchen, wird die Trennflaeche selbst verschoben, bevor
Klingenmaterial entsteht -- eine Verbindung an der Unterkante einer Platte
wird dringender gebraucht als ein weiteres Stueck Fuellung an einer
Schneide, die dort ohnehin schon haengt. Aus dem fertigen Koerper Korridore
herauszuschneiden waere das Gegenteil: es laesst die Saeulen als Inseln im
Loch stehen und zerlegt die Schneide.

Bleibt danach ein Plattenstueck uebrig, das im Bewegungsspalt der Schneide
weder anzubinden noch zu stuetzen ist, wird es **aufgegeben** statt als
loses Teil mitgedruckt: genau zwei Koerper ist die Bedingung, an der das
Bauteil haengt. Wie viel das kostet, nennt der Report als
``ejector_volume_dropped_mm3`` und die App als Hinweis.

Der Radius hat dabei eine **Untergrenze** (Schneidentiefe + Nabe + vier
Maschenmasse, mit den Voreinstellungen rund 27 mm): darunter bleibt zwischen
Nabe und Schneidentiefe zu wenig Platz fuer die Gyroid-Zone, und der
Ausstoesser kaeme in mehreren Teilen heraus. Die App sperrt den
Generieren-Knopf und nennt den Grund. Ebenso muss die Voxelkante deutlich
unter dem Spalt liegen -- sonst frisst die Diskretisierung ihn auf; zu grobe
Werte werden automatisch verfeinert und gemeldet.

## 💻 Technologies Used

- **Python**
- **Streamlit:** For the web application interface.
- **Trimesh:** For robust 3D mesh creation and manipulation.
- **PyVista:** For 3D visualization.
- **Pillow:** For image processing.
- **NumPy:** For numerical operations.
- **Shapely:** For geometric operations on the cap surfaces.
- **Scipy:** For scientific computing, specifically for Delaunay triangulation.
- **manifold3d:** For robust boolean operations.
- **networkx:** For graph operations on the mesh.

## ✅ Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

`test_gyroid_coexistence.py` prueft die Voxel-Logik einzeln (Metrik des
Bewegungsspalts, 45deg-Treppe, Stuetz- und Verbindungsreparatur, Mesh),
`test_app.py` die ganze Kette ueber die Streamlit-Oberflaeche -- bewusst mit
den **Voreinstellungen der App**, weil die Fehler dieses Generators an den
physischen Massen haengen und nicht am Bildinhalt.