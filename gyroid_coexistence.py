"""
gyroid_coexistence.py
=====================

Koexistenz von Schneide und Ausstoesser im selben Bauraum -- Voxel-Ansatz.

Warum ein kompletter Rewrite
-----------------------------
Die Vorgaengerlogik beschrieb beide Teile als Radiusfeld r(theta, z): pro
Winkel und Hoehe genau EIN Radius je Koerper. Damit ist die radiale Ordnung
fest verdrahtet -- aussen immer Schale, innen immer Kern -- und die einzige
Antwort auf ein zerfallendes Muster waren Stege IN DER BILDEBENE.
Eingeschlossene Segmente (Puzzleteil-Innenflaechen) mussten quer durch das
Schnittmuster angebunden werden, und jeder solche Steg zerstoert Muster.

Der Rewrite gibt die radiale Ordnung auf. Der Bauraum wird voxelisiert, und
jedes Voxel hat einen von drei Zustaenden: leer, SCHNEIDE oder AUSSTOESSER.
Damit darf ein Ausstoesser-Bereich unter einer Klinge hindurchlaufen, und ein
eingeschlossenes Segment wird nicht mehr in der Bildebene angebunden, sondern
in der Tiefe. Kein einziger Steg im Muster.

Was hier gebaut wird
---------------------
Ein Ausstech-Roller. Die dunklen Linien des Bildes sind die KLINGE; die
hellen Flaechen dazwischen sind das, was der AUSSTOESSER herausdrueckt. Der
Ausstoesser sitzt auf einer exzentrischen Achse: verschiebt man ihn in der
XY-Ebene, druecken seine Platten auf der Seite, die gerade aus dem Teig
laeuft, nach aussen. Daraus folgt die Bedingung, die den ganzen Aufbau
bestimmt:

    Eine Verschiebung des Ausstoessers um ``travel_mm`` (Default 3 mm) in
    IRGENDEINE Richtung der XY-Ebene darf nirgends zu einer Beruehrung mit
    der Schneide fuehren.

In z bewegt sich nichts, dort genuegt Druckspiel. Geprueft wird das nicht per
Formel, sondern gemessen (``check_xy_travel``): der Ausstoesser wird um
``travel_mm`` in XY dilatiert; schneidet das Ergebnis die Schneide, ist die
Bedingung verletzt.

Grundprinzip 1: 45deg-Projektion nach innen
--------------------------------------------
Jedes Musterpixel wird radial nach innen projiziert und dabei um 45deg nach
UNTEN gekippt: ein Schritt nach innen ist ein Schritt nach unten.

Der Grund ist die Druckbarkeit. Der Zylinder wird STEHEND gedruckt
(Zylinderachse = Aufbaurichtung, eine Druckschicht = eine Bildspalte). Bei
gerader radialer Projektion hinge die Unterkante jedes Musterdetails als
waagerechter Kragarm in der Luft. Durch den 45deg-Versatz sitzt jedes Voxel
auf dem Diagonalnachbarn darunter -- genau die Grenze, die FDM ohne
Stuetzmaterial kann. Diese Treppe ist so zentral, dass ein Rundungsfehler in
ihrer Definition (Verschiebung aus Millimetern statt aus Zellen gerechnet,
Auflage-Pruefung mit dr > dz) jedes Mal denselben Schaden anrichtet: das
Muster gilt als schwebend und wird weggetrimmt.

Grundprinzip 2: Koexistenz durch Gyroid-Fuellung
-------------------------------------------------
Tiefer im Bauteil koennen sich die Projektionen nicht mehr ausweichen: die
Klingen bilden ein geschlossenes Wandnetz, und jede Ausstoesserflaeche
zwischen den Klingen waere davon abgeschnitten.

Die Antwort kommt aus dem Waermetauscherbau: ein Gyroid (TPMS) zerlegt den
Raum in ZWEI ineinander verschlungene, jeweils fuer sich zusammenhaengende
Netzwerke, die sich nirgends beruehren -- genau das, was zwei Koerper
brauchen, die denselben Bauraum teilen, ohne sich zu vermischen. Verschlungen
heisst dabei nicht verklemmt: der Spalt laesst den Hub in jeder Richtung zu.

Das Zugehoerigkeitsfeld: ein Feld statt zweier Koerper
-------------------------------------------------------
Aussen gibt das Muster die Zugehoerigkeit vor, innen das Gyroid. Beides wird
in EINEM Skalarfeld phi zusammengefuehrt (``allegiance_field``), dessen
Vorzeichen den Bauraum teilt:

    phi >= 0 -> Seite der Schneide      phi < 0 -> Seite des Ausstoessers

Aussen ist phi der vorzeichenbehaftete Abstand zur Klingenkante (per
``scipy.ndimage.distance_transform_edt``), um 45deg nach innen/unten
geschert; innen das Gyroid; dazwischen wird ueberblendet. Ein
``np.minimum`` mit einem radialen Kegel macht den Ausstoesser innen massiv
(die Nabe) und haelt die Klinge von ihr fern.

Der Spalt entsteht durch EROSION, nicht durch eine Niveaumenge
---------------------------------------------------------------
Beide Koerper sind erodierte Haelften von phi. Das ist exakt und nicht nur
naeherungsweise: liegt x in der erodierten Schneide und y im erodierten
Ausstoesser, kreuzt die Verbindungsstrecke die Trennflaeche in einem Punkt p,
und es gilt

    d(x,y) = d(x,p) + d(p,y) >= e_Schneide + e_Ausstoesser.

Die Summe der beiden Erosionen IST der Spalt -- ohne Bisektion, ohne
Gradientenabschaetzung. Die frueher probierte Variante ("Spalt als
|phi| <= h") musste h so lange aufziehen, bis auch die steilste Stelle des
Feldes passte, und bezahlte 5.95 mm Spalt fuer 3 mm Hub mit der Wandstaerke
beider Koerper.

Aufgeteilt wird die Summe tiefenabhaengig (``split_bodies``): im Aussenband
ist die Klinge oft nur ein bis zwei Millimeter breit und kann nichts abgeben
-- dort traegt der Ausstoesser den ganzen Hub, wie bei jeder
Auswerfer-Geometrie. Tief innen sind beide Strukturen gleich dick und teilen
sich den Hub haelftig. Die Rampe dazwischen ist monoton und wird beim
Ausstoesser eine Hubtiefe versetzt ausgewertet; damit ist die Summe in jeder
Kombination mindestens der Hub.

Was danach noch repariert werden muss
--------------------------------------
Vier Bedingungen bleiben, die aus phi nicht von selbst folgen. Jede hat ihre
Reparatur, jede ihren Preis im Report -- das ist das Mass fuer die
"Verzerrung", die es zu minimieren gilt:

- ``repair_support``   : kein Voxel schwebt. Unter jedes Voxel ohne
                         45deg-Auflage waechst eine Stuetze aus eigenem
                         Material; was sich nicht abstuetzen laesst, faellt
                         weg.
- ``repair_connectivity``: genau EIN Koerper je Zustand. Eine Breitensuche von
                         allen Fragmenten gleichzeitig verteilt den erlaubten
                         Freiraum unter ihnen; wo zwei Reviere aneinander
                         stossen, liegt der kuerzeste Weg. Daraus waehlt ein
                         minimaler Spannbaum (scipy) genau die noetigen
                         Verbindungen aus.
- ``resolve_diagonal_contacts``: Voxel, die sich nur ueber eine KANTE
                         beruehren, sind kein Verbund, sondern ein Scharnier
                         mit Querschnitt null -- und im Mesh eine Kante mit
                         vier statt zwei Dreiecken, an der jede
                         Boolean-Operation scheitert. Erst fuellen, sonst
                         trennen.
- ``drop_specks`` / ``drop_unreachable``: was zu klein oder nicht anbindbar
                         ist, wird entfernt statt als loses Teil exportiert
                         -- und gemeldet.

Die Reparaturen machen einander Arbeit: Stuetzen loeschen, Loeschen zerreisst
den Zusammenhang, Verbinden setzt neues Material, das wieder schweben kann.
``repair_body`` laesst sie deshalb im Wechsel laufen, bis alle drei
Bedingungen zugleich erfuellt sind, und schliesst mit einer Phase ab, die nur
noch WEGNIMMT -- monoton, damit sie garantiert terminiert und das Mesh am
Ende geschlossen ist.

Alles Weitere wird nachgemessen, nicht geglaubt (``measure``): Zahl der
Koerper, schwebende Voxel, Bodenkontakt, Bewegungsspalt, Wandstaerke,
verlorene Musterflaeche. Was nicht aufgeht, steht als Warnung im Report --
mit Zahlen und mit dem Hinweis, an welcher Stellschraube es haengt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
import trimesh


# ---------------------------------------------------------------------------
# 1. Zylindrisches Voxelgitter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CylGrid:
    """Voxelgitter in Zylinderkoordinaten.

    Alle 3D-Felder haben die Form (nt, nz, nr):
        Achse 0 = theta (Umfang)   -> PERIODISCH
        Achse 1 = z (Zylinderachse = Aufbaurichtung) -> offen, z=0 ist Druckplatte
        Achse 2 = r (radial)       -> offen, Index 0 = Achse, Index nr-1 = Mantel

    Die Zellgroesse ist in z und r exakt ``voxel_mm``; in theta wird die
    Zellzahl so gewaehlt, dass die Bogenlaenge AM AUSSENRADIUS ebenfalls
    ``voxel_mm`` betraegt. Weiter innen werden die Zellen in Umfangsrichtung
    feiner -- das ist der Grund, warum jede Operation, die in Millimetern
    bemessen ist, die Ringbreite radiusabhaengig umrechnen muss (siehe
    ``dilate_xy``).
    """

    radius_mm: float
    height_mm: float
    voxel_mm: float
    nt: int
    nz: int
    nr: int

    @classmethod
    def from_dimensions(cls, radius_mm: float, height_mm: float,
                        voxel_mm: float, max_voxels: int = 40_000_000) -> "CylGrid":
        voxel_mm = float(voxel_mm)
        for _ in range(12):
            nt = max(8, int(round(2 * np.pi * radius_mm / voxel_mm)))
            nz = max(4, int(round(height_mm / voxel_mm)))
            nr = max(4, int(round(radius_mm / voxel_mm)))
            if nt * nz * nr <= max_voxels:
                break
            # Zu fein fuer den Speicher: Kantenlaenge vergroebern und neu rechnen.
            voxel_mm *= 1.25
        return cls(float(radius_mm), float(height_mm), voxel_mm, nt, nz, nr)

    # -- Metrik ------------------------------------------------------------
    @property
    def dtheta(self) -> float:
        return 2 * np.pi / self.nt

    @property
    def dz(self) -> float:
        return self.height_mm / self.nz

    @property
    def dr(self) -> float:
        return self.radius_mm / self.nr

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nt, self.nz, self.nr)

    def r_centers(self) -> np.ndarray:
        return (np.arange(self.nr) + 0.5) * self.dr

    def z_centers(self) -> np.ndarray:
        return (np.arange(self.nz) + 0.5) * self.dz

    def theta_centers(self) -> np.ndarray:
        return (np.arange(self.nt) + 0.5) * self.dtheta

    def depth_centers(self) -> np.ndarray:
        """Tiefe unter der Mantelflaeche, je Radiusindex."""
        return self.radius_mm - self.r_centers()

    def radial_index_at_depth(self, depth_mm: float) -> int:
        """Kleinster Radiusindex, dessen Zellmitte noch flacher liegt als
        ``depth_mm`` -- die Zone ``depth < depth_mm`` ist also
        ``[idx:nr]``."""
        idx = int(np.searchsorted(-self.depth_centers(), -float(depth_mm)))
        return int(np.clip(idx, 0, self.nr))

    def arc_mm(self) -> np.ndarray:
        """Bogenlaenge einer Umfangszelle je Radiusindex."""
        return self.r_centers() * self.dtheta

    def voxel_volume(self) -> np.ndarray:
        """Volumen einer Zelle je Radiusindex (fuer Volumenberichte)."""
        r0 = np.arange(self.nr) * self.dr
        r1 = r0 + self.dr
        return 0.5 * self.dtheta * (r1 ** 2 - r0 ** 2) * self.dz


# ---------------------------------------------------------------------------
# 2. Morphologie auf dem Zylindergitter
# ---------------------------------------------------------------------------
#
# Alle Radien hier sind MILLIMETER, nie Pixel: die Druckbarkeit haengt an der
# physikalischen Groesse, nicht an der gewaehlten Aufloesung.

def _wrap_max_theta(mask: np.ndarray, half_width: int) -> np.ndarray:
    """Maximumfilter entlang theta (Achse 0) mit umlaufendem Rand."""
    if half_width <= 0:
        return mask
    return ndimage.maximum_filter1d(
        mask, size=2 * half_width + 1, axis=0, mode="wrap"
    )


def dilate_xy(mask: np.ndarray, mm: float, grid: CylGrid) -> np.ndarray:
    """Dilatation um ``mm`` in der XY-EBENE (theta und r), NICHT in z.

    Das ist die Operation hinter der Bewegungsfreiheit: der Ausstoesser sitzt
    exzentrisch und wird als Ganzes in XY verschoben. In der Schale dreht sich
    die Verschiebungsrichtung ueber eine Umdrehung einmal komplett durch --
    gefordert ist also Spiel in JEDER Richtung der XY-Ebene, nicht nur radial.
    In z bewegt sich nichts; dort genuegt Druckspiel.

    Umgesetzt als exakte Kreisscheibe in der Metrik (Bogenlaenge, Radius):
    fuer jeden radialen Versatz k wird in Umfangsrichtung um
    sqrt(mm^2 - (k*dr)^2) dilatiert. Die Umfangsbreite ist dabei
    radiusabhaengig -- innen sind die Zellen schmaler -- deshalb ringweise.

    Die Naeherung liegt auf der sicheren Seite: gemessen wird entlang des
    BOGENS, die echte Sehne ist kuerzer. Der Spalt faellt also eher zu gross
    aus als zu klein.
    """
    if mm <= 0:
        return np.asarray(mask, dtype=bool).copy()
    mask = np.asarray(mask, dtype=bool)
    r = grid.r_centers()
    kmax = int(np.ceil(mm / grid.dr)) + 1
    out = np.zeros_like(mask)
    ring = np.arange(grid.nr)
    for k in range(-kmax, kmax + 1):
        shifted = _shift_r(mask, k)
        r2 = r
        r1 = r[np.clip(ring - k, 0, grid.nr - 1)]
        # Winkelbreite EXAKT ueber die Sehne, und zwar zwischen den
        # naechstliegenden KANTEN der beiden Zellen -- nicht zwischen ihren
        # Mittelpunkten und nicht ueber den Bogen.
        #
        # Zwei Gruende, beide im Test aufgefallen:
        #
        # 1. Der Bogen ist laenger als die Sehne. Zwei Punkte auf
        #    verschiedenen Radien koennen 3.0 mm Bogenabstand haben und
        #    trotzdem nur 2.9 mm Luftlinie -- die Verbindungsgerade schneidet
        #    die Kurve ab.
        # 2. Ein Voxel ist kein Punkt. Zwei Zellmittelpunkte im Abstand
        #    3.16 mm gehoeren bei 1.6 mm Kantenlaenge zu Koerpern, zwischen
        #    denen nur noch 1.56 mm Luft ist. Gemessen wurde deshalb ein
        #    "eingehaltener" Spalt, und der Ausstoesser hat sich beim
        #    tatsaechlichen Verschieben um 3 mm mit 4000 mm3 in die Schneide
        #    gefressen.
        #
        # Also: Radien um je eine halbe Zelle aufeinander zu ruecken (die
        # einander zugewandten Kanten), und die Winkelbreite um eine ganze
        # Zelle aufweiten (je eine halbe Zelle Breite auf beiden Seiten).
        lo = np.minimum(r1, r2) + grid.dr / 2.0
        hi = np.maximum(r1, r2) - grid.dr / 2.0
        lo = np.minimum(lo, hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_max = (lo ** 2 + hi ** 2 - mm ** 2) / (2 * lo * hi)
        cos_max = np.clip(cos_max, -1.0, 1.0)
        dtheta_max = np.arccos(cos_max)
        half = np.ceil(dtheta_max / grid.dtheta - 1e-9).astype(int) + 1
        half = np.where(hi - lo > mm + 1e-9, -1, half)
        half = np.minimum(half, grid.nt // 2)
        for width in np.unique(half):
            if width < 0:
                continue
            sel = np.flatnonzero(half == width)
            if width == 0:
                out[:, :, sel] |= shifted[:, :, sel]
            else:
                out[:, :, sel] |= _wrap_max_theta(shifted[:, :, sel], int(width))
    return out


def _shift_r(a: np.ndarray, k: int) -> np.ndarray:
    """Verschiebung entlang r um k Zellen.

    Ausserhalb des Gitters wird mit LEER aufgefuellt, nicht mit dem Randwert
    fortgesetzt. Eine Fortsetzung schmiert die Dilatation an der Achse und an
    der Mantelflaeche in den jeweils letzten Ring hinein und macht die
    Abstandsrelation dort unsymmetrisch -- das hat den Bewegungsspalt nach den
    Reparaturen an ein paar Dutzend Stellen scheinbar verletzt, obwohl beide
    Koerper ihn eingehalten hatten. Ausserhalb des Zylinders gibt es kein
    Material; leer ist die richtige Fortsetzung."""
    if k == 0:
        return a
    out = np.zeros_like(a)
    if k > 0:
        out[:, :, k:] = a[:, :, :-k]
    else:
        out[:, :, :k] = a[:, :, -k:]
    return out


def dilate_3d(mask: np.ndarray, mm: float, grid: CylGrid) -> np.ndarray:
    """Wie ``dilate_xy``, zusaetzlich in z -- das normale Druckspiel."""
    out = dilate_xy(mask, mm, grid)
    half_z = int(np.ceil(mm / grid.dz))
    if half_z > 0:
        out = ndimage.maximum_filter1d(out, size=2 * half_z + 1, axis=1,
                                       mode="nearest")
    return out


def _disk_offsets(radius_px: int) -> np.ndarray:
    r = int(radius_px)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2) <= r * r + 1e-9


def signed_pattern_distance(mask2d: np.ndarray, pitch_theta_mm: float,
                            pitch_z_mm: float) -> np.ndarray:
    """Vorzeichenbehafteter Abstand zur Klingenkante, in Millimetern.

    Positiv INNERHALB der Klinge, negativ ausserhalb; der Betrag ist der
    Abstand zur Kante. Gerechnet mit ``scipy.ndimage.distance_transform_edt``
    (anisotrop ueber ``sampling``, theta umlaufend gepolstert) statt mit
    selbstgebauten Struktur-Elementen -- die Abstandstransformation ist exakt,
    linear in der Pixelzahl und liefert nebenbei genau die Eigenschaft, auf
    der der ganze Aufbau steht: sie ist 1-LIPSCHITZ. Zwei Niveaumengen
    ``{d >= +h}`` und ``{d <= -h}`` eines 1-Lipschitz-Feldes haben immer
    mindestens den Abstand ``2h``. Der Bewegungsspalt folgt damit aus der
    Konstruktion und muss nicht nachtraeglich freigeschnitten werden.
    """
    mask2d = np.asarray(mask2d, dtype=bool)
    pad = mask2d.shape[0]  # halbe Umdrehung reicht immer
    pad = max(1, min(pad // 2, 256))
    padded = np.pad(mask2d, ((pad, pad), (0, 0)), mode="wrap")
    sampling = (pitch_theta_mm, pitch_z_mm)
    inside = ndimage.distance_transform_edt(padded, sampling=sampling)
    outside = ndimage.distance_transform_edt(~padded, sampling=sampling)
    sd = inside - outside
    return sd[pad:pad + mask2d.shape[0], :]


def shear_field(field2d: np.ndarray, grid: CylGrid) -> np.ndarray:
    """Die 45deg-Projektion fuer ein SKALARFELD: der Wert in der Tiefe k
    stammt aus der Bildzeile z + k.

    Damit wandert das ganze Muster beim Weg nach innen um genau eine Schicht
    nach unten je Schicht nach innen -- die Treppe, auf der die Druckbarkeit
    beruht. Ueber dem Bildrand wird die letzte Zeile fortgesetzt: eine
    senkrechte Verlaengerung ist druckbar, ein Loch im Ruecken des Musters
    waere es nicht.
    """
    nt, nz, nr = grid.shape
    out = np.empty((nt, nz, nr), dtype=np.float32)
    zz = np.arange(nz)
    for i in range(nr):
        k = (nr - 1) - i
        rows = np.minimum(zz + k, nz - 1)
        out[:, :, i] = field2d[:, rows]
    return out


def min_thickness_ok(mask: np.ndarray, mm: float, grid: CylGrid) -> float:
    """Anteil der Voxel, die nach einer Erosion um ``mm/2`` in XY uebrig
    bleiben -- ein grobes, aber ehrliches Mass fuer "wie viel von dieser
    Struktur ist wirklich dicker als ``mm``"."""
    if not mask.any():
        return 0.0
    eroded = ~dilate_xy(~mask, mm / 2.0, grid)
    return float(eroded.sum()) / float(mask.sum())


# ---------------------------------------------------------------------------
# 3. Zusammenhang (theta periodisch)
# ---------------------------------------------------------------------------
#
# Konnektivitaet ist hier immer die 6er-Nachbarschaft (Flaechenkontakt).
# Zwei Voxel, die sich nur ueber eine Kante oder Ecke beruehren, sind im
# gedruckten Teil kein Verbund, sondern eine Sollbruchstelle mit Querschnitt
# null -- wer sie als "verbunden" zaehlt, exportiert genau diese Bruchstellen.

_STRUCT6 = ndimage.generate_binary_structure(3, 1)


def label_periodic(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """``ndimage.label`` mit umlaufender theta-Achse."""
    labels, n = ndimage.label(mask, structure=_STRUCT6)
    if n <= 1:
        return labels, n
    # Naht theta = 0 / theta = nt-1 zusammenfuehren.
    a = labels[0]
    b = labels[-1]
    touching = (a > 0) & (b > 0)
    if touching.any():
        pairs = np.unique(np.stack([a[touching], b[touching]], axis=1), axis=0)
        parent = np.arange(n + 1)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in pairs:
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
        roots = np.array([find(i) for i in range(n + 1)])
        # Auf 1..m umnummerieren
        uniq = np.unique(roots[1:])
        remap = np.zeros(n + 1, dtype=np.int32)
        for new, old in enumerate(uniq, start=1):
            remap[roots == old] = new
        remap[0] = 0
        labels = remap[labels]
        n = len(uniq)
    return labels, n


def count_components(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return label_periodic(mask)[1]


# ---------------------------------------------------------------------------
# 4. Grundprinzip 1: 45deg-Projektion nach innen
# ---------------------------------------------------------------------------

def project_45(mask2d: np.ndarray, grid: CylGrid,
               from_depth_mm: float, to_depth_mm: float,
               tilt: float = 1.0) -> np.ndarray:
    """Projiziert eine 2D-Mustermaske (nt, nz) radial nach innen und dabei um
    45deg nach unten -- und gibt alle Voxel zurueck, die dabei ueberstrichen
    werden.

    Ein Schritt nach innen (dr) = ``tilt`` Schritte nach unten (dz). Bei
    ``tilt = 1`` und gleicher Kantenlaenge ist das exakt 45deg: das Voxel in
    der Tiefe d stammt aus dem Pixel bei z + d/dz, sitzt also entweder direkt
    auf seinem Vorgaenger oder auf dessen Diagonalnachbarn.

    Am oberen Bildrand wird die letzte Zeile fortgesetzt (Clamp) statt Material
    verschwinden zu lassen: eine senkrechte Verlaengerung ist druckbar, ein
    Loch im Ruecken des Musters waere es nicht.
    """
    nt, nz, nr = grid.shape
    out = np.zeros((nt, nz, nr), dtype=bool)
    depth = grid.depth_centers()
    zone = (depth >= from_depth_mm - 1e-9) & (depth < to_depth_mm - 1e-9)
    if not zone.any():
        return out
    idx = np.flatnonzero(zone)
    # Die Verschiebung wird in ZELLEN gerechnet, nicht aus der Tiefe in
    # Millimetern gerundet. Sonst springt sie zwischen benachbarten Ringen um
    # zwei Schichten (0, 2, 2, 4, ...), und genau der uebersprungene Schritt
    # ist die Auflage, von der die 45deg-Stuetzung lebt: gemessen am
    # Puzzle-Bild hingen dadurch 4796 von 9166 Voxeln der Massivzone in der
    # Luft. Nur ein LUECKENLOS um eins wachsender Versatz ergibt die
    # durchgehende Treppe.
    steps = (nr - 1) - idx                       # Zellen nach innen
    shift = np.rint(steps * (grid.dr / grid.dz) * tilt).astype(int)
    shift = np.clip(shift, 0, nz - 1)
    zz = np.arange(nz)
    for s in np.unique(shift):
        src_rows = np.minimum(zz + int(s), nz - 1)
        sheared = mask2d[:, src_rows]
        cols = idx[shift == s]
        out[:, :, cols] = sheared[:, :, None]
    return out


# ---------------------------------------------------------------------------
# 5. Grundprinzip 2: das Gyroid
# ---------------------------------------------------------------------------

def gyroid_field(grid: CylGrid, period_mm: float,
                 phase: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 z_stretch: float = 1.0) -> np.ndarray:
    """Das (unverzerrte) Gyroid, ausgewertet in den Zellmitten.

    Bewusst im KARTESISCHEN Raum ausgewertet und erst dann auf das
    Zylindergitter abgebildet: ein in Zylinderkoordinaten "gerade gezogenes"
    Gyroid waere zur Achse hin gestaucht und nach aussen gestreckt -- genau
    die Verzerrung, die es zu vermeiden gilt. So ist die Zellgroesse ueberall
    dieselbe, und Wandstaerke wie Spaltbreite sind im ganzen Bauteil gleich.
    """
    k = 2 * np.pi / float(period_mm)
    kz = k / max(float(z_stretch), 1e-6)
    th = grid.theta_centers()[:, None, None]
    z = grid.z_centers()[None, :, None]
    r = grid.r_centers()[None, None, :]
    x = k * (r * np.cos(th)) + phase[0]
    y = k * (r * np.sin(th)) + phase[1]
    zz = kz * z + phase[2]
    sx, cx = np.sin(x), np.cos(x)
    sy, cy = np.sin(y), np.cos(y)
    sz, cz = np.sin(zz), np.cos(zz)
    return sx * cy + sy * cz + sz * cx


@dataclass
class GyroidFit:
    """Die gefundenen Gyroid-Parameter -- und woran sie gemessen wurden."""
    period_mm: float
    phase: tuple[float, float, float]
    z_stretch: float
    wall_ratio_a: float = 0.0
    wall_ratio_b: float = 0.0
    fragments: int = 0
    floating: int = 0
    tried: list = field(default_factory=list)


def gyroid_halves(grid: CylGrid, zone: np.ndarray, fit: GyroidFit,
                  erosion_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Die beiden Gyroid-Haelften, so wie sie spaeter wirklich gebaut werden:
    Trennflaeche bei g = 0, Spalt durch beidseitige Erosion."""
    g = gyroid_field(grid, fit.period_mm, fit.phase, fit.z_stretch)
    side = g >= 0
    a = side & ~dilate_xy(~side, erosion_mm, grid) & zone
    b = ~side & ~dilate_xy(side, erosion_mm, grid) & zone
    return a, b


def _phase_list(n: int) -> list[tuple[float, float, float]]:
    """Phasenlagen des Gyroids. Das GANZE Gyroid zu verschieben ist die
    billigste Form der erlaubten "Verzerrung": sie kostet keine
    Regelmaessigkeit und entscheidet trotzdem darueber, ob ein Musterdetail
    sein eigenes Netzwerk direkt trifft oder erst angebunden werden muss."""
    if n <= 1:
        return [(0.0, 0.0, 0.0)]
    return [(2 * np.pi * i / n, np.pi * i / n, 0.5 * np.pi * i / n)
            for i in range(n)]


def fit_gyroid(grid: CylGrid, zone: np.ndarray, erosion_mm: float,
               min_wall_mm: float, periods_mm: list[float],
               z_stretches: list[float] | None = None, n_phases: int = 4,
               anchor_a: np.ndarray | None = None,
               anchor_b: np.ndarray | None = None) -> GyroidFit:
    """Sucht Periode, z-Streckung und Phase des Gyroids.

    Bewertet wird an der Struktur, die spaeter TATSAECHLICH gebaut wird --
    also an den um den halben Hub erodierten Haelften, nicht an einer
    Niveaumenge ``|g| > t``. Das ist keine Formalie: die erodierte Haelfte
    zerfaellt bei einer anderen Periode als die Niveaumenge, und eine
    Anpassung, die etwas anderes bewertet als sie baut, sucht das Optimum
    fuer die falsche Struktur.

    Kriterien, in dieser Reihenfolge:

    1. beide Haelften bleiben ZUSAMMEN MIT der Massivzone je ein Koerper --
       das ist der eigentliche Zweck des Gyroids,
    2. moeglichst wenig schwebendes Material,
    3. Wandstaerke.

    Die Periode waechst aufsteigend nur so weit, wie es dafuer noetig ist:
    ein feines Gyroid bietet dem Muster mehr Anbindungspunkte. Untergrenze
    ist der Hub selbst -- eine Masche, die schmaler ist als die beidseitige
    Erosion, verschwindet beim Erodieren vollstaendig.

    Warum die z-Streckung mitgesucht wird: ein isotropes Gyroid hat in jeder
    Masche ein lokales z-Minimum, und jedes davon faengt beim Drucken in der
    Luft an. Streckt man es in z, stellen sich die Waende auf -- bei Faktor 8
    sind es am Puzzle-Bild noch 4 statt 1295 schwebende Voxel, ohne dass die
    Netzwerke ihren dreidimensionalen Zusammenhang verlieren (bei reiner
    Extrusion, also Faktor unendlich, zerfallen sie wieder in je zwei Teile).
    """
    z_stretches = z_stretches or [8.0, 4.0, 2.0]
    tried: list = []
    best: GyroidFit | None = None
    best_key = None
    for period in periods_mm:
        for zs in z_stretches:
            for phase in _phase_list(n_phases):
                cand = GyroidFit(period, phase, zs)
                a, b = gyroid_halves(grid, zone, cand, erosion_mm)
                if not a.any() or not b.any():
                    continue
                wa = min_thickness_ok(a, min_wall_mm, grid)
                wb = min_thickness_ok(b, min_wall_mm, grid)
                aa = a if anchor_a is None else (a | anchor_a)
                bb = b if anchor_b is None else (b | anchor_b)
                frag = count_components(aa) + count_components(bb)
                floating = check_floating(a, grid) + check_floating(b, grid)
                cand.wall_ratio_a, cand.wall_ratio_b = wa, wb
                cand.fragments, cand.floating = frag, floating
                tried.append({"period_mm": period, "z_stretch": zs,
                              "phase": [round(float(p), 2) for p in phase],
                              "fragments": frag, "floating": floating,
                              "wall_ratio": (round(wa, 2), round(wb, 2))})
                key = (frag, floating, -min(wa, wb))
                if best_key is None or key < best_key:
                    best_key, best = key, cand
        if best_key is not None and best_key[0] <= 2 and -best_key[2] > 0.5:
            break   # zusammenhaengend und dickwandig -- groeber muss es nicht
    if best is None:
        best = GyroidFit(periods_mm[-1], (0.0, 0.0, 0.0),
                         (z_stretches or [1.0])[0])
    best.tried = tried
    return best


# ---------------------------------------------------------------------------
# 6. Reparaturen (in dieser Reihenfolge)
# ---------------------------------------------------------------------------
#
# Jede Reparatur darf ausschliesslich in den fuer ihren Koerper ERLAUBTEN
# Freiraum schreiben. Sonst repariert die eine Bedingung genau das kaputt,
# was die andere gerade sichergestellt hat -- der Bewegungsspalt zuerst.

def _dilate6(mask: np.ndarray) -> np.ndarray:
    """6er-Dilatation, theta umlaufend, z/r am Rand beschnitten."""
    out = mask.copy()
    out |= np.roll(mask, 1, axis=0)
    out |= np.roll(mask, -1, axis=0)
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def support_reach(layer: np.ndarray, grid: CylGrid,
                  max_overhang_deg: float = 45.0) -> np.ndarray:
    """Welche Zellen der NAECHSTEN Schicht kann diese Schicht tragen?

    Erlaubt ist ein seitlicher Versatz von ``tan(max_overhang) * dz``. Bei
    45deg ist das genau eine Zelle diagonal -- und diese eine Zelle wird auch
    dann zugestanden, wenn die Zelle in r oder in Umfangsrichtung ein
    Haerchen groesser ist als die Schichthoehe.

    Das klingt nach Erbsenzaehlerei, war aber ein handfester Fehler: bei
    R = 30 mm und H = 188 mm ergaben sich dr = 0.909 mm und dz = 0.899 mm,
    also dr > dz -- und damit erlaubte die Abrundung NULL radiale Zellen. Die
    45deg-Treppe, auf der die ganze Konstruktion beruht, galt damit als
    schwebend: die Stuetzreparatur hat 208391 Voxel weggetrimmt, darunter das
    halbe Muster, und uebrig blieb ein loser Mantel im Aussenband.
    """
    reach = np.tan(np.radians(max_overhang_deg)) * grid.dz
    diagonal = max_overhang_deg >= 45.0 - 1e-9
    arc = grid.arc_mm()
    half = np.floor(reach / np.maximum(arc, 1e-9) + 1e-9).astype(int)
    if diagonal:
        half = np.maximum(half, 1)
    half = np.minimum(half, grid.nt // 2)
    out = layer.copy()
    for width in np.unique(half):
        if width <= 0:
            continue
        sel = np.flatnonzero(half == width)
        out[:, sel] |= ndimage.maximum_filter1d(layer[:, sel], size=2 * int(width) + 1,
                                                axis=0, mode="wrap")
    steps = int(np.floor(reach / grid.dr + 1e-9))
    if diagonal:
        steps = max(steps, 1)
    for k in range(1, steps + 1):
        out[:, k:] |= layer[:, :-k]
        out[:, :-k] |= layer[:, k:]
    return out


def _shift_tr(mask: np.ndarray, dt: int, dr: int) -> np.ndarray:
    """2D-Schicht (theta, r) verschieben; theta umlaufend, r beschnitten."""
    out = np.roll(mask, dt, axis=0) if dt else mask
    if dr > 0:
        shifted = np.zeros_like(out)
        shifted[:, dr:] = out[:, :-dr]
        return shifted
    if dr < 0:
        shifted = np.zeros_like(out)
        shifted[:, :dr] = out[:, -dr:]
        return shifted
    return out


def support_map(occ: np.ndarray, grid: CylGrid,
                max_overhang_deg: float = 45.0) -> np.ndarray:
    """Fuer jede Zelle: haette sie eine Auflage im 45deg-Kegel darunter?
    Die unterste Schicht steht auf der Druckplatte und gilt immer als
    getragen.

    Bewusst als eine Handvoll Filter ueber das GANZE Feld statt als Schleife
    ueber die Schichten: die Reparaturen rufen diese Funktion einige hundert
    Mal auf, und schichtweise waren das im Profil 31 von 46 Sekunden
    Laufzeit -- fuer dasselbe Ergebnis.
    """
    reach = _lateral_reach(occ, grid, max_overhang_deg)
    out = np.zeros_like(occ)
    out[:, 0, :] = True
    out[:, 1:, :] = reach[:, :-1, :]
    return out


def _lateral_reach(occ: np.ndarray, grid: CylGrid,
                   max_overhang_deg: float) -> np.ndarray:
    """Seitliche Reichweite einer Schicht, auf dem ganzen Feld auf einmal --
    dieselbe Regel wie ``support_reach``, nur nicht schichtweise."""
    reach = np.tan(np.radians(max_overhang_deg)) * grid.dz
    diagonal = max_overhang_deg >= 45.0 - 1e-9
    arc = grid.arc_mm()
    half = np.floor(reach / np.maximum(arc, 1e-9) + 1e-9).astype(int)
    if diagonal:
        half = np.maximum(half, 1)
    half = np.minimum(half, grid.nt // 2)
    out = occ.copy()
    for width in np.unique(half):
        if width <= 0:
            continue
        sel = np.flatnonzero(half == width)
        out[:, :, sel] |= ndimage.maximum_filter1d(
            occ[:, :, sel], size=2 * int(width) + 1, axis=0, mode="wrap")
    steps = int(np.floor(reach / grid.dr + 1e-9))
    if diagonal:
        steps = max(steps, 1)
    for k in range(1, steps + 1):
        out[:, :, k:] |= occ[:, :, :-k]
        out[:, :, :-k] |= occ[:, :, k:]
    return out


def trim_floating(occ: np.ndarray, grid: CylGrid,
                  max_overhang_deg: float = 45.0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Alles entfernen, was keine Auflage hat -- und was dadurch seinerseits
    die Auflage verliert, gleich mit.

    Nur LOESCHEN, nie setzen: dadurch schrumpft die Belegung monoton, das
    Verfahren terminiert garantiert, und am Ende schwebt nichts mehr.
    """
    occ = occ.copy()
    removed = np.zeros(grid.shape, dtype=bool)
    for _ in range(grid.nz + 2):
        floating = occ & ~support_map(occ, grid, max_overhang_deg)
        if not floating.any():
            break
        occ &= ~floating
        removed |= floating
    return occ, removed


def repair_support(occ: np.ndarray, allowed: np.ndarray, grid: CylGrid,
                   max_overhang_deg: float = 45.0,
                   max_pillar_layers: int = 60) -> tuple[np.ndarray, dict]:
    """Kein Voxel schwebt.

    Schicht fuer Schicht von unten nach oben: hat ein Voxel keine Auflage im
    45deg-Kegel darunter, waechst eine Stuetze aus EIGENEM Material nach
    unten, bis sie auf eigenes Material oder auf die Druckplatte trifft --
    und nur durch erlaubten Freiraum. Die Stuetze darf dabei seitlich
    ausweichen (erst senkrecht, dann radial, dann in Umfangsrichtung, also
    immer so gerade wie moeglich); ohne dieses Ausweichen endet jede Stuetze,
    deren senkrechter Weg zufaellig durch den Bewegungsspalt des anderen
    Koerpers laeuft, im Loeschen -- und Loeschen ist es, was den Zusammenhang
    zerreisst.

    Findet die Stuetze keinen Weg bis nach unten, wird die GANZE bis dahin
    gebaute Kette zurueckgerollt, das schwebende Voxel eingeschlossen, und
    ihre Zellen werden fuer weitere Versuche gesperrt. Ohne dieses
    Zurueckrollen blieb das urspruengliche Voxel schwebend stehen, waehrend
    nur die halbfertige Stuetze wieder verschwand: gemessen ein stabiles
    Hin und Her von exakt 132 gesetzten und 132 geloeschten Voxeln je
    Durchlauf, das nie konvergierte.

    Nebenbei erledigt das die Forderung "beide Koerper in der untersten
    Schicht": eine ununterbrochene Stuetzkette endet zwangslaeufig auf z=0.
    """
    occ = occ.copy()
    dead = np.zeros(grid.shape, dtype=bool)
    added = removed = 0
    offsets = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1),
               (-1, 1), (1, 1)]

    def rollback(fail: np.ndarray, level: int, chain: list) -> None:
        """Eine Stuetze, die den Boden nicht erreicht, wird komplett wieder
        abgebaut -- aber nur SIE. Das schwebende Voxel, das sie tragen
        sollte, bleibt zunaechst stehen; ueber sein Schicksal entscheidet die
        Trimm-Phase, und zwar von oben nach unten und damit ohne das
        Hin und Her, das ein Loeschen mitten im Aufbau ausgeloest hat."""
        nonlocal removed
        while fail.any():
            if not chain or chain[-1][0] != level:
                dead[:, level, :] |= fail
                return
            occ[:, level, :] &= ~fail
            dead[:, level, :] |= fail
            removed += int(fail.sum())
            _, off_idx = chain.pop()
            parents = np.zeros_like(fail)
            for oi, (dt, dr) in enumerate(offsets):
                sel = fail & (off_idx == oi)
                if sel.any():
                    parents |= _shift_tr(sel, -dt, -dr)
            level += 1
            fail = parents & occ[:, level, :]

    for j in range(1, grid.nz):
        below = support_reach(occ[:, j - 1, :], grid, max_overhang_deg)
        need = occ[:, j, :] & ~below
        if not need.any():
            continue
        chain: list = []
        level = j
        for _ in range(max_pillar_layers):
            if level == 0 or not need.any():
                break
            free_layer = (allowed[:, level - 1, :] & ~occ[:, level - 1, :]
                          & ~dead[:, level - 1, :])
            new_cells = np.zeros_like(need)
            off_idx = np.full(need.shape, -1, dtype=np.int8)
            remaining = need.copy()
            for oi, (dt, dr) in enumerate(offsets):
                if not remaining.any():
                    break
                cand = _shift_tr(remaining, dt, dr) & free_layer & ~new_cells
                if not cand.any():
                    continue
                new_cells |= cand
                off_idx[cand] = oi
                remaining &= ~_shift_tr(cand, -dt, -dr)
            if new_cells.any():
                occ[:, level - 1, :] |= new_cells
                added += int(new_cells.sum())
                chain.append((level - 1, off_idx))
            if remaining.any():
                rollback(remaining, level, chain)
            if not new_cells.any() or level - 1 == 0:
                break
            below2 = support_reach(occ[:, level - 2, :], grid, max_overhang_deg)
            need = new_cells & occ[:, level - 1, :] & ~below2
            level -= 1
        else:
            # Stuetze laenger als erlaubt -> nicht endlos weiterbauen.
            if need.any():
                rollback(need & occ[:, level, :], level, chain)

    # -- Trimmen: was jetzt noch schwebt, faellt weg -----------------------
    occ, trimmed_mask = trim_floating(occ, grid, max_overhang_deg)
    trimmed = int(trimmed_mask.sum())
    removed += trimmed
    return occ, {"support_voxels_added": added,
                 "floating_voxels_removed": removed,
                 "trimmed_voxels": trimmed,
                 "trimmed_mask": trimmed_mask | dead}


def supportable_mask(occ: np.ndarray, allowed: np.ndarray, grid: CylGrid,
                     max_overhang_deg: float = 45.0) -> np.ndarray:
    """Freie Zellen, von denen aus eine Stuetze bis nach unten gebaut werden
    KANN -- also Zellen, die entweder auf der Druckplatte stehen oder im
    45deg-Kegel ueber einer bereits stuetzbaren Zelle bzw. ueber vorhandenem
    Material liegen. Ein einziger Durchlauf von unten nach oben.

    Wozu: eine Verbindung, die durch nicht stuetzbaren Raum gelegt wird, wird
    von der naechsten Stuetzreparatur sofort wieder weggetrimmt -- und die
    Verbindungssuche legt sie in der Runde darauf genau dorthin zurueck.
    Beschraenkt man die Suche von vornherein auf stuetzbaren Raum, kann
    dieses Wechselspiel gar nicht erst entstehen.
    """
    out = np.zeros_like(occ)
    out[:, 0, :] = allowed[:, 0, :]
    for j in range(1, grid.nz):
        base = out[:, j - 1, :] | occ[:, j - 1, :]
        out[:, j, :] = allowed[:, j, :] & support_reach(base, grid, max_overhang_deg)
    return out


def repair_connectivity(occ: np.ndarray, allowed: np.ndarray, grid: CylGrid,
                        max_steps: int = 120, thicken_mm: float = 0.0
                        ) -> tuple[np.ndarray, dict]:
    """Genau EIN Koerper -- verbunden durch die Tiefe, nicht durch das Muster.

    Vorgehen (dasselbe Prinzip wie die bewaehrte Steg-Strategie des
    Vorgaengermoduls, nur in 3D und durch den erlaubten Freiraum):

    1. Eine Breitensuche startet GLEICHZEITIG von allen Fragmenten und
       verteilt den Freiraum unter ihnen -- eine Voronoi-Zerlegung mit
       Wegabstand statt Luftlinie, also unter Beachtung des Bewegungsspalts.
    2. Wo zwei Reviere aneinanderstossen, liegt der kuerzeste Weg zwischen
       den beiden Fragmenten. Das ergibt einen Nachbarschaftsgraphen mit
       echten Weglaengen als Kantengewicht.
    3. ``scipy.sparse.csgraph.minimum_spanning_tree`` waehlt daraus genau die
       Verbindungen aus, die noetig sind, um alles zu einem Koerper zu machen
       -- die insgesamt kuerzesten, ohne eine einzige ueberfluessige.

    Der Unterschied zur naiven Variante (jedes Fragment einzeln zum groessten
    ziehen) ist nicht kosmetisch: Fragmente, die vom Hauptkoerper aus gar
    nicht erreichbar sind, weil ihr Weg durch das Revier eines dritten
    Fragments fuehrt, werden so trotzdem angebunden -- ueber dieses dritte.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    occ = occ.copy()
    labels, n = label_periodic(occ)
    info = {"components_before": int(n), "links_added": 0, "link_voxels": 0,
            "components_after": int(n), "unreachable_components": 0}
    if n <= 1:
        return occ, info

    field = allowed | occ
    dist = np.full(grid.shape, -1, dtype=np.int32)
    owner = np.zeros(grid.shape, dtype=np.int32)
    dist[occ] = 0
    owner[occ] = labels[occ]
    frontier = occ.copy()
    for step in range(1, max_steps + 1):
        nxt = _dilate6(frontier) & field & (dist < 0)
        if not nxt.any():
            break
        dist[nxt] = step
        # Revier vererben: die erste Richtung, die ein Voxel erreicht, gewinnt.
        todo = nxt.copy()
        for shifted in _neighbour_views(owner):
            take = todo & (shifted > 0)
            if take.any():
                owner[take] = shifted[take]
                todo &= ~take
            if not todo.any():
                break
        owner[todo] = 0
        frontier = nxt

    # Reviergrenzen -> Kandidatenkanten (vektorisiert, kein Python-Loop
    # ueber hunderttausend Grenzvoxel)
    nt, nz, nr = grid.shape
    keys_all, w_all, pa_all, pb_all = [], [], [], []
    for axis in (0, 1, 2):
        a_own, b_own = owner, _gather(owner, axis)
        a_d, b_d = dist, _gather(dist, axis)
        border = (a_own > 0) & (b_own > 0) & (a_own != b_own) & (a_d >= 0) & (b_d >= 0)
        if not border.any():
            continue
        ti, zi, ri = np.nonzero(border)
        u = a_own[ti, zi, ri].astype(np.int64)
        v = b_own[ti, zi, ri].astype(np.int64)
        lo_, hi_ = np.minimum(u, v), np.maximum(u, v)
        keys_all.append(lo_ * (n + 1) + hi_)
        w_all.append((a_d[ti, zi, ri] + b_d[ti, zi, ri] + 1).astype(np.int64))
        pa_all.append(np.stack([ti, zi, ri], axis=1))
        nb = np.stack([ti, zi, ri], axis=1)
        nb[:, axis] = (nb[:, axis] + 1) % nt if axis == 0 else nb[:, axis] + 1
        pb_all.append(nb)
    edges: dict[tuple[int, int], tuple[int, tuple, tuple]] = {}
    if keys_all:
        keys = np.concatenate(keys_all)
        weights = np.concatenate(w_all)
        pa = np.concatenate(pa_all)
        pb = np.concatenate(pb_all)
        order = np.lexsort((weights, keys))
        keys, weights, pa, pb = keys[order], weights[order], pa[order], pb[order]
        first = np.concatenate([[True], keys[1:] != keys[:-1]])
        for k, w, a_pt, b_pt in zip(keys[first], weights[first], pa[first], pb[first]):
            edges[(int(k) // (n + 1), int(k) % (n + 1))] = (
                int(w), tuple(int(x) for x in a_pt), tuple(int(x) for x in b_pt))

    if not edges:
        info["unreachable_components"] = int(n) - 1
        return occ, info

    edge_keys = list(edges)
    rows = np.array([k[0] - 1 for k in edge_keys])
    cols = np.array([k[1] - 1 for k in edge_keys])
    data = np.array([edges[k][0] for k in edge_keys], dtype=float)
    graph = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    mst = minimum_spanning_tree(graph).tocoo()

    for u, v in zip(mst.row, mst.col):
        key = (min(u, v) + 1, max(u, v) + 1)
        if key not in edges:
            continue
        _, pa, pb = edges[key]
        path = _backtrack(dist, pa, grid) + _backtrack(dist, pb, grid) + [pa, pb]
        added = 0
        for (t, z, rr) in path:
            if not occ[t, z, rr]:
                occ[t, z, rr] = True
                added += 1
        info["links_added"] += 1
        info["link_voxels"] += added

    info["components_after"] = count_components(occ)
    info["unreachable_components"] = max(info["components_after"] - 1, 0)
    if thicken_mm > 0:
        occ |= dilate_3d(occ, thicken_mm, grid) & allowed & _dilate6(occ)
    return occ, info


def _neighbour_views(a: np.ndarray) -> list[np.ndarray]:
    """Die sechs Nachbarschichten eines Feldes (theta umlaufend)."""
    views = [np.roll(a, 1, axis=0), np.roll(a, -1, axis=0)]
    for axis in (1, 2):
        for sign in (1, -1):
            out = np.zeros_like(a)
            if sign > 0:
                sl_dst = [slice(None)] * 3; sl_src = [slice(None)] * 3
                sl_dst[axis] = slice(1, a.shape[axis]); sl_src[axis] = slice(0, -1)
            else:
                sl_dst = [slice(None)] * 3; sl_src = [slice(None)] * 3
                sl_dst[axis] = slice(0, -1); sl_src[axis] = slice(1, a.shape[axis])
            out[tuple(sl_dst)] = a[tuple(sl_src)]
            views.append(out)
    return views


def _backtrack(dist: np.ndarray, start: tuple[int, int, int],
               grid: CylGrid) -> list[tuple[int, int, int]]:
    """Vom Startvoxel dem Gefaelle des BFS-Abstandsfeldes folgen, bis der
    Hauptkoerper (Abstand 0) erreicht ist."""
    nt, nz, nr = grid.shape
    t, z, r = start
    d = int(dist[t, z, r])
    if d < 0:
        return []
    path: list[tuple[int, int, int]] = []
    while d > 0:
        best = None
        for dt, dz_, dr_ in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                             (0, 0, 1), (0, 0, -1)):
            tt = (t + dt) % nt
            zz = z + dz_
            rr = r + dr_
            if not (0 <= zz < nz and 0 <= rr < nr):
                continue
            dd = int(dist[tt, zz, rr])
            if dd >= 0 and dd < d:
                best = (tt, zz, rr, dd)
                break
        if best is None:
            return path
        t, z, r, d = best
        if d > 0:
            path.append((t, z, r))
    return path


def resolve_diagonal_contacts(occ: np.ndarray, allowed: np.ndarray,
                              protect: np.ndarray, grid: CylGrid | None = None,
                              max_overhang_deg: float = 45.0,
                              max_passes: int = 4,
                              require_support: bool = True,
                              separate_only: bool = False
                              ) -> tuple[np.ndarray, dict]:
    """Kantenkontakte aufloesen -- durch Fuellen, sonst durch Trennen.

    Betrachtet werden alle 2x2-Fenster in den drei Achsenebenen:

        A B     Ist eine Diagonale besetzt und die andere leer, beruehren
        C D     sich zwei Voxel nur ueber eine KANTE.

    Das ist zweierlei Aerger auf einmal: mechanisch eine Sollbruchstelle mit
    Querschnitt null (im gedruckten Teil ein Scharnier, kein Verbund), und im
    Mesh eine Kante mit vier statt zwei Dreiecken -- schon dreissig davon
    machen das Ergebnis nicht mehr geschlossen, und jede Boolean-Operation
    steigt dann mit "Not all meshes are volumes" aus.

    Erste Wahl ist FUELLEN: eine der beiden leeren Zellen bekommt Material.
    Gefuellt wird nur, wo der Bewegungsspalt des anderen Koerpers es zulaesst
    UND die neue Zelle selbst eine Auflage hat -- sonst schaukeln sich
    Kantenschluss und Stuetzreparatur gegenseitig auf: die eine setzt, die
    andere loescht, endlos.

    Geht das nicht, wird GETRENNT: einer der beiden Partner faellt weg,
    bevorzugt der, der kein Muster traegt. Ein verlorenes Voxel ist
    verschmerzbar, eine Sollbruchstelle nicht.
    """
    occ = occ.copy()
    filled = removed = 0
    for _ in range(max_passes):
        support = (support_map(occ, grid, max_overhang_deg)
                   if (grid is not None and require_support)
                   else np.ones_like(occ))
        free = (np.zeros_like(occ) if separate_only
                else (allowed & ~occ & support))
        add = np.zeros_like(occ)
        drop = np.zeros_like(occ)
        for axu, axv in ((0, 1), (0, 2), (1, 2)):
            a = occ
            b = _gather(occ, axu)
            c = _gather(occ, axv)
            d = _gather(b, axv)
            prot_b = _gather(protect, axu)
            prot_c = _gather(protect, axv)
            prot_d = _gather(prot_b, axv)
            free_b = _gather(free, axu)
            free_c = _gather(free, axv)
            cases = (
                # Diagonale A--D besetzt -> fuellen bei B oder C,
                # sonst A oder D wegnehmen.
                (a & d & ~b & ~c,
                 (free_b, (axu,)), (free_c, (axv,)),
                 (protect, ()), (prot_d, (axu, axv))),
                # Diagonale B--C besetzt -> fuellen bei A oder D,
                # sonst B oder C wegnehmen.
                (b & c & ~a & ~d,
                 (free, ()), (_gather(free_b, axv), (axu, axv)),
                 (prot_b, (axu,)), (prot_c, (axv,))),
            )
            for diag, (f1, fax1), (f2, fax2), (p1, ax1), (p2, ax2) in cases:
                if not diag.any():
                    continue
                add |= _scatter_axes(diag & f1, fax1)
                add |= _scatter_axes(diag & ~f1 & f2, fax2)
                rest = diag & ~f1 & ~f2
                if not rest.any():
                    continue
                drop |= _scatter_axes(rest & ~p2, ax2)
                drop |= _scatter_axes(rest & p2 & ~p1, ax1)
                drop |= _scatter_axes(rest & p2 & p1, ax2)   # beide Muster
        add &= allowed & ~occ
        drop &= occ & ~add
        if not add.any() and not drop.any():
            break
        occ |= add
        occ &= ~drop
        filled += int(add.sum())
        removed += int(drop.sum())
    return occ, {"diagonal_contacts_closed": filled,
                 "diagonal_contacts_separated": removed}


def _scatter_axes(mask: np.ndarray, axes) -> np.ndarray:
    out = mask
    for axis in axes:
        out = _scatter(out, axis)
    return out


def _gather(a: np.ndarray, axis: int) -> np.ndarray:
    """out[i] = a[i + 1] entlang ``axis`` (theta umlaufend)."""
    if axis == 0:
        return np.roll(a, -1, axis=0)
    return _shift_neg(a, axis)


def _scatter(a: np.ndarray, axis: int) -> np.ndarray:
    """out[i + 1] = a[i] entlang ``axis`` (theta umlaufend)."""
    if axis == 0:
        return np.roll(a, 1, axis=0)
    return _shift_pos(a, axis)


def _shift_neg(a: np.ndarray, axis: int) -> np.ndarray:
    """out[i] = a[i + 1] entlang ``axis``; ausserhalb leer."""
    out = np.zeros_like(a)
    sl_dst = [slice(None)] * 3
    sl_src = [slice(None)] * 3
    sl_dst[axis] = slice(0, a.shape[axis] - 1)
    sl_src[axis] = slice(1, a.shape[axis])
    out[tuple(sl_dst)] = a[tuple(sl_src)]
    return out


def _shift_pos(a: np.ndarray, axis: int) -> np.ndarray:
    out = np.zeros_like(a)
    sl_dst = [slice(None)] * 3
    sl_src = [slice(None)] * 3
    sl_dst[axis] = slice(1, a.shape[axis])
    sl_src[axis] = slice(0, a.shape[axis] - 1)
    out[tuple(sl_dst)] = a[tuple(sl_src)]
    return out


def drop_specks(occ: np.ndarray, grid: CylGrid, min_fragment_mm: float
                ) -> tuple[np.ndarray, dict]:
    """Fragmente entfernen, die kleiner sind als das kleinste sinnvolle
    Detail.

    Nach dem Trimmen bleiben regelmaessig Splitter von ein bis zwei Voxeln
    uebrig. Sie als eigene Koerper zu exportieren waere falsch (sie klappern
    im fertigen Teil herum), und Stege dorthin zu ziehen waere es auch: der
    Steg waere laenger als der Splitter. Also weg damit -- unabhaengig davon,
    ob sie zufaellig im Musterband liegen; ein einzelnes Voxel ist kein
    Musterdetail, sondern Diskretisierungsstaub.

    Bemessen wird in Millimetern (Wuerfel mit Kante ``min_fragment_mm``),
    damit die Schwelle nicht an der gewaehlten Voxelgroesse haengt. Ein
    Erosionstest waere hier die falsche Wahl: die Klingenwaende sind
    absichtlich nur eine Mindestwandstaerke dick und wuerden von ihm
    reihenweise als "unbaubar" aussortiert.
    """
    labels, n = label_periodic(occ)
    if n <= 1:
        return occ, {"specks_removed": 0, "speck_voxels": 0}
    min_voxels = max(2, int(round((min_fragment_mm / grid.voxel_mm) ** 3)))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes[0] = 0
    keep = sizes >= min_voxels
    keep[int(np.argmax(sizes))] = True
    keep[0] = False
    drop = ~keep[labels] & occ
    removed = int(drop.sum())
    if removed:
        occ = occ & ~drop
    return occ, {"specks_removed": int(n - int(keep.sum())),
                 "speck_voxels": removed}


def drop_unreachable(occ: np.ndarray, protect: np.ndarray, grid: CylGrid,
                     droppable_mm3: float) -> tuple[np.ndarray, dict]:
    """Was nach der Verbindungssuche immer noch lose ist, darf nicht
    stillschweigend mitexportiert werden -- es wuerde im fertigen Teil lose
    herumklappern.

    Entfernt wird JEDES Fragment ausser dem groessten -- auch eines, das
    Muster traegt. Ein nicht angebundenes Musterdetail ist naemlich kein
    halbes Produkt, sondern ein loses Teil im Bauraum: eine Ausstoesserplatte
    ohne Verbindung zum Ausstoesser drueckt nichts heraus, und ein
    abgetrenntes Stueck Klinge faellt beim ersten Gebrauch heraus. Beides
    exportiert man nicht.

    Gezaehlt und gemeldet wird es aber, getrennt nach "trug Muster" und "trug
    keines", mit Volumen. Ob der Verlust hinnehmbar ist, entscheidet der
    Report weiter oben anhand von ``droppable_fragment_mm3``: kleine Reste
    sind eine Randnotiz, ein grosser ist ein Grund, die Parameter zu aendern
    (kleinerer Hub, groesserer Radius, feineres Gitter).
    """
    labels, n = label_periodic(occ)
    info = {"loose_removed": 0, "loose_with_pattern": 0,
            "loose_volume_dropped_mm3": 0.0,
            "pattern_fragments_dropped": 0, "pattern_volume_dropped_mm3": 0.0}
    if n <= 1:
        return occ, info
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes[0] = 0
    main = int(np.argmax(sizes))
    vol_per_ring = grid.voxel_volume()
    volumes = np.zeros(n + 1)
    for i in range(grid.nr):
        volumes += np.bincount(labels[:, :, i].ravel(), minlength=n + 1) * vol_per_ring[i]
    protected = set(int(x) for x in np.unique(labels[protect & occ]) if x > 0)

    keep = np.zeros(n + 1, dtype=bool)
    keep[main] = True
    for i in range(1, n + 1):
        if i == main or sizes[i] == 0:
            continue
        if i in protected:
            info["pattern_fragments_dropped"] += 1
            info["pattern_volume_dropped_mm3"] += float(volumes[i])
        else:
            info["loose_removed"] += 1
            info["loose_volume_dropped_mm3"] = (
                info.get("loose_volume_dropped_mm3", 0.0) + float(volumes[i]))
    return occ & keep[labels], info


def count_edge_contacts(occ: np.ndarray) -> int:
    """Wie viele Voxelpaare beruehren sich nur ueber eine Kante?"""
    total = 0
    for axu, axv in ((0, 1), (0, 2), (1, 2)):
        a = occ
        b = _gather(occ, axu)
        c = _gather(occ, axv)
        d = _gather(b, axv)
        total += int((a & d & ~b & ~c).sum()) + int((b & c & ~a & ~d).sum())
    return total


def finalize_body(occ: np.ndarray, allowed: np.ndarray, protect: np.ndarray,
                  grid: CylGrid, cfg: "CoexistenceConfig"
                  ) -> tuple[np.ndarray, dict]:
    """Aufraeumen bis zum Festpunkt: Kantenkontakte aufloesen, Schwebendes
    trimmen, Splitter und lose Teile entsorgen.

    Die drei Schritte erzeugen einander gegenseitig neue Arbeit -- ein
    Trimmen hinterlaesst frische Kantenkontakte, ein Trennen hinterlaesst
    frisch schwebendes Material -- deshalb laufen sie im Wechsel, bis sich
    nichts mehr aendert. Ohne diesen Festpunkt haengt es an der Reihenfolge,
    ob ein geschlossenes Mesh herauskommt oder eines mit drei Dutzend
    vierfach benutzten Kanten.
    """
    info = {"diagonal_contacts_closed": 0, "diagonal_contacts_separated": 0,
            "floating_voxels_removed": 0, "specks_removed": 0,
            "loose_removed": 0, "loose_with_pattern": 0,
            "pattern_fragments_dropped": 0, "pattern_volume_dropped_mm3": 0.0}
    for attempt in range(6):
        before = occ.copy()
        # Ab dem dritten Anlauf wird auch ohne Auflage gefuellt: die letzten
        # ein, zwei Kontakte sind sonst nicht aufzuloesen (fuellen verboten,
        # weil unstuetzbar -- trennen erzeugt den naechsten Kontakt), und
        # genau sie machen das Mesh nicht mehr geschlossen. Die anschliessende
        # Stuetzreparatur zieht der neuen Zelle eine Saeule nach unten.
        occ, diag = resolve_diagonal_contacts(occ, allowed, protect, grid,
                                              cfg.max_overhang_deg,
                                              require_support=attempt < 2)
        occ, trimmed = trim_floating(occ, grid, cfg.max_overhang_deg)
        occ, spk = drop_specks(occ, grid, cfg.min_fragment_mm())
        occ, loose = drop_unreachable(occ, protect, grid,
                                      cfg.droppable_fragment_mm3)
        info["diagonal_contacts_closed"] += diag["diagonal_contacts_closed"]
        info["diagonal_contacts_separated"] += diag["diagonal_contacts_separated"]
        info["floating_voxels_removed"] += int(trimmed.sum())
        info["specks_removed"] += spk["specks_removed"]
        info["loose_removed"] += loose["loose_removed"]
        info["loose_with_pattern"] = loose["loose_with_pattern"]
        info["pattern_fragments_dropped"] += loose["pattern_fragments_dropped"]
        info["pattern_volume_dropped_mm3"] += loose["pattern_volume_dropped_mm3"]
        if np.array_equal(occ, before):
            break
    return occ, info


def repair_body(occ: np.ndarray, allowed: np.ndarray, protect: np.ndarray,
                grid: CylGrid, cfg: "CoexistenceConfig") -> tuple[np.ndarray, dict]:
    """Alle Reparaturen eines Koerpers, bis sie sich nicht mehr gegenseitig
    aufheben.

    Die Reihenfolge innerhalb einer Runde ist nicht beliebig. Stuetzen,
    Trimmen und Trennen LOESCHEN im Zweifel Material, und Loeschen zerreisst
    den Zusammenhang -- also kommt das Verbinden danach. Die Stege brauchen
    ihrerseits eine Auflage, also folgt darauf noch einmal die Stuetzung. Und
    weil auch das wieder neue Kantenkontakte hinterlassen kann, laeuft das
    Ganze bis zur Konvergenz und nicht in einem festen Zweischritt.

    Abgebrochen wird, sobald alle drei Bedingungen zugleich erfuellt sind:
    ein Koerper, nichts schwebt, kein Kantenkontakt. Wird das in
    ``max_repair_rounds`` Runden nicht erreicht, steht der erreichte Zustand
    im Report -- mit Zahlen, nicht mit einem Achselzucken.
    """
    occ, info = drop_specks(occ, grid, cfg.min_fragment_mm())
    # Zellen, an denen eine Stuetze schon einmal gescheitert ist, bleiben
    # gesperrt. Sonst waehlt die Verbindungssuche in der naechsten Runde
    # denselben kuerzesten Weg, die Stuetzreparatur schneidet ihn genauso
    # wieder weg, und beide Schritte wechseln sich endlos ab.
    blocked = np.zeros(grid.shape, dtype=bool)
    totals: dict = {"support_voxels_added": 0, "floating_voxels_removed": 0,
                    "diagonal_contacts_closed": 0,
                    "diagonal_contacts_separated": 0, "links_added": 0,
                    "link_voxels": 0, "loose_removed": 0,
                    "loose_with_pattern": 0, "specks_removed": 0,
                    "pattern_fragments_dropped": 0,
                    "pattern_volume_dropped_mm3": 0.0}
    for round_no in range(1, cfg.max_repair_rounds + 1):
        occ, sup = repair_support(occ, allowed & ~blocked, grid,
                                  cfg.max_overhang_deg)
        blocked |= sup.pop("trimmed_mask")
        occ, fin = finalize_body(occ, allowed & ~blocked, protect, grid, cfg)
        link_space = (allowed & ~blocked
                      & supportable_mask(occ, allowed & ~blocked, grid,
                                         cfg.max_overhang_deg))
        occ, con = repair_connectivity(occ, link_space, grid, cfg.max_link_steps)
        occ, sup2 = repair_support(occ, allowed & ~blocked, grid,
                                   cfg.max_overhang_deg)
        blocked |= sup2.pop("trimmed_mask")
        occ, fin2 = finalize_body(occ, allowed & ~blocked, protect, grid, cfg)

        totals["support_voxels_added"] += (sup["support_voxels_added"]
                                           + sup2["support_voxels_added"])
        totals["floating_voxels_removed"] += (sup["floating_voxels_removed"]
                                               + sup2["floating_voxels_removed"])
        totals["links_added"] += con["links_added"]
        totals["link_voxels"] += con["link_voxels"]
        for part in (fin, fin2):
            for key, value in part.items():
                if key == "loose_with_pattern":
                    totals[key] = value
                else:
                    totals[key] = totals.get(key, 0) + value
        info = {**info, **totals, "rounds": round_no,
                "components": count_components(occ),
                "floating": check_floating(occ, grid, cfg.max_overhang_deg),
                "edge_contacts": count_edge_contacts(occ)}
        if (info["components"] == 1 and info["floating"] == 0
                and info["edge_contacts"] == 0):
            break

    # -- Letzte Instanz: nur noch wegnehmen -------------------------------
    # Fuellen und Trennen koennen einander endlos neue Arbeit machen (ein
    # Trimmen hinterlaesst einen frischen Kantenkontakt, ein Trennen ein
    # frisch schwebendes Voxel), und ein einziger uebriggebliebener Kontakt
    # genuegt, damit das Mesh nicht mehr geschlossen ist. Dieser Abschluss
    # nimmt deshalb nur noch weg: die Belegung schrumpft streng monoton, das
    # Verfahren terminiert garantiert, und es endet ohne schwebendes Material
    # und ohne Kantenkontakt -- die beiden Bedingungen, an denen das
    # exportierte Mesh haengt.
    def settle(state: np.ndarray) -> np.ndarray:
        for _ in range(30):
            before = state.copy()
            state, trimmed = trim_floating(state, grid, cfg.max_overhang_deg)
            state, diag = resolve_diagonal_contacts(
                state, allowed, protect, grid, cfg.max_overhang_deg,
                separate_only=True)
            totals["floating_voxels_removed"] += int(trimmed.sum())
            totals["diagonal_contacts_separated"] += diag["diagonal_contacts_separated"]
            if np.array_equal(state, before):
                break
        state, spk = drop_specks(state, grid, cfg.min_fragment_mm())
        state, loose = drop_unreachable(state, protect, grid,
                                        cfg.droppable_fragment_mm3)
        totals["specks_removed"] += spk["specks_removed"]
        totals["loose_removed"] += loose["loose_removed"]
        totals["pattern_fragments_dropped"] += loose["pattern_fragments_dropped"]
        totals["pattern_volume_dropped_mm3"] += loose["pattern_volume_dropped_mm3"]
        return state

    occ = settle(occ)
    # Das Wegnehmen kann eine Verbindung gekappt haben. Also noch einmal
    # verbinden und noch einmal abraeumen -- und am Ende den Zustand
    # behalten, der am wenigsten Teile hat. Besser ein paar Voxel weniger als
    # ein Bauteil, das in zwei Stuecken aus dem Drucker kommt.
    best = occ
    best_components = count_components(occ)
    for _ in range(2):
        if best_components <= 1:
            break
        link_space = (allowed & ~blocked
                      & supportable_mask(best, allowed & ~blocked, grid,
                                         cfg.max_overhang_deg))
        candidate, con = repair_connectivity(best, link_space, grid,
                                             cfg.max_link_steps)
        candidate, sup = repair_support(candidate, allowed & ~blocked, grid,
                                        cfg.max_overhang_deg)
        sup.pop("trimmed_mask", None)
        candidate = settle(candidate)
        components = count_components(candidate)
        totals["links_added"] += con["links_added"]
        totals["link_voxels"] += con["link_voxels"]
        totals["support_voxels_added"] += sup["support_voxels_added"]
        if components < best_components:
            best, best_components = candidate, components
        else:
            break
    occ = best

    info = {**info, **totals,
            "components": count_components(occ),
            "floating": check_floating(occ, grid, cfg.max_overhang_deg),
            "edge_contacts": count_edge_contacts(occ)}
    return occ, info


# ---------------------------------------------------------------------------
# 7. Konfiguration
# ---------------------------------------------------------------------------

@dataclass
class CoexistenceConfig:
    """Alle Masse in Millimetern; nichts haengt an der Bildaufloesung."""

    voxel_mm: float = 0.8
    #: Schneidentiefe: so tief steht die Klinge ueber dem Ausstoesser.
    cut_depth_mm: float = 4.0
    #: Auswerferhub = geforderter Freiraum zwischen den Koerpern in X&Y.
    travel_mm: float = 3.0
    #: Druckspiel; wirkt zusaetzlich in ALLE Richtungen (auch z).
    print_clearance_mm: float = 0.4
    #: Ueberblendstrecke: darin geht das Muster in das Gyroid ueber.
    blend_mm: float = 6.0
    #: Duennste Wand, die noch als druckbar gilt.
    min_wall_mm: float = 1.2
    #: Achsbohrung im Ausstoesser (None = keine).
    axis_diameter_mm: float | None = 6.0
    #: Nabenradius des Ausstoessers; None = Achse + 2 * min_wall.
    hub_radius_mm: float | None = None
    #: Gyroid-Periode; None = automatisch aus dem Hub gesucht.
    gyroid_period_mm: float | None = None
    #: Streckung des Gyroids in z (>1 = aufgestellte Waende, besser
    #: druckbar). None = automatisch suchen.
    gyroid_z_stretch: float | None = None
    #: Zahl der getesteten Phasenlagen (Verschiebung des Gyroids als Ganzes).
    phase_candidates: int = 4
    max_overhang_deg: float = 45.0
    #: Obergrenze fuer die Gittergroesse -- schuetzt vor DPI-Ausrutschern.
    max_voxels: int = 40_000_000
    #: Reichweite der Verbindungssuche in Voxeln.
    max_link_steps: int = 120
    #: Wie oft Stuetzen/Verbinden/Kantenschluss einander abwechseln duerfen.
    max_repair_rounds: int = 6

    #: Bis zu dieser Groesse wird ein nicht anbindbares MUSTER-Fragment
    #: geloescht (und gemeldet) statt als loses Teil exportiert.
    droppable_fragment_mm3: float = 250.0
    #: Kleinstes Fragment, das noch als Detail zaehlt (Wuerfelkante).
    #: None = doppelte Mindestwandstaerke.
    min_fragment_mm_value: float | None = None

    def min_radius_mm(self) -> float:
        """Kleinster Radius, bei dem der Aufbau ueberhaupt Platz hat.

        Von aussen nach innen aufaddiert, jeder Posten unverzichtbar:

            Schneidentiefe + Ueberblendstrecke   das Muster und sein Uebergang
          + 2 * Mindestwandstaerke + Hub         eine Gyroid-Masche: zwei
                                                 Waende und der Spalt dazwischen
          + Hub + Druckspiel                     Abstand der Klinge zur Nabe
          + Nabenradius                          Nabe samt Achsbohrung

        Darunter bleibt fuer die Gyroid-Zone weniger als eine Masche uebrig.
        Die Koerper kommen dann zwar immer noch heraus, aber die
        eingeschlossenen Musterflaechen finden in der Tiefe keinen Weg mehr
        zueinander: gemessen an einem Puzzlemuster mit Radius 20 mm mussten
        633 mm3 Musterflaeche entfallen, weil sie ringsum vom Bewegungsspalt
        eingeschlossen waren. Das ist keine Frage der Aufloesung, sondern des
        Platzes -- deshalb wird es gesperrt statt hinterher gemeldet.
        """
        gyroid_zone = 2.0 * self.min_wall_mm + self.travel_mm
        return (self.cut_depth_mm + self.blend_mm
                + gyroid_zone
                + self.travel_mm + self.print_clearance_mm
                + self.hub_radius())

    def min_fragment_mm(self) -> float:
        if self.min_fragment_mm_value is not None:
            return float(self.min_fragment_mm_value)
        return 2.0 * self.min_wall_mm

    def hub_radius(self) -> float:
        if self.hub_radius_mm is not None:
            return float(self.hub_radius_mm)
        axis_r = (self.axis_diameter_mm or 0.0) / 2.0
        return axis_r + 2.0 * self.min_wall_mm


# ---------------------------------------------------------------------------
# 8. Der Aufbau: aus der Maske werden drei Zustaende
# ---------------------------------------------------------------------------

def resample_mask(mask2d: np.ndarray, nt: int, nz: int) -> np.ndarray:
    """Mustermaske auf das Voxelgitter bringen -- per MAXIMUM, nicht per
    naechstem Nachbarn.

    Das ist kein Detail: die Schnittlinien eines Musters sind oft nur ein bis
    zwei Bildpixel breit. Naechster-Nachbar-Abtastung zerhackt so eine Linie
    in eine gepunktete Spur -- aus einem zusammenhaengenden Klingennetz werden
    (gemessen am Puzzle-Bild) 165 lose Fragmente, und die Verbindungssuche
    darf hinterher aufraeumen, was die Abtastung kaputtgemacht hat. Mit dem
    Maximum ueberlebt jede Linie, die im Bild vorhanden ist; sie wird
    hoechstens ein Voxel breiter.

    Eingabe (theta, z) wie im Vorgaengermodul: Achse 0 = Umfang.
    """
    mask2d = np.asarray(mask2d, dtype=bool)
    src_t, src_z = mask2d.shape
    out = np.zeros((nt, nz), dtype=bool)
    ti = np.minimum((np.arange(src_t) * nt) // src_t, nt - 1)
    zi = np.minimum((np.arange(src_z) * nz) // src_z, nz - 1)
    np.logical_or.at(out, (ti[:, None], zi[None, :]), mask2d)
    if src_t < nt or src_z < nz:
        # Hochskalieren: das Maximum trifft nicht jede Zielzelle -> Luecken
        # per naechstem Nachbarn auffuellen.
        bi = np.minimum((np.arange(nt) * src_t) // nt, src_t - 1)
        bj = np.minimum((np.arange(nz) * src_z) // nz, src_z - 1)
        out |= mask2d[np.ix_(bi, bj)]
    return out


def coarsen(mask: np.ndarray, src: CylGrid, dst: CylGrid) -> np.ndarray:
    """Ein 3D-Feld auf ein groeberes Gitter bringen (Maximum je Zielzelle)."""
    out = np.zeros(dst.shape, dtype=bool)
    ti = np.minimum((np.arange(src.nt) * dst.nt) // src.nt, dst.nt - 1)
    zi = np.minimum((np.arange(src.nz) * dst.nz) // src.nz, dst.nz - 1)
    ri = np.minimum((np.arange(src.nr) * dst.nr) // src.nr, dst.nr - 1)
    np.logical_or.at(out, (ti[:, None, None], zi[None, :, None], ri[None, None, :]), mask)
    return out


def physical_gradient(field: np.ndarray, grid: CylGrid) -> np.ndarray:
    """Betrag des Gradienten in MILLIMETERN -- also mit der echten Metrik des
    Zylindergitters (die Umfangszelle ist innen schmaler als aussen)."""
    d_theta = np.diff(field, axis=0, append=field[:1]) / np.maximum(
        grid.arc_mm()[None, None, :], 1e-9)
    d_z = np.diff(field, axis=1, append=field[:, -1:]) / grid.dz
    d_r = np.diff(field, axis=2, append=field[:, :, -1:]) / grid.dr
    return np.sqrt(d_theta ** 2 + d_z ** 2 + d_r ** 2)


def allegiance_field(blade2d: np.ndarray, grid: CylGrid, fit: GyroidFit,
                     cfg: CoexistenceConfig) -> np.ndarray:
    """Das Zugehoerigkeitsfeld phi. Sein VORZEICHEN teilt den Bauraum in zwei
    Haelften:

        phi >= 0  ->  Seite der Schneide
        phi <  0  ->  Seite des Ausstoessers

    Mehr muss phi nicht leisten. Es legt nur die Trennflaeche fest; der
    Bewegungsspalt entsteht danach durch Erosion (siehe ``split_bodies``) und
    haengt deshalb nicht davon ab, wie steil phi hier oder dort verlaeuft.
    Genau daran hing die Vorgaengerfassung: sie hat den Spalt als
    Niveaumengen-Abstand ``|phi| <= h`` erzeugt und musste h so lange
    aufziehen, bis auch die steilste Stelle passte -- am Ende 5.95 mm Spalt
    fuer 3 mm Hub, bezahlt mit der Wandstaerke beider Koerper.

    Zusammengesetzt aus zwei Anteilen:

    1. MUSTER (aussen): der vorzeichenbehaftete Abstand zur Klingenkante, um
       45deg nach innen/unten geschert. An der Mantelflaeche ist ``phi >= 0``
       damit exakt die Klingenlinie.
       Der Faktor r/R traegt der Bogenlaenge Rechnung -- derselbe
       Winkelabstand ist weiter innen weniger Millimeter wert.
    2. GYROID (innen): dasselbe in Millimeter-Steigung umgerechnet, damit die
       Ueberblendung zwei vergleichbare Groessen mischt und nicht Aepfel mit
       Birnen.

    Beide Anteile werden auf ``+-clip_mm`` begrenzt. Ohne das dominiert tief
    in einer grossen Musterflaeche der Abstand zur weit entfernten Kante
    (leicht 40 mm) die Ueberblendung vollstaendig, und die Trennflaeche
    springt an der Zonengrenze statt weich zu wandern.
    """
    r = grid.r_centers().astype(np.float32)
    depth = grid.depth_centers().astype(np.float32)
    clip_mm = np.float32(max(2.0 * cfg.travel_mm, 2.0 * cfg.min_wall_mm))

    sd2d = signed_pattern_distance(blade2d, grid.dtheta * grid.radius_mm, grid.dz)
    pattern = shear_field(sd2d.astype(np.float32), grid)
    pattern *= (r / grid.radius_mm)[None, None, :]
    np.clip(pattern, -clip_mm, clip_mm, out=pattern)

    g = gyroid_field(grid, fit.period_mm, fit.phase, fit.z_stretch).astype(np.float32)
    g_slope = float(np.percentile(physical_gradient(g, grid), 99.0))
    gyro = np.clip(g / max(g_slope, 1e-9), -clip_mm, clip_mm)

    blend = max(cfg.blend_mm, grid.dz)
    w = np.clip((cfg.cut_depth_mm + blend - depth) / blend, 0.0, 1.0)
    w = (0.5 - 0.5 * np.cos(np.pi * w)).astype(np.float32)   # weicher Uebergang
    phi = w[None, None, :] * pattern + (1.0 - w)[None, None, :] * gyro

    # Nabe: innen gehoert alles dem Ausstoesser.
    hub_field = (r - cfg.hub_radius()).astype(np.float32)
    return np.minimum(phi, hub_field[None, None, :])


def split_bodies(blade2d: np.ndarray, grid: CylGrid, fit: GyroidFit,
                 cfg: CoexistenceConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Aus der Trennflaeche werden zwei Koerper mit garantiertem Spalt.

    Der Spalt entsteht durch EROSION der beiden Haelften, nicht durch eine
    Niveaumenge. Das ist exakt und nicht nur naeherungsweise: liegt ein Punkt
    x in der erodierten Schneide und ein Punkt y im erodierten Ausstoesser,
    dann kreuzt die Verbindungsstrecke die Trennflaeche in einem Punkt p, und
    es gilt ``d(x,y) = d(x,p) + d(p,y) >= e_Schneide + e_Ausstoesser``. Die
    Summe der beiden Erosionen IST der Spalt -- ohne Bisektion, ohne
    Gradientenabschaetzung, ohne Sicherheitsaufschlag.

    Aufgeteilt wird die Summe tiefenabhaengig, weil beide Koerper sehr
    unterschiedlich viel abgeben koennen:

    - Im Aussenband ist die Klinge oft nur ein bis zwei Millimeter breit. Sie
      kann gar nichts abgeben, sonst verschwindet das Muster; dort traegt der
      Ausstoesser den ganzen Hub. Genau das ist die uebliche
      Auswerfer-Geometrie: die Platte sitzt ringsum ``travel`` von der Klinge
      entfernt.
    - Tief im Bauteil sind beide Strukturen gleich dick; dort teilen sie sich
      den Hub haelftig, was beiden Wandstaerke laesst.

    Damit die Summe auch am Uebergang stimmt, wird die Erosion des
    Ausstoessers erst ``travel`` TIEFER umgeschaltet als die der Schneide:
    zwei Punkte im Abstand des Hubs liegen hoechstens ``travel`` in der Tiefe
    auseinander, und so ist in jeder Kombination die Summe >= travel.
    """
    phi = allegiance_field(blade2d, grid, fit, cfg)
    depth = grid.depth_centers()
    travel = cfg.travel_mm
    half = travel / 2.0
    side = phi >= 0

    # Erosionsprofil ueber der Tiefe. e_b waechst monoton von 0 (Aussenband:
    # die Klinge ist dort oft nur ein bis zwei Millimeter breit und kann
    # nichts abgeben) auf den halben Hub (tief innen sind beide Strukturen
    # gleich dick und teilen sich den Hub). e_e ist der Rest -- ausgewertet
    # eine Hubtiefe WEITER INNEN, denn zwei Punkte, die sich beim Verschieben
    # um den Hub beruehren koennten, liegen hoechstens ``travel`` in der Tiefe
    # auseinander. Weil e_b monoton waechst, gilt damit in jeder Kombination
    #     e_b(d1) + e_e(d2) >= e_b(d1) + travel - e_b(d2 + travel) >= travel.
    #
    # Ein SPRUNG statt einer Rampe war hier ein echter Fehler: er legte die
    # volle 3-mm-Erosion des Ausstoessers genau in die Ueberblendzone, wo die
    # Platten in das Gyroid uebergehen -- die Uebergaenge schnuerten ab, und
    # der Ausstoesser zerfiel in 46 Teile statt in eines.
    # Die Rampe beginnt ERST UNTERHALB der Ueberblendzone. Darueber leben die
    # 45deg-Treppen des Musters, und eine Erosion nimmt ihnen von unten die
    # Auflage: das Muster steht danach in der Luft, die Stuetzreparatur trimmt
    # es weg (gemessen: 1199 schwebende Voxel in Tiefe 5 mm, daraus 15108
    # getrimmte und ein in 240 Teile zerfallener Koerper), und
    # nachwachsen kann es dort auch nicht -- der Erosionssaum ist ja gerade
    # der Streifen, der dem Ausstoesser gehoert. Tiefer unten steht statt der
    # Treppe das Gyroid, und dessen erodierte Haelften sind bei der Anpassung
    # bereits auf Stuetzfreiheit geprueft.
    def ramp(d: np.ndarray) -> np.ndarray:
        start = cfg.cut_depth_mm + cfg.blend_mm
        t = np.clip((d - start) / max(cfg.blend_mm, 1e-6), 0.0, 1.0)
        return half * t

    levels = np.linspace(0.0, half, 4)

    def eroded(mask: np.ndarray, other: np.ndarray, amounts: np.ndarray
               ) -> np.ndarray:
        """Erosion mit tiefenabhaengigem Betrag: je Stufe einmal dilatieren,
        dann pro Radiusring die passende Stufe auswaehlen (eine staerkere
        Erosion ist immer in der schwaecheren enthalten)."""
        out = np.zeros_like(mask)
        uniq = np.unique(np.round(amounts, 6))
        for amount in uniq:
            sel = np.round(amounts, 6) == amount
            if not sel.any():
                continue
            if amount <= 0:
                out[:, :, sel] = mask[:, :, sel]
            else:
                keep = mask & ~dilate_xy(other, float(amount), grid)
                out[:, :, sel] = keep[:, :, sel]
        return out

    e_blade = levels[np.abs(levels[None, :] - ramp(depth)[:, None]).argmin(axis=1)]
    e_ej_raw = travel - ramp(depth + travel)
    ej_levels = np.unique(np.concatenate([levels + half, [travel]]))
    e_ej = ej_levels[np.abs(ej_levels[None, :] - e_ej_raw[:, None]).argmin(axis=1)]
    e_ej = np.maximum(e_ej, e_ej_raw)      # nie weniger als noetig

    blade = eroded(side, ~side, e_blade)
    ejector = eroded(~side, side, e_ej)
    # Druckspiel wirkt zusaetzlich in ALLE Richtungen, auch in z.
    ejector &= ~dilate_3d(side, cfg.print_clearance_mm, grid)

    # Vor der Kavitaet steht nur die Klinge: der Hohlraum haelt den Teig, und
    # seine Tiefe IST der Auswerferhub.
    ejector[:, :, depth < travel] = False

    # Zum Schluss die Bedingung mit GENAU DEM OPERATOR durchsetzen, mit dem
    # sie hinterher auch geprueft wird. Die Erosionen sind im Kontinuum
    # exakt, auf dem Gitter aber auf ganze Zellen gerundet -- und Rundung in
    # zwei Schritten deckt sich nicht immer mit Rundung in einem. Der Rest
    # ist klein; entscheidend ist, dass "erzeugt" und "geprueft" dieselbe
    # Rechnung benutzen. Die Klinge hat Vorrang: sie ist das Produkt.
    slack = ejector & (dilate_xy(blade, travel, grid)
                       | dilate_3d(blade, cfg.print_clearance_mm, grid))
    ejector &= ~slack
    return blade, ejector, {"discretisation_slack_voxels": int(slack.sum()),
                            "blade_erosion_mm": [round(float(x), 2) for x in
                                                 np.unique(e_blade)],
                            "ejector_erosion_mm": [round(float(x), 2) for x in
                                                   np.unique(e_ej)],
                            "cavity_depth_mm": travel}


def build_state_field(blade_mask2d: np.ndarray, grid: CylGrid,
                      cfg: CoexistenceConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Aus der Mustermaske werden die drei Zustaende: leer, Schneide,
    Ausstoesser.

    ``blade_mask2d``: True == hier steht die KLINGE (die dunklen Linien des
    Bildes). Die hellen Flaechen dazwischen sind das, was der Ausstoesser
    herausdrueckt.
    """
    report: dict = {}
    warnings: list[str] = []
    nt, nz, nr = grid.shape
    r = grid.r_centers()

    blade2d = resample_mask(blade_mask2d, nt, nz)
    raw_area = float(blade2d.mean())
    # Zu duenne Klingenlinien waeren nicht druckbar -> auf Mindestwandstaerke
    # aufdicken. Das ist die einzige Stelle, an der das Muster selbst
    # veraendert wird, und sie macht es dicker statt kaputt.
    grow = max(cfg.min_wall_mm / 2.0 - grid.voxel_mm / 2.0, 0.0)
    if grow > 0:
        sd = signed_pattern_distance(blade2d, grid.dtheta * grid.radius_mm, grid.dz)
        blade2d = sd >= -grow
    report["blade_area_fraction"] = float(blade2d.mean())
    report["blade_area_fraction_raw"] = raw_area
    if not blade2d.any():
        warnings.append("Das Muster enthaelt keine Klingenlinien -- "
                        "Schwellwert erhoehen.")
    if blade2d.all():
        warnings.append("Das Muster ist vollstaendig Klinge -- "
                        "Schwellwert senken.")

    solid_depth = cfg.cut_depth_mm + cfg.blend_mm
    hub_r = cfg.hub_radius()
    r_gyro_max = grid.radius_mm - cfg.cut_depth_mm
    gyro_ring = (r >= hub_r) & (r < r_gyro_max)
    zone3d = np.zeros(grid.shape, dtype=bool)
    zone3d[:, :, gyro_ring] = True
    report["gyroid_zone_mm"] = (round(float(hub_r), 2), round(float(r_gyro_max), 2))
    if r_gyro_max - hub_r < 2 * cfg.min_wall_mm + cfg.travel_mm:
        warnings.append(
            f"Zwischen Nabe ({hub_r:.1f} mm) und Schneidentiefe "
            f"({r_gyro_max:.1f} mm) bleiben nur {r_gyro_max - hub_r:.1f} mm fuer "
            f"die Gyroid-Zone -- zu wenig fuer Hub plus zwei Waende. Radius "
            f"vergroessern, Achse duenner waehlen oder Schneidentiefe verringern."
        )

    # -- Gyroid anpassen (auf grobem Gitter) -------------------------------
    anchor_blade = project_45(blade2d, grid, 0.0, solid_depth)
    anchor_plate = project_45(~blade2d, grid, cfg.travel_mm, solid_depth)
    fit_voxel = max(grid.voxel_mm, cfg.travel_mm / 3.0)
    coarse = CylGrid.from_dimensions(grid.radius_mm, grid.height_mm, fit_voxel,
                                     cfg.max_voxels)
    if coarse.nt * coarse.nz * coarse.nr < grid.nt * grid.nz * grid.nr:
        zone_fit = coarsen(zone3d, grid, coarse)
        anchor_a_fit = coarsen(anchor_blade, grid, coarse)
        anchor_b_fit = coarsen(anchor_plate, grid, coarse)
        fit_grid = coarse
    else:
        zone_fit, anchor_a_fit, anchor_b_fit, fit_grid = (
            zone3d, anchor_blade, anchor_plate, grid)

    if cfg.gyroid_period_mm:
        periods = [float(cfg.gyroid_period_mm)]
    else:
        # Untergrenze: eine Gyroid-Masche muss den beidseitig erodierten Hub
        # plus zwei Waende noch hergeben.
        base = max(cfg.travel_mm + 2 * cfg.min_wall_mm, 1e-3)
        periods = [round(f * base, 2) for f in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5)]
    z_stretches = ([float(cfg.gyroid_z_stretch)] if cfg.gyroid_z_stretch
                   else [8.0, 4.0, 2.0])
    fit = fit_gyroid(fit_grid, zone_fit, cfg.travel_mm / 2.0, cfg.min_wall_mm,
                     periods, z_stretches, cfg.phase_candidates,
                     anchor_a_fit, anchor_b_fit)

    # -- Ein Feld, zwei Koerper -------------------------------------------
    blade, ejector, split = split_bodies(blade2d, grid, fit, cfg)

    # Die Achsbohrung geht durch beide Koerper.
    if cfg.axis_diameter_mm:
        bore = r < cfg.axis_diameter_mm / 2.0
        blade[:, :, bore] = False
        ejector[:, :, bore] = False

    report["gyroid"] = {
        "period_mm": fit.period_mm,
        "fragments": fit.fragments,
        "floating": fit.floating,
        "phase": [round(float(p), 3) for p in fit.phase],
        "z_stretch": fit.z_stretch,
        "wall_ratio_blade": round(fit.wall_ratio_a, 3),
        "wall_ratio_ejector": round(fit.wall_ratio_b, 3),
        "fit_grid_voxel_mm": round(fit_grid.voxel_mm, 2),
        "candidates": fit.tried,
    }
    report["split"] = split
    report["warnings"] = warnings
    return blade, ejector, report


# ---------------------------------------------------------------------------
# 9. Nachmessen statt hoffen
# ---------------------------------------------------------------------------

def check_xy_travel(blade: np.ndarray, ejector: np.ndarray, grid: CylGrid,
                    travel_mm: float) -> tuple[bool, int]:
    """Die Kernbedingung: verschiebt man den Ausstoesser um ``travel_mm`` in
    IRGENDEINE Richtung der XY-Ebene, darf er die Schneide nicht beruehren.

    Gemessen als Dilatation, nicht gerechnet: die Radienbuchhaltung der
    Vorgaengerversion hat schon einmal "richtig aussehende" Formeln geliefert
    und trotzdem zwei sich durchdringende Koerper."""
    hit = dilate_xy(ejector, travel_mm, grid) & blade
    return (not hit.any()), int(hit.sum())


def check_floating(occ: np.ndarray, grid: CylGrid,
                   max_overhang_deg: float = 45.0) -> int:
    """Voxel ohne Auflage im 45deg-Kegel darunter (z=0 ausgenommen)."""
    return int((occ & ~support_map(occ, grid, max_overhang_deg)).sum())


def measure(blade: np.ndarray, ejector: np.ndarray, grid: CylGrid,
            cfg: CoexistenceConfig) -> dict:
    """Alle Randbedingungen am fertigen Voxelfeld nachmessen."""
    vol = grid.voxel_volume()
    travel_ok, travel_hits = check_xy_travel(blade, ejector, grid, cfg.travel_mm)
    print_ok = not (dilate_3d(ejector, cfg.print_clearance_mm, grid) & blade).any()
    return {
        "blade_bodies": count_components(blade),
        "ejector_bodies": count_components(ejector),
        "blade_floating_voxels": check_floating(blade, grid, cfg.max_overhang_deg),
        "ejector_floating_voxels": check_floating(ejector, grid, cfg.max_overhang_deg),
        "blade_on_build_plate": bool(blade[:, 0, :].any()),
        "ejector_on_build_plate": bool(ejector[:, 0, :].any()),
        "xy_travel_ok": travel_ok,
        "xy_travel_violations": travel_hits,
        "print_clearance_ok": bool(print_ok),
        "blade_volume_mm3": float((blade.sum(axis=(0, 1)) * vol).sum()),
        "ejector_volume_mm3": float((ejector.sum(axis=(0, 1)) * vol).sum()),
        "blade_wall_ratio": min_thickness_ok(blade, cfg.min_wall_mm, grid),
        "ejector_wall_ratio": min_thickness_ok(ejector, cfg.min_wall_mm, grid),
    }


# ---------------------------------------------------------------------------
# 10. Voxel -> Mesh
# ---------------------------------------------------------------------------

def voxels_to_mesh(occ: np.ndarray, grid: CylGrid) -> trimesh.Trimesh:
    """Erzeugt die Oberflaeche der Voxelmenge direkt in Zylinderkoordinaten.

    Ausgegeben wird genau die Flaeche zwischen einem besetzten und einem
    freien Voxel -- damit ist das Ergebnis geschlossen (jede Kante gehoert zu
    genau zwei Dreiecken), solange keine reinen Kanten-/Eckkontakte
    uebrigbleiben; dafuer sorgt ``resolve_diagonal_contacts``.

    Kein Marching Cubes: die Voxelgrenzen SIND die Konstruktionsgrenzen. Jede
    Glaettung wuerde genau den Bewegungsspalt anknabbern, der vorher muehsam
    eingehalten wurde.
    """
    nt, nz, nr = grid.shape
    if not occ.any():
        return trimesh.Trimesh()

    nvz, nvr = nz + 1, nr + 1

    def vid(i, j, k):
        return ((i % nt) * nvz + j) * nvr + k

    quads: list[np.ndarray] = []

    def emit(cells, corners):
        """corners: Liste von (di, dj, dk) im Gegenuhrzeigersinn von aussen."""
        if len(cells[0]) == 0:
            return
        i, j, k = cells
        quads.append(np.stack([vid(i + di, j + dj, k + dk)
                               for di, dj, dk in corners], axis=1))

    # +r / -r (Mantelflaechen)
    outer = occ & ~np.pad(occ[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
    emit(np.nonzero(outer), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
    inner = occ & ~np.pad(occ[:, :, :-1], ((0, 0), (0, 0), (1, 0)))
    emit(np.nonzero(inner), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)])

    # +z / -z (Stirnflaechen)
    top = occ & ~np.pad(occ[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
    emit(np.nonzero(top), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)])
    bottom = occ & ~np.pad(occ[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
    emit(np.nonzero(bottom), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])

    # +theta / -theta (umlaufend -- hier gibt es keinen Rand)
    plus = occ & ~np.roll(occ, -1, axis=0)
    emit(np.nonzero(plus), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)])
    minus = occ & ~np.roll(occ, 1, axis=0)
    emit(np.nonzero(minus), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)])

    if not quads:
        return trimesh.Trimesh()
    quad = np.concatenate(quads, axis=0)
    faces = np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]], axis=0)

    used, faces_compact = np.unique(faces, return_inverse=True)
    faces_compact = faces_compact.reshape(faces.shape).astype(np.int64)
    kk = used % nvr
    jj = (used // nvr) % nvz
    ii = used // (nvr * nvz)
    th = ii * grid.dtheta
    rr = kk * grid.dr
    verts = np.stack([rr * np.cos(th), rr * np.sin(th), jj * grid.dz], axis=1)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces_compact, process=False)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


# ---------------------------------------------------------------------------
# 11. Gesamtablauf
# ---------------------------------------------------------------------------

def build_gyroid_dual_cylinder(blade_mask2d: np.ndarray, radius_mm: float,
                               height_mm: float,
                               cfg: CoexistenceConfig | None = None,
                               build_meshes: bool = True,
                               ) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Schneide und Ausstoesser aus einer Mustermaske.

    ``blade_mask2d``: True == Klinge. Achse 0 = Umfang (periodisch),
    Achse 1 = Zylinderachse = Aufbaurichtung.

    Rueckgabe: (schneide_mesh, ausstoesser_mesh, report). Der Report ist der
    eigentliche Ertrag: er sagt fuer JEDE Randbedingung, ob sie am fertigen
    Voxelfeld gemessen eingehalten ist -- und wie viel Reparatur (also
    Verzerrung gegenueber dem reinen Gyroid) dafuer noetig war.
    """
    cfg = cfg or CoexistenceConfig()
    if blade_mask2d.ndim != 2 or min(blade_mask2d.shape) < 2:
        raise ValueError(f"Maske zu klein: {blade_mask2d.shape}")
    min_radius = cfg.min_radius_mm()
    if radius_mm < min_radius:
        raise ValueError(
            f"Radius {radius_mm:.1f} mm ist zu klein: Schneidentiefe "
            f"({cfg.cut_depth_mm:.1f}) + Ueberblendung ({cfg.blend_mm:.1f}) + "
            f"Gyroid-Zone ({2 * cfg.min_wall_mm + cfg.travel_mm:.1f}) + Abstand "
            f"zur Nabe ({cfg.travel_mm + cfg.print_clearance_mm:.1f}) + Nabe "
            f"({cfg.hub_radius():.1f}) brauchen mindestens "
            f"{min_radius:.1f} mm. Radius vergroessern, Hub oder "
            f"Schneidentiefe verkleinern oder eine duennere Achse waehlen."
        )

    grid = CylGrid.from_dimensions(radius_mm, height_mm, cfg.voxel_mm,
                                   cfg.max_voxels)
    blade, ejector, report = build_state_field(blade_mask2d, grid, cfg)
    warnings: list[str] = report.pop("warnings", [])
    report["grid"] = {"n_theta": grid.nt, "n_z": grid.nz, "n_r": grid.nr,
                      "voxel_mm": round(grid.voxel_mm, 3),
                      "voxels": int(grid.nt * grid.nz * grid.nr)}
    report["before_repair"] = measure(blade, ejector, grid, cfg)

    r = grid.r_centers()
    bore = np.zeros(grid.shape, dtype=bool)
    if cfg.axis_diameter_mm:
        bore[:, :, r < cfg.axis_diameter_mm / 2.0] = True

    def allowed_for(other: np.ndarray) -> np.ndarray:
        blocked = (dilate_xy(other, cfg.travel_mm, grid)
                   | dilate_3d(other, cfg.print_clearance_mm, grid) | bore)
        return ~blocked

    depth = grid.depth_centers()
    protect = np.zeros(grid.shape, dtype=bool)
    protect[:, :, depth < cfg.cut_depth_mm] = True

    repairs: dict = {}
    # Reihenfolge: erst der Ausstoesser (er bewegt sich und traegt die Nabe),
    # dann die Schneide gegen den FERTIGEN Ausstoesser -- so kann die zweite
    # Reparatur die erste nicht wieder verletzen.
    ejector, repairs["ejector"] = repair_body(
        ejector, allowed_for(blade), protect, grid, cfg)
    blade, repairs["blade"] = repair_body(
        blade, allowed_for(ejector), protect, grid, cfg)
    report["repairs"] = repairs

    result = measure(blade, ejector, grid, cfg)
    report.update(result)

    # -- Bewertung: was ist nicht aufgegangen? -----------------------------
    if not result["xy_travel_ok"]:
        warnings.append(
            f"Der Ausstoesser kann sich nicht ueberall um {cfg.travel_mm} mm in "
            f"XY bewegen: {result['xy_travel_violations']} Voxel liegen zu dicht "
            f"an der Schneide."
        )
    lost_ok = True
    for label, key in (("Schneide", "blade"), ("Ausstoesser", "ejector")):
        dropped = report["repairs"][key].get("pattern_fragments_dropped", 0)
        vol = report["repairs"][key].get("pattern_volume_dropped_mm3", 0.0)
        if not dropped:
            continue
        too_much = vol > cfg.droppable_fragment_mm3
        lost_ok = lost_ok and not too_much
        warnings.append(
            f"{label}: {dropped} Musterdetail(s) mit zusammen {vol:.0f} mm3 "
            f"mussten entfallen -- sie waren ringsum vom Bewegungsspalt "
            f"eingeschlossen und liessen sich nicht anbinden, ohne dem "
            f"anderen Koerper den Weg zu verbauen. Als loses Teil im Bauraum "
            f"waeren sie schlimmer."
            + (f" Das ist mehr als die Toleranz von "
               f"{cfg.droppable_fragment_mm3:.0f} mm3: kleineren Hub, "
               f"groesseren Radius oder feineres Gitter waehlen."
               if too_much else "")
        )
    if result["blade_bodies"] != 1:
        warnings.append(
            f"Die Schneide besteht aus {result['blade_bodies']} Teilen -- die "
            f"Verbindungssuche hat nicht alle Fragmente erreicht "
            f"(max_link_steps = {cfg.max_link_steps})."
        )
    if result["ejector_bodies"] != 1:
        warnings.append(
            f"Der Ausstoesser besteht aus {result['ejector_bodies']} Teilen."
        )
    for label, key in (("Schneide", "blade"), ("Ausstoesser", "ejector")):
        if result[f"{key}_floating_voxels"]:
            warnings.append(
                f"{label}: {result[f'{key}_floating_voxels']} Voxel haengen "
                f"beim Drucken in der Luft (keine Auflage im "
                f"{cfg.max_overhang_deg:.0f}deg-Kegel)."
            )
        if not result[f"{key}_on_build_plate"]:
            warnings.append(f"{label} beruehrt die Druckplatte nicht.")
        if result[f"{key}_wall_ratio"] < 0.5:
            warnings.append(
                f"{label}: nur {result[f'{key}_wall_ratio'] * 100:.0f} % des "
                f"Materials sind dicker als {cfg.min_wall_mm} mm."
            )

    report["warnings"] = warnings
    report["pattern_volume_dropped_mm3"] = sum(
        report["repairs"][k].get("pattern_volume_dropped_mm3", 0.0)
        for k in ("blade", "ejector"))
    report["ok"] = (lost_ok and result["xy_travel_ok"] and result["print_clearance_ok"]
                    and result["blade_bodies"] == 1
                    and result["ejector_bodies"] == 1
                    and result["blade_floating_voxels"] == 0
                    and result["ejector_floating_voxels"] == 0
                    and result["blade_on_build_plate"]
                    and result["ejector_on_build_plate"])

    if not build_meshes:
        return blade, ejector, report

    blade_mesh = voxels_to_mesh(blade, grid)
    ejector_mesh = voxels_to_mesh(ejector, grid)
    report["blade_mesh_watertight"] = bool(blade_mesh.is_watertight)
    report["ejector_mesh_watertight"] = bool(ejector_mesh.is_watertight)
    return blade_mesh, ejector_mesh, report


def image_to_blade_mask(img_array: np.ndarray, threshold: int) -> np.ndarray:
    """True == Klinge. Dunkle Bildpixel sind die Schnittlinien; die hellen
    Flaechen dazwischen sind das, was ausgeworfen wird.

    Konvention wie im Vorgaengermodul: Achse 0 = Umfang (periodisch),
    Achse 1 = Zylinderachse."""
    return np.asarray(img_array) < threshold
