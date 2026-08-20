"""
dual_cylinder_ejector.py
=========================

Konzept D: "Rotatorischer Wisch-Auswurf" fuer den Zwei-Zylinder-Roller.

Ausgangsproblem
----------------
Die Schneiden-Schale (aussen) und der Auswerfer-Kern (innen) muessen im
GLEICHEN Bauraum koexistieren, jeweils fuer sich eigensteif sein, sich
zueinander bewegen koennen -- und das Ganze muss FDM-druckbar sein, wenn
beide Teile direkt ineinander gedruckt werden ("print-in-place").

Warum reines radiales Ausstossen scheitert
-------------------------------------------
Frei stehende Radial-Finger, die durch Loecher in der Schale nach aussen
schieben, sind (a) mechanisch schwach (Kragarme) und (b) im engen Spalt
zwischen Schale und Kern nicht supportbar -- Stuetzmaterial in einem
geschlossenen, ineinander gedruckten Hohlraum laesst sich nach dem Druck
nicht mehr entfernen.

Geloeste Loesung: reine Rotation
---------------------------------
Schale und Kern sind zwei konzentrische Rotationskoerper mit variablem
Radius r(theta, z). Der einzige Freiheitsgrad zwischen ihnen ist die
RELATIVE ROTATION um die gemeinsame Achse -- niemals eine radiale
Translation. Das hat zwei Konsequenzen:

1. Innerhalb einer Druckschicht (konstantes z) aendert sich der Radius nur
   in der Ebene (mit theta) -- es gibt nie einen Ueberhang in Z-Richtung,
   also nie einen Bruecken-/Support-Bedarf im Spalt zwischen den Teilen.
2. Beide Teile sind durchgehende Vollkoerper (Schale = Rohr, Kern =
   Vollzylinder mit Reliefmuster), keine freikragenden Finger.

Der Kern traegt an der Ruheposition (relativer Winkel 0) exakt dieselbe
Lochmaske wie die Schale (mit Erosions-Toleranz fuer Spiel) -- die
"Stopfen" fuellen die Loecher der Schale flaechenbuendig aus. Wichtig:
das Muster wird NICHT phasenverschoben aufgetragen, weil ein beliebiges
Bildmuster keinen konsistenten Rasterabstand hat, um den man verschieben
koennte. Stattdessen entsteht der Auswurf durch die Dreh-BEWEGUNG selbst:
verdreht man den Kern um ein paar Grad, wandert die Stopfenkante ueber die
Lochkante der Schale und schiebt/wischt so den anhaftenden Teig durch das
Loch. Das funktioniert fuer beliebige, unregelmaessige Bildmuster.

Der eigentliche Teufel im Detail: Masken-Topologie
----------------------------------------------------
Ein aus einem Bild per Schwellwert gewonnenes "Loch"-Muster kann die
gedruckte Schale in mehrere Teile zerfallen lassen. Statt einzelner
Symptome ("Insel", "Trennring") gilt hier eine einzige, exakte Bedingung:

    Die Materialmaske muss GENAU EINE zusammenhaengende Komponente
    bilden (8er-Nachbarschaft, theta-Achse periodisch).

Alles andere sind Spezialfaelle davon:

- INSELN: ein Materialbereich, der vollstaendig von Loechern umgeben ist
  (z.B. das Innere eines Puzzleteils oder der Punkt ueber einem "i") --
  er faellt nach dem Schneiden heraus. Solche Segmente koennen beliebig
  tief verschachtelt sein (Zelle in Zelle in Zelle).
- TRENN-RINGE: eine Bildspalte (= eine z-Schicht = ein voller Umlauf), bei
  der ALLE Theta-Werte "Loch" sind. Das zerschneidet die Schale in zwei
  unabhaengige Ringe.
- GESCHLOSSENE SCHLEIFEN UM DEN UMFANG: eine wellenfoermige Schnittlinie,
  die einmal um den Zylinder laeuft und sich selbst schliesst. Sie trennt
  die Schale genauso in zwei Teile -- ohne dass eine einzige Bildspalte
  komplett Loch waere und ohne dass eines der Teile "freischwebend"
  aussieht. Eine Regel wie "Insel = beruehrt keinen Bildrand" uebersieht
  diesen Fall vollstaendig.

Die allgemeine Verbindungsstrategie
------------------------------------
Fuer alle diese Faelle wird dieselbe Strategie benutzt (Abschnitt 4):

  1. Jedes Materialfragment ist ein Knoten eines Graphen.
  2. Zwischen benachbarten Fragmenten liegt eine Kante mit dem Gewicht des
     KUERZEST MOEGLICHEN Stegs zwischen ihnen. Die Kandidaten liefert die
     Feature-Transformation der Distanztransformation: an der
     "Wasserscheide" zweier Fragmente stossen Loch-Pixel aneinander, deren
     jeweils naechstes Material zu verschiedenen Fragmenten gehoert.
  3. Ein MINIMALER SPANNBAUM (Kruskal) waehlt daraus genau die Stege aus,
     die noetig sind, um alles zu einem Koerper zu verbinden -- die
     insgesamt kuerzesten, also mit dem geringstmoeglichen Eingriff ins
     Schnittmuster und ohne einen einzigen ueberfluessigen Steg.

Weil verschachtelte Fragmente so an ihren direkten NACHBARN haengen (und
nicht einzeln quer durch das ganze Muster zu einem "verankerten" Bereich
gezogen werden), skaliert das auch fuer Muster mit hunderten
eingeschlossenen Segmenten.

Ein voller Umlauf-Schnitt bekommt zusaetzlich mehrere, ueber den Umfang
verteilte Stege: ein einzelner minimaler Steg wuerde die beiden Haelften
des Zylinders zwar topologisch verbinden, aber mechanisch nicht tragen.

Haltbarkeit: Wandstaerken statt Haut
--------------------------------------
Die Schale ist ein Rohr mit ueberall voller Wandstaerke, in dem die
Loecher durchgehen; der Kern ist ein VOLLZYLINDER, aus dem nur die
Achsbohrung entfernt wird. Die Kollisionsfreiheit ergibt sich aus der
festen Radienstaffelung (Abschnitt 6) und der seitlichen Erosion der
Stopfen -- nicht aus einem musterfoermigen Boolean-Ausschnitt, der
frueher auch unter der Schneidenwand Material weggenommen und beide Teile
duennwandig gemacht hat.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
import trimesh


# ---------------------------------------------------------------------------
# Kleine Union-Find-Struktur (fuer die periodische Verbindungsanalyse)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# 1. Maske <-> Bild
# ---------------------------------------------------------------------------

def image_to_cut_mask(img_array: np.ndarray, threshold: int) -> np.ndarray:
    """True == wird als Loch aus der Schale geschnitten.

    Konvention fuer dieses Modul: mask.shape == (ny, nx)
    axis 0 (ny) = theta / Umfang -> PERIODISCH (umlaufend)
    axis 1 (nx) = z / Zylinderachse -> OFFEN (Stirnkappen)
    """
    return img_array < threshold


# ---------------------------------------------------------------------------
# 2. Periodische Konnektivitaetsanalyse (theta-Achse umlaufend)
# ---------------------------------------------------------------------------

def label_periodic_theta(mask: np.ndarray):
    """Wie scipy.ndimage.label, aber axis 0 (theta) ist umlaufend.

    Gibt (labels, num_labels) zurueck; labels==0 ist Hintergrund (kein Material
    an dieser Stelle in der jeweils analysierten Maske).
    """
    structure = np.ones((3, 3), dtype=int)  # 8-Konnektivitaet
    labels, num = ndimage.label(mask, structure=structure)
    if num == 0:
        return labels, 0

    uf = _UnionFind(num + 1)
    ny, nx = mask.shape
    top_row, bot_row = labels[0, :], labels[-1, :]

    # Nachbarschaft ueber die Naht theta=0 <-> theta=ny-1, inkl. Diagonalen
    for c in range(nx):
        if top_row[c] == 0 and bot_row[c] == 0:
            continue
        for dc in (-1, 0, 1):
            c2 = c + dc
            if 0 <= c2 < nx:
                a, b = bot_row[c], top_row[c2]
                if a != 0 and b != 0:
                    uf.union(a, b)

    # Kompakte Neu-Nummerierung der Wurzeln
    roots = {}
    remap = np.zeros(num + 1, dtype=int)
    next_id = 1
    for lbl in range(1, num + 1):
        r = uf.find(lbl)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        remap[lbl] = roots[r]

    merged_labels = remap[labels]
    return merged_labels, next_id - 1


# ---------------------------------------------------------------------------
# 3. Trenn-Ringe (volle Umlauf-Schnitte, die die Schale in Stuecke teilen)
# ---------------------------------------------------------------------------

def find_severing_rings(hole_mask: np.ndarray) -> list[int]:
    """Spaltenindizes (z-Schichten), an denen ueber den GESAMTEN Umfang
    geschnitten wird -- das trennt die Schale in unabhaengige Teile."""
    ny, nx = hole_mask.shape
    return [j for j in range(nx) if np.all(hole_mask[:, j])]


def _group_consecutive(cols: list[int]) -> list[list[int]]:
    """Aufeinanderfolgende Spaltenindizes zu Baendern gruppieren."""
    runs: list[list[int]] = []
    for c in sorted(cols):
        if runs and c == runs[-1][-1] + 1:
            runs[-1].append(c)
        else:
            runs.append([c])
    return runs


def fix_severing_rings(hole_mask: np.ndarray, severing_cols: list[int],
                        tab_width_px: int = 3, n_tabs: int = 2) -> np.ndarray:
    """Legt ueber jedes Trenn-Band n_tabs schmale Materialstege (mask=False).

    Die Stege sind eine STRUKTURELLE VERSTAERKUNG, keine reine Topologie-
    Reparatur: die allgemeine Verbindungsstrategie (Abschnitt 4) wuerde die
    beiden Ringhaelften mit einem einzigen minimalen Steg verbinden -- fuer
    einen vollen Umlauf-Schnitt, der das Bauteil sonst in zwei Haelften
    zerlegt, ist ein einzelner Steg mechanisch zu schwach. Deshalb werden
    hier mehrere, ueber den Umfang verteilte Stege gesetzt.

    Die Winkelpositionen werden dabei am Muster ausgerichtet: bevorzugt
    liegen die Stege dort, wo links und rechts vom Trenn-Band ohnehin
    Material steht (dann verbindet der Steg tatsaechlich zwei tragende
    Bereiche und zerstoert moeglichst wenig vom Schnittmuster).
    """
    repaired = hole_mask.copy()
    ny, nx = hole_mask.shape
    if ny == 0 or not severing_cols:
        return repaired

    material = ~hole_mask
    half = tab_width_px // 2
    min_sep = max(1, ny // (2 * max(n_tabs, 1)))

    for run in _group_consecutive(severing_cols):
        left, right = run[0] - 1, run[-1] + 1
        # Wie viel Material steht direkt links/rechts neben dem Trenn-Band?
        score = np.zeros(ny, dtype=float)
        if left >= 0:
            score += material[:, left].astype(float)
        if right < nx:
            score += material[:, right].astype(float)
        # leicht glaetten, damit ein Steg in der MITTE eines Materialbereichs
        # landet statt an dessen Kante
        if score.any():
            score = ndimage.uniform_filter1d(
                score, size=max(3, tab_width_px), mode="wrap"
            )

        chosen: list[int] = []
        candidates = list(np.argsort(-score))
        for r in candidates:
            if len(chosen) >= n_tabs:
                break
            r = int(r)
            if all(min(abs(r - c), ny - abs(r - c)) >= min_sep for c in chosen):
                chosen.append(r)
        # Falls das Muster keine n_tabs ausreichend getrennten Stellen hergibt:
        # gleichmaessig ueber den Umfang auffuellen.
        k = 0
        while len(chosen) < min(n_tabs, ny):
            r = int(round(k * ny / max(n_tabs, 1))) % ny
            if r not in chosen:
                chosen.append(r)
            k += 1
            if k > 4 * n_tabs:
                break

        for center in chosen:
            for col in run:
                for d in range(-half, half + 1):
                    repaired[(center + d) % ny, col] = False
                # Der Steg muss die Nachbarspalten tatsaechlich erreichen --
                # bei sehr duennen Materialresten kann das Bild dort selbst
                # Loch sein; dann waere der Steg wirkungslos. Die allgemeine
                # Verbindungsstrategie raeumt den Rest auf.
    return repaired


# ---------------------------------------------------------------------------
# 4. Allgemeine Verbindungsstrategie: EIN zusammenhaengender Koerper
# ---------------------------------------------------------------------------
#
# Konzept (das eigentliche Kernstueck dieses Moduls)
# --------------------------------------------------
# Die Schale ist ein Rohr, dessen Wand genau dort steht, wo die Maske
# Material sagt. Damit gilt eine einzige, exakte und allgemeine Bedingung
# fuer die Druckbarkeit:
#
#       Die Materialmaske muss GENAU EINE zusammenhaengende Komponente
#       bilden (Nachbarschaft 8er, theta-Achse periodisch).
#
# Jede andere Formulierung ("Insel = beruehrt keinen Bildrand") ist nur ein
# Spezialfall davon und uebersieht Faelle, die in echten Mustern vorkommen:
#
#   * PUZZLE-FALL: ein Segment ist rundum von Schnittlinien eingeschlossen
#     (Puzzleteil-Inneres, Punkt ueber dem i). Klassische Insel.
#   * VERSCHACHTELUNG: Insel in einer Insel in einer Insel. Wer jede Insel
#     einzeln zum naechsten "verankerten" Bereich verbindet, zieht dabei
#     lange Stege quer durch das ganze Muster.
#   * GESCHLOSSENE SCHLEIFE UM DEN UMFANG: eine Schnittlinie, die einmal
#     um den Zylinder herumlaeuft und sich selbst schliesst, ohne dabei
#     eine einzige volle Bildspalte zu treffen (z.B. eine wellenfoermige
#     Naht). Sie zerlegt die Schale in zwei Ringe -- beide beruehren den
#     Bildrand, sind also nach der alten Insel-Definition "verankert" und
#     wurden damit NICHT erkannt.
#
# Die Loesung ist dieselbe fuer alle drei Faelle und braucht keine
# Fallunterscheidung:
#
#   1. Alle Materialfragmente sind Knoten eines Graphen.
#   2. Zwischen benachbarten Fragmenten gibt es eine Kante, deren Gewicht
#      die Laenge des KUERZESTEN moeglichen Stegs zwischen ihnen ist.
#      Diese Kandidaten kommen direkt aus der Feature-Transformation
#      (Distanztransformation mit Rueckgabe des naechsten Materialpixels):
#      wo zwei Loch-Pixel nebeneinander liegen, deren jeweils naechstes
#      Material zu verschiedenen Fragmenten gehoert, verlaeuft die
#      "Wasserscheide" zwischen genau diesen beiden Fragmenten -- und der
#      dortige Uebergang ist der kuerzeste Weg zwischen ihnen.
#   3. Ein MINIMALER SPANNBAUM (Kruskal) ueber diesen Graphen waehlt genau
#      so viele Stege, wie noetig sind, um alles zu verbinden -- und zwar
#      die insgesamt kuerzesten. Jeder Steg zerstoert damit ein Minimum an
#      Schnittmuster, und es entsteht kein einziger ueberfluessiger Steg.
#
# Das ist die allgemein verwendbare Strategie: sie kennt weder "Insel" noch
# "Trennring" als Spezialfall, sondern stellt nur den Zusammenhang her --
# fuer beliebige Muster, beliebige Verschachtelungstiefe und ueber die
# theta-Naht hinweg.


def find_material_components(hole_mask: np.ndarray):
    """(labels, num) der Materialfragmente, theta periodisch."""
    return label_periodic_theta(~hole_mask)


def _shortest_bridge_candidates(material: np.ndarray, labels: np.ndarray,
                                 num: int):
    """Kuerzeste Verbindungskandidaten zwischen benachbarten Fragmenten.

    Liefert eine Liste von (laenge, label_a, label_b, (r_a, c_a), (r_b, c_b)),
    je Fragmentpaar nur den kuerzesten Kandidaten.

    Die Analyse laeuft auf einer in theta dreifach gekachelten Kopie, damit
    Verbindungen ueber die Naht theta=0 <-> theta=ny-1 genauso gefunden
    werden wie alle anderen.
    """
    ny, nx = material.shape
    if num <= 1:
        return []

    tiled_material = np.concatenate([material] * 3, axis=0)
    tiled_labels = np.concatenate([labels] * 3, axis=0)
    dist, (idx_r, idx_c) = ndimage.distance_transform_edt(
        ~tiled_material, return_indices=True
    )
    owner = tiled_labels[idx_r, idx_c]  # naechstes Fragment je Pixel

    rows = np.arange(ny, 2 * ny)
    cols = np.arange(nx)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    # Nachbarpaare: in theta (periodisch, deshalb reicht +1 im gekachelten
    # Raum) und in z (nicht periodisch -> letzte Spalte auslassen)
    pair_sets = [
        (rr, cc, rr + 1, cc),
        (rr[:, :-1], cc[:, :-1], rr[:, :-1], cc[:, :-1] + 1),
    ]

    w_all, a_all, b_all = [], [], []
    ra_all, ca_all, rb_all, cb_all = [], [], [], []
    for r0, c0, r1, c1 in pair_sets:
        o0 = owner[r0, c0]
        o1 = owner[r1, c1]
        sel = o0 != o1
        if not sel.any():
            continue
        r0, c0, r1, c1 = r0[sel], c0[sel], r1[sel], c1[sel]
        o0, o1 = o0[sel], o1[sel]
        w = dist[r0, c0] + dist[r1, c1] + 1.0
        w_all.append(w)
        a_all.append(np.minimum(o0, o1))
        b_all.append(np.maximum(o0, o1))
        # Endpunkte: die jeweils naechsten Materialpixel der beiden Seiten,
        # passend zur Reihenfolge (a=kleineres Label) sortiert
        swap = o0 > o1
        pr0, pc0 = idx_r[r0, c0], idx_c[r0, c0]
        pr1, pc1 = idx_r[r1, c1], idx_c[r1, c1]
        ra_all.append(np.where(swap, pr1, pr0))
        ca_all.append(np.where(swap, pc1, pc0))
        rb_all.append(np.where(swap, pr0, pr1))
        cb_all.append(np.where(swap, pc0, pc1))

    if not w_all:
        return []

    w = np.concatenate(w_all)
    a = np.concatenate(a_all).astype(np.int64)
    b = np.concatenate(b_all).astype(np.int64)
    ra, ca = np.concatenate(ra_all), np.concatenate(ca_all)
    rb, cb = np.concatenate(rb_all), np.concatenate(cb_all)

    # je Fragmentpaar den kuerzesten Kandidaten behalten
    key = a * (num + 1) + b
    order = np.lexsort((w, key))
    key_sorted = key[order]
    _, first = np.unique(key_sorted, return_index=True)
    pick = order[first]

    return [
        (
            float(w[i]),
            int(a[i]),
            int(b[i]),
            (int(ra[i]) % ny, int(ca[i])),
            (int(rb[i]) % ny, int(cb[i])),
        )
        for i in pick
    ]


def connect_material_components(hole_mask: np.ndarray,
                                 bridge_width_px: int = 2,
                                 max_rounds: int = 3) -> tuple[np.ndarray, dict]:
    """Verbindet ALLE Materialfragmente zu genau einem Koerper -- ueber einen
    minimalen Spannbaum der kuerzest moeglichen Stege (siehe Konzept oben).

    Gibt (reparierte Maske, Info-Dict) zurueck.
    """
    info = {
        "components_before": 0,
        "components_after": 0,
        "bridges": [],
        "bridge_pixels_added": 0,
        "rounds": 0,
        "connected": False,
    }
    repaired = hole_mask.copy()
    ny, nx = repaired.shape

    labels, num = find_material_components(repaired)
    info["components_before"] = num
    if num == 0:
        # Vollstaendig weggeschnittenes Bild -- hier ist nichts zu verbinden.
        return repaired, info
    if num == 1:
        info["components_after"] = 1
        info["connected"] = True
        return repaired, info

    before_material = (~repaired).sum()

    for _ in range(max_rounds):
        info["rounds"] += 1
        material = ~repaired
        candidates = _shortest_bridge_candidates(material, labels, num)
        if not candidates:
            break

        # Kruskal: kuerzeste Stege zuerst, nur behalten was wirklich verbindet
        candidates.sort(key=lambda e: e[0])
        uf = _UnionFind(num + 1)
        for length, a, b, pa, pb in candidates:
            if uf.find(a) == uf.find(b):
                continue
            uf.union(a, b)
            _draw_bridge(repaired, pa[0], pa[1], pb[0], pb[1], ny,
                         bridge_width_px)
            info["bridges"].append({
                "from": a, "to": b, "length_px": length,
                "at": (pa, pb),
            })

        labels, num = find_material_components(repaired)
        if num <= 1:
            break

    info["components_after"] = num
    info["connected"] = num == 1
    info["bridge_pixels_added"] = int((~repaired).sum() - before_material)
    return repaired, info


# ---------------------------------------------------------------------------
# 4b. Diagnose: klassische "Inseln" (nur noch fuer den Report)
# ---------------------------------------------------------------------------

def find_material_islands(hole_mask: np.ndarray):
    """Findet Materialfragmente, die weder den oberen noch den unteren
    Bildrand (z=0 / z=nx-1) beruehren.

    HINWEIS: das ist nur noch eine DIAGNOSE fuer den Report. Die tatsaechliche
    Reparatur richtet sich nach der schaerferen und allgemeineren Bedingung
    aus Abschnitt 4 (genau eine Komponente) -- ein randberuehrendes Fragment
    kann sehr wohl ein loses Einzelteil sein, wenn die Schnittlinie als
    geschlossene Schleife um den Umfang laeuft.
    """
    material = ~hole_mask
    labels, num = label_periodic_theta(material)
    if num == 0:
        return {}, labels

    ny, nx = hole_mask.shape
    anchored = set(labels[:, 0][material[:, 0]].tolist()) | \
               set(labels[:, nx - 1][material[:, nx - 1]].tolist())
    anchored.discard(0)

    sizes = np.bincount(labels.ravel(), minlength=num + 1)
    objects = ndimage.find_objects(labels)

    islands = {}
    for lbl in range(1, num + 1):
        if lbl in anchored or sizes[lbl] == 0:
            continue
        sl = objects[lbl - 1]
        if sl is None:
            continue
        sub = labels[sl] == lbl
        ys, xs = np.where(sub)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        islands[lbl] = {
            "size": int(sizes[lbl]),
            "pixels": (ys, xs),
            "centroid": (float(ys.mean()), float(xs.mean())),
        }
    return islands, labels


def _draw_bridge(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int,
                  ny: int, width_px: int):
    """Zeichnet eine gerade Verbindung (Bresenham) zwischen zwei Punkten und
    setzt mask=False (== Material) entlang der Linie, inkl. periodischem
    Wrap in theta-Richtung (waehlt den kuerzeren Weg um den Umfang)."""
    # kuerzeren Weg in theta-Richtung waehlen (kann ueber die Naht laufen)
    diff = r1 - r0
    if abs(diff) > ny / 2:
        if diff > 0:
            r1 -= ny
        else:
            r1 += ny

    half = max(width_px, 1) // 2
    for r, c in _bresenham(r0, c0, r1, c1):
        rr = r % ny
        for d in range(-half, half + 1):
            mask[(rr + d) % ny, c] = False


def _bresenham(r0, c0, r1, c1):
    points = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return points


# ---------------------------------------------------------------------------
# 5. Orchestrierung: Maske reparieren
# ---------------------------------------------------------------------------

def repair_cut_mask(hole_mask: np.ndarray, bridge_width_px: int = 2,
                     severing_tab_width_px: int = 3,
                     severing_n_tabs: int = 2) -> tuple[np.ndarray, dict]:
    """Macht aus einer beliebigen Lochmaske eine druckbare Lochmaske.

    Reihenfolge:
      1. Trenn-Ringe (volle Umlauf-Schnitte) mit mehreren Stegen verstaerken
         -- rein strukturell, damit ein durchtrennter Zylinder nicht nur an
         einem einzigen duennen Steg haengt.
      2. Allgemeine Verbindungsstrategie: minimaler Spannbaum ueber alle
         Materialfragmente -> genau ein zusammenhaengender Koerper.

    Gibt (reparierte Maske, Report) zurueck.
    """
    report = {
        "severing_rings_found": [],
        "severing_rings_fixed": False,
        "islands_found": 0,
        "island_details": [],
    }

    working = hole_mask.copy()

    severing = find_severing_rings(working)
    report["severing_rings_found"] = severing
    if severing:
        working = fix_severing_rings(
            working, severing, severing_tab_width_px, severing_n_tabs
        )
        report["severing_rings_fixed"] = True

    islands, _ = find_material_islands(working)
    report["islands_found"] = len(islands)
    report["island_details"] = [
        {"size": v["size"], "centroid": v["centroid"]} for v in islands.values()
    ]

    working, connect_info = connect_material_components(
        working, bridge_width_px=bridge_width_px
    )
    report["components_found"] = connect_info["components_before"]
    report["bridges_added"] = len(connect_info["bridges"])
    report["bridge_pixels_added"] = connect_info["bridge_pixels_added"]
    report["single_body"] = connect_info["connected"]
    report["components_remaining"] = connect_info["components_after"]

    # Sicherheits-Check: nach der Reparatur darf nichts mehr uebrig sein.
    remaining_islands, _ = find_material_islands(working)
    remaining_severing = find_severing_rings(working)
    report["remaining_islands"] = len(remaining_islands)
    report["remaining_severing_rings"] = remaining_severing

    return working, report


# ---------------------------------------------------------------------------
# 6. Geometrieerzeugung (Konzept D: reine Rotation, keine Radial-Finger)
# ---------------------------------------------------------------------------
#
# Radiale Zonierung (von aussen nach innen), analytisch garantiert:
#
#   R_out   = radius_mm                              Aussenflaeche der Schale
#   R_in    = R_out - wall_thickness_mm              Innenflaeche der Schale
#   R_core  = R_in  - radial_clearance_mm            Mantelflaeche des Kerns
#   R_plug  = R_out - flush_offset_mm                Oberkante der Stopfen
#
# Die Schale ist ein Rohr konstanter Wandstaerke mit durchgehenden Loechern.
# Der Kern ist ein VOLLZYLINDER (nur die Achsbohrung ist hohl) mit erhabenen
# Stopfen, die in die Loecher der Schale ragen. Weil die Stopfen seitlich um
# das Bewegungsspiel kleiner sind als die Loecher, koennen sich Schale und
# Kern niemals durchdringen -- das ist hier nicht mehr per Boolean-Carve
# erzwungen, sondern folgt direkt aus der Radienstaffelung.


def _plug_mask_for_clearance(hole_mask: np.ndarray, clearance_mm: float,
                              pixel_pitch_theta_mm: float,
                              pixel_pitch_z_mm: float) -> np.ndarray:
    """Die Stopfen des Kerns sitzen IN den Loechern der Schale (nicht unter
    deren Wand!) und sind ringsum um das Bewegungsspiel kleiner als das Loch.
    Deshalb wird die LOCH-Maske erodiert."""
    if clearance_mm <= 0:
        return hole_mask.copy()
    er_theta = max(1, int(round(clearance_mm / max(pixel_pitch_theta_mm, 1e-6))))
    er_z = max(1, int(round(clearance_mm / max(pixel_pitch_z_mm, 1e-6))))
    struct = np.ones((2 * er_theta + 1, 2 * er_z + 1), dtype=bool)
    # Die theta-Achse ist PERIODISCH: ohne umlaufendes Padding wuerde an der
    # Naht theta=0/theta=ny-1 ein Stopfen direkt neben der Schalenwand stehen
    # bleiben (er "sieht" die Wand auf der anderen Seite der Naht nicht) --
    # genau dort haben sich Schale und Kern dann durchdrungen.
    padded = np.pad(hole_mask, ((er_theta, er_theta), (0, 0)), mode="wrap")
    # border_value=1 wirkt danach nur noch auf den z-Rand (Stirnflaechen):
    # dort grenzt kein Schalenmaterial an, der Stopfen darf stehen bleiben.
    eroded = ndimage.binary_erosion(padded, structure=struct, border_value=1)
    return eroded[er_theta:er_theta + hole_mask.shape[0], :]


def _vertices_from_radius_field(radius_field: np.ndarray, radius_mm: float,
                                 height_mm: float) -> np.ndarray:
    """radius_field.shape == (ny, nx); axis0=theta (periodisch), axis1=z."""
    ny, nx = radius_field.shape
    theta = np.linspace(0, 2 * np.pi, ny, endpoint=False)
    z = np.linspace(0, height_mm, nx)
    tt, zz = np.meshgrid(theta, z, indexing="ij")  # shape (ny, nx)
    xx = radius_field * np.cos(tt)
    yy = radius_field * np.sin(tt)
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _body_mesh(vertices: np.ndarray, ny: int, nx: int) -> trimesh.Trimesh:
    faces = []
    for t in range(ny):
        t2 = (t + 1) % ny
        for z in range(nx - 1):
            p1 = t * nx + z
            p2 = t2 * nx + z
            p3 = t2 * nx + (z + 1)
            p4 = t * nx + (z + 1)
            faces.append([p1, p2, p4])
            faces.append([p2, p3, p4])
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def _cap_mesh(all_vertices: np.ndarray, edge_indices: np.ndarray,
              is_bottom: bool) -> trimesh.Trimesh:
    """Triangulates the (usually highly non-convex/star-shaped) cap boundary
    ring.

    HISTORY / BUGFIX: this used to run an unconstrained scipy Delaunay
    triangulation over the ring points and then keep only the triangles
    whose centroid tested inside the boundary polygon (shapely
    ``contains``). That approach is fundamentally unreliable for the kind
    of jagged, spiky ring shapes that come out of a busy raster pattern
    (e.g. many thin holes reaching the cap edge): legitimate boundary
    triangles near concave notches routinely have a centroid that falls
    just outside the polygon and get silently dropped, and any ring for
    which shapely flags the polygon as technically "invalid" (self-touching
    at a single point, an extremely common occurrence for spiky pixel-grid
    rings) caused the cap to be skipped ENTIRELY. Either way the result is
    a cap with holes or no cap at all -> a non-watertight shell/core mesh
    -> ``trimesh``'s boolean/volume ops on that mesh return garbage or
    raise, which is exactly the "overlap becomes NaN" symptom.

    Fix: use a proper constrained polygon triangulation (ear clipping via
    ``mapbox_earcut``) that triangulates strictly using the ring's own
    vertices/edges and therefore always closes the cap for any simple
    polygon, no matter how concave/star-shaped.
    """
    edge_vertices = all_vertices[edge_indices]
    if edge_vertices.shape[0] < 3:
        return trimesh.Trimesh()

    points_2d = edge_vertices[:, :2]
    boundary_polygon = Polygon(points_2d)
    if boundary_polygon.area == 0:
        return trimesh.Trimesh()

    try:
        tri_vertices_2d, cap_faces = trimesh.creation.triangulate_polygon(
            boundary_polygon, engine="earcut"
        )
    except Exception:
        return trimesh.Trimesh()

    if cap_faces is None or len(cap_faces) == 0:
        return trimesh.Trimesh()

    # earcut triangulates using exactly the input ring vertices (in order,
    # with the closing/repeated first point appended) -- no Steiner points
    # are inserted -- so face indices map 1:1 back onto ``edge_indices``.
    n_ring = points_2d.shape[0]
    if tri_vertices_2d.shape[0] != n_ring + 1 or cap_faces.max() >= n_ring:
        return trimesh.Trimesh()

    if is_bottom:
        cap_faces = cap_faces[:, [0, 2, 1]]
    global_faces = edge_indices[cap_faces]
    return trimesh.Trimesh(vertices=all_vertices, faces=global_faces)


def _finish_mesh(vertices: np.ndarray, ny: int, nx: int) -> trimesh.Trimesh:
    body = _body_mesh(vertices, ny, nx)
    bottom_idx = np.array([t * nx + 0 for t in range(ny)])
    top_idx = np.array([t * nx + (nx - 1) for t in range(ny)])
    bottom_cap = _cap_mesh(vertices, bottom_idx, is_bottom=True)
    top_cap = _cap_mesh(vertices, top_idx, is_bottom=False)

    parts = [m for m in (body, bottom_cap, top_cap) if not m.is_empty]
    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def _axis_cylinder(mesh: trimesh.Trimesh, axis_diameter_mm: float,
                    safety_margin: float = 0.5) -> trimesh.primitives.Cylinder:
    bounds = mesh.bounds
    h = bounds[1, 2] - bounds[0, 2] + 2 * safety_margin
    cz = (bounds[1, 2] + bounds[0, 2]) / 2
    cyl = trimesh.primitives.Cylinder(radius=axis_diameter_mm / 2, height=h, sections=32)
    cyl.apply_translation([0, 0, cz])
    return cyl


def _bore_cylinder(radius: float, height_mm: float, sections: int = 96
                    ) -> trimesh.primitives.Cylinder:
    """Zylinder von z=-margin bis z=height+margin -- zum Aushoehlen der
    Schale (Innenflaeche) bzw. fuer die Achsbohrung."""
    margin = max(1.0, 0.05 * height_mm)
    cyl = trimesh.primitives.Cylinder(
        radius=radius, height=height_mm + 2 * margin, sections=sections
    )
    cyl.apply_translation([0, 0, height_mm / 2])
    return cyl


def count_bodies(mesh: trimesh.Trimesh) -> int:
    """Anzahl zusammenhaengender Teilkoerper eines Meshes -- der finale
    Beweis, dass keine losen Fragmente ("Inseln") im Export stecken."""
    if mesh.is_empty or len(mesh.faces) == 0:
        return 0
    try:
        components = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
        )
        return len(components)
    except Exception:  # pragma: no cover - defensive
        return len(mesh.split(only_watertight=False))


def check_no_overlap(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                      volume_tolerance: float = 1e-6) -> tuple[bool, float]:
    """Rechnet die tatsaechliche Boolean-Schnittmenge zweier Meshes aus und
    gibt (ist_ueberlappungsfrei, ueberlappungsvolumen_mm3) zurueck.

    Das ist die einzige verlaessliche Pruefung dafuer, ob zwei Teile sich im
    selben Bauraum wirklich nicht beruehren -- reine Radius-Buchhaltung kann
    (wie sich hier gezeigt hat) trotz "richtig aussehender" Formeln trotzdem
    zwei sich massiv ueberlappende Volumenkoerper erzeugen.
    """
    try:
        overlap = mesh_a.intersection(mesh_b, engine="manifold")
    except Exception:  # pragma: no cover - defensive
        return False, float("nan")
    if overlap.is_empty:
        return True, 0.0
    vol = abs(overlap.volume)
    return vol <= volume_tolerance, vol


def build_dual_cylinder(
    cut_mask: np.ndarray,
    radius_mm: float,
    height_mm: float,
    wall_thickness_mm: float = 2.0,
    radial_clearance_mm: float = 0.4,
    flush_offset_mm: float = 0.2,
    axis_diameter_mm: float | None = 6.0,
    bridge_width_px: int = 2,
    cut_through: bool = True,
    verify_no_overlap: bool = True,
    min_core_wall_mm: float = 2.0,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Erzeugt Schale (Rohr mit Loechern) und Kern (Vollzylinder mit
    Stopfen) fuer das Rotations-Auswerfer-Konzept.

    cut_mask.shape == (ny, nx), True == hier wird geschnitten (Loch).
    axis 0 (ny) = Umfang/theta (periodisch), axis 1 (nx) = Achse/z (offen).

    HALTBARKEIT (das war der zweite gemeldete Fehler)
    --------------------------------------------------
    Die Vorgaengerversion hat die Schale mit einem MUSTERFOERMIGEN
    Freiraum-Koerper ausgeschnitten und dabei auch unter der Schneidenwand
    Material weggenommen -- uebrig blieb eine papierduenne Haut statt einer
    Wand. Zusaetzlich sassen die Kern-Stopfen wegen einer vertauschten
    Maske UNTER der Wand statt in den Loechern, wodurch der Kern die Schale
    von innen ausgehoehlt hat. Ergebnis: zwei duennwandige, kaum belastbare
    Teile.

    Jetzt gilt die feste Radienstaffelung aus dem Kommentar oben:
    die Schale ist ein Rohr mit ueberall voller Wandstaerke (nur die Loecher
    gehen durch), der Kern ist massiv (nur die Achsbohrung ist hohl). Die
    Kollisionsfreiheit folgt aus den Radien und der seitlichen Erosion der
    Stopfen -- sie wird zusaetzlich per Boolean nachgerechnet.
    """
    ny, nx = cut_mask.shape
    repaired_mask, report = repair_cut_mask(cut_mask, bridge_width_px=bridge_width_px)

    r_out = float(radius_mm)
    r_in = r_out - float(wall_thickness_mm)
    r_core = r_in - float(radial_clearance_mm)
    r_plug = r_out - float(flush_offset_mm)

    axis_r = (float(axis_diameter_mm) / 2.0) if (axis_diameter_mm and cut_through) else 0.0
    if r_in <= 0 or r_core <= 0:
        raise ValueError(
            f"Wandstaerke ({wall_thickness_mm} mm) + Spiel ({radial_clearance_mm} mm) "
            f"passen nicht in den Radius ({radius_mm} mm)."
        )

    warnings: list[str] = []
    core_wall = r_core - axis_r
    if axis_r > 0 and core_wall < min_core_wall_mm:
        warnings.append(
            f"Kernwand zwischen Achsbohrung und Mantel ist nur {core_wall:.2f} mm "
            f"(empfohlen >= {min_core_wall_mm:.1f} mm) -- Achse duenner waehlen "
            f"oder Radius vergroessern."
        )
    if wall_thickness_mm < 1.0:
        warnings.append(
            f"Schalenwand {wall_thickness_mm:.2f} mm ist sehr duenn "
            f"(< 1 mm ~ 2 Perimeter) -- geringe Haltbarkeit."
        )

    circumference_mm = 2 * np.pi * r_out
    pixel_pitch_theta_mm = circumference_mm / ny
    pixel_pitch_z_mm = height_mm / max(nx - 1, 1)

    # --- Kern: massiver Zylinder mit Stopfen IN den Loechern der Schale ---
    plug_mask = _plug_mask_for_clearance(
        repaired_mask, radial_clearance_mm, pixel_pitch_theta_mm, pixel_pitch_z_mm
    )
    core_field = np.where(plug_mask, r_plug, r_core)
    core_vertices = _vertices_from_radius_field(core_field, r_out, height_mm)
    core_mesh = _finish_mesh(core_vertices, ny, nx)

    # --- Schale: volle Wand ueberall, Loecher gehen durch ---
    # Ausserhalb der Loecher steht die Wand von r_in bis r_out; in den
    # Loechern wird das Radiusfeld unter r_in gelegt, sodass beim Aushoehlen
    # mit dem Innenzylinder dort nichts uebrig bleibt -> durchgehendes Loch.
    shell_field = np.where(repaired_mask, r_core, r_out)
    shell_vertices = _vertices_from_radius_field(shell_field, r_out, height_mm)
    shell_mesh = _finish_mesh(shell_vertices, ny, nx)

    bore = _bore_cylinder(r_in, height_mm)
    try:
        hollowed = shell_mesh.difference(bore, engine="manifold")
        if not hollowed.is_empty:
            shell_mesh = hollowed
            report["shell_hollowed"] = True
        else:
            report["shell_hollowed"] = False
            warnings.append("Aushoehlen der Schale lieferte ein leeres Mesh.")
    except Exception as exc:  # pragma: no cover - defensive
        report["shell_hollowed"] = False
        report["shell_hollow_error"] = str(exc)
        warnings.append(f"Aushoehlen der Schale fehlgeschlagen: {exc}")

    # --- Achsbohrung: NUR durch den Kern. Die Schale ist bereits ein Rohr
    # mit deutlich groesserem Innendurchmesser -- eine zusaetzliche Bohrung
    # wuerde dort nichts entfernen (und die Achse laeuft im Kern). ---
    if axis_diameter_mm and cut_through:
        try:
            axis_cyl = _axis_cylinder(core_mesh, axis_diameter_mm)
            cut_result = core_mesh.difference(axis_cyl, engine="manifold")
            if not cut_result.is_empty:
                core_mesh = cut_result
                report["axis_hole_cut_core"] = True
            else:
                report["axis_hole_cut_core"] = False
        except Exception as exc:  # pragma: no cover - defensive
            report["axis_hole_cut_core"] = False
            report["axis_hole_error_core"] = str(exc)
        # Die Schale bekommt keine eigene Bohrung mehr; fuer Aufrufer, die
        # den alten Report-Schluessel lesen, bleibt er aussagekraeftig.
        report["axis_hole_cut_shell"] = False

    report["plug_pixels"] = int(plug_mask.sum())
    report["radii_mm"] = {
        "shell_outer": r_out,
        "shell_inner": r_in,
        "core_outer": r_core,
        "plug_top": r_plug,
        "axis": axis_r,
    }
    report["shell_wall_thickness_mm"] = r_out - r_in
    report["core_wall_thickness_mm"] = core_wall if axis_r > 0 else r_core
    report["shell_volume_mm3"] = float(shell_mesh.volume) if shell_mesh.is_volume else float("nan")
    report["core_volume_mm3"] = float(core_mesh.volume) if core_mesh.is_volume else float("nan")
    # Massivitaet des Kerns: 1.0 == Vollzylinder (abzueglich Achsbohrung)
    nominal_core = float(np.pi * (r_core ** 2 - axis_r ** 2) * height_mm)
    if nominal_core > 0 and np.isfinite(report["core_volume_mm3"]):
        report["core_fill_ratio"] = report["core_volume_mm3"] / nominal_core
    report["shell_bodies"] = count_bodies(shell_mesh)
    report["core_bodies"] = count_bodies(core_mesh)
    if report["shell_bodies"] > 1:
        warnings.append(
            f"Schale besteht aus {report['shell_bodies']} losen Teilen -- "
            f"die Verbindungsstrategie konnte das Muster nicht vollstaendig "
            f"zusammenfuehren."
        )
    report["warnings"] = warnings

    if verify_no_overlap:
        ok, vol = check_no_overlap(shell_mesh, core_mesh)
        report["overlap_free"] = ok
        report["overlap_volume_mm3"] = vol

    return shell_mesh, core_mesh, report


# ---------------------------------------------------------------------------
# Selbsttest: genau die Grenzfaelle, um die es ging
# ---------------------------------------------------------------------------

def _make_blank(ny=60, nx=40):
    return np.zeros((ny, nx), dtype=bool)


def _disk(mask, cy, cx, r):
    ny, nx = mask.shape
    yy, xx = np.ogrid[:ny, :nx]
    dy = np.minimum(np.abs(yy - cy), ny - np.abs(yy - cy))  # periodisch in y
    dist = np.sqrt(dy ** 2 + (xx - cx) ** 2)
    mask[dist <= r] = True
    return mask


def _test_island_is_detected_and_fixed():
    """Klassischer 'Punkt ueber dem i': eine freischwebende Materialinsel,
    komplett von Loechern umgeben, beruehrt keinen Rand."""
    ny, nx = 60, 40
    hole = np.ones((ny, nx), dtype=bool)   # alles Loch...
    hole[5:15, 5:15] = False               # ...ausser ein Materialklotz weit weg vom Rand
    # Rand oben/unten bleibt Loch -> der Klotz beruehrt keinen der beiden Raender

    islands, _ = find_material_islands(hole)
    assert len(islands) == 1, f"Erwartete 1 Insel, gefunden: {len(islands)}"

    repaired, report = repair_cut_mask(hole)
    assert report["single_body"], "Maske ist nach der Reparatur nicht zusammenhaengend"
    assert report["remaining_islands"] == 0, "Insel nach Reparatur nicht mehr vorhanden sein"
    print("PASS: Insel wird erkannt und automatisch angebunden")


def _test_enclosed_puzzle_segments_are_all_connected():
    """Der gemeldete Grenzfall: ein Puzzle-artiges Muster, bei dem die
    Schnittlinien geschlossene Zellen bilden und JEDES Segment vollstaendig
    eingeschlossen ist -- inklusive Verschachtelung (Zelle in Zelle).
    Danach muss die Maske GENAU EINE Komponente sein, und der Spannbaum darf
    dafuer nur die minimal noetige Zahl an Stegen setzen (Komponenten - 1)."""
    ny, nx = 90, 70
    hole = np.zeros((ny, nx), dtype=bool)
    # Gitter aus geschlossenen Zellen (Puzzle-Raster), umlaufend in theta
    for r in range(0, ny, 30):
        hole[r:r + 2, 6:64] = True
    for c in range(6, 65, 16):
        hole[:, c:c + 2] = True
    # verschachtelte Zelle: eine Zelle bekommt noch einen inneren Ring
    hole[10:24, 40:56] = True
    hole[13:21, 43:53] = False
    hole[16:18, 46:50] = True

    _, num_before = find_material_components(hole)
    assert num_before > 5, f"Testmuster liefert nur {num_before} Fragmente"

    repaired, report = repair_cut_mask(hole)
    _, num_after = find_material_components(repaired)
    assert num_after == 1, (
        f"Nach der Reparatur sind noch {num_after} lose Fragmente uebrig"
    )
    assert report["single_body"]
    # Der Spannbaum setzt nie mehr Stege als noetig.
    assert report["bridges_added"] <= report["components_found"] - 1, (
        f"{report['bridges_added']} Stege fuer {report['components_found']} "
        f"Fragmente -- der Spannbaum haette hoechstens "
        f"{report['components_found'] - 1} setzen duerfen"
    )
    print(
        f"PASS: {num_before} eingeschlossene Puzzle-Segmente mit "
        f"{report['bridges_added']} minimalen Stegen zu einem Koerper verbunden"
    )


def _test_closed_loop_around_circumference_is_reconnected():
    """Eine wellenfoermige Schnittlinie, die sich einmal um den Umfang
    schliesst, trennt die Schale in zwei Ringe -- OHNE dass eine einzige
    Bildspalte komplett Loch waere und ohne dass eines der beiden Teile den
    Bildrand verfehlt. Die alte 'Insel = beruehrt keinen Rand'-Regel hat
    genau hier nichts gefunden und eine in zwei Teile zerfallende Schale
    ausgeliefert."""
    ny, nx = 80, 60
    hole = np.zeros((ny, nx), dtype=bool)
    for r in range(ny):
        c = int(30 + 8 * np.sin(2 * np.pi * r / ny))
        hole[r, c:c + 3] = True

    assert find_severing_rings(hole) == [], "Testfall soll KEINEN Trennring haben"
    islands, _ = find_material_islands(hole)
    assert len(islands) == 0, "Testfall soll nach alter Definition 'sauber' aussehen"
    _, num_before = find_material_components(hole)
    assert num_before == 2, f"Erwartet 2 getrennte Ringe, gefunden {num_before}"

    repaired, report = repair_cut_mask(hole)
    _, num_after = find_material_components(repaired)
    assert num_after == 1, "Geschlossene Umfangsschleife wurde nicht ueberbrueckt"
    assert report["bridges_added"] >= 1
    print("PASS: Geschlossene Schnittschleife um den Umfang wird erkannt und ueberbrueckt")


def _test_bridge_is_taken_to_the_nearest_neighbour_not_across_the_pattern():
    """Kern der Strategie: verschachtelte Fragmente werden an ihren
    NACHBARN angebunden (Spannbaum), nicht einzeln quer durch das ganze
    Muster zu einem 'verankerten' Bereich gezogen. Messbar am insgesamt
    zugefuegten Material."""
    ny, nx = 80, 60
    hole = np.zeros((ny, nx), dtype=bool)
    hole[10:70, 10:50] = True
    hole[14:66, 14:46] = False
    hole[18:62, 18:42] = True
    hole[22:58, 22:38] = False
    hole[26:54, 26:34] = True
    hole[30:50, 28:32] = False

    repaired, report = repair_cut_mask(hole)
    _, num_after = find_material_components(repaired)
    assert num_after == 1
    # Drei Ringe a 4 px Wandstaerke: die Summe der kuerzesten Stege liegt bei
    # rund 3 * 4 px * Stegbreite. Alles deutlich darueber bedeutet, dass ein
    # Steg quer durch mehrere Ringe gezogen wurde.
    assert report["bridge_pixels_added"] <= 45, (
        f"{report['bridge_pixels_added']} Steg-Pixel -- die Stege laufen quer "
        f"durch das Muster statt zum jeweils naechsten Nachbarn"
    )
    print(
        f"PASS: Verschachtelte Fragmente werden nachbarschaftlich verbunden "
        f"({report['bridge_pixels_added']} Steg-Pixel)"
    )


def _test_boundary_touching_line_is_not_an_island():
    """'Linie nach unten': ein duenner Materialsteg, der bis zum unteren
    Bildrand (Stirnkappe) laeuft, ringsum von Loechern umgeben. Das MUSS
    sicher sein (ueber die Kappe verankert), nicht als Insel zaehlen."""
    ny, nx = 60, 40
    hole = np.zeros((ny, nx), dtype=bool)
    hole[24:36, 10:nx] = True    # Lochfeld, das bis zum rechten (unteren) Bildrand reicht
    hole[28:32, 10:nx] = False   # Materialsteg mittendrin laeuft bis zum Bildrand durch

    islands, _ = find_material_islands(hole)
    assert len(islands) == 0, (
        f"Randberuehrende Linie faelschlich als Insel markiert: {len(islands)}"
    )
    severing = find_severing_rings(hole)
    assert severing == [], "Steg darf keinen vollen Umlauf-Schnitt ausloesen"
    print("PASS: Bis zum Rand laufende Linie wird korrekt NICHT als Insel gewertet")


def _test_severing_ring_is_detected_and_fixed():
    """Ein voller Umlauf-Schnitt an einer z-Schicht trennt die Schale in
    zwei unabhaengige Teile -- muss erkannt und mit Stegen repariert werden."""
    ny, nx = 60, 40
    hole = np.zeros((ny, nx), dtype=bool)
    hole[:, 20] = True   # gesamte Spalte 20 (alle Theta) ist Loch

    severing = find_severing_rings(hole)
    assert severing == [20], f"Trennring nicht korrekt erkannt: {severing}"

    repaired, report = repair_cut_mask(hole)
    assert report["severing_rings_fixed"] is True
    assert find_severing_rings(repaired) == [], "Trennring nach Reparatur noch vorhanden"
    print("PASS: Voller Umlauf-Schnitt (Trennring) wird erkannt und uebersteg-repariert")


def _test_wraparound_seam_is_one_component():
    """Eine Materialinsel, die genau auf der theta=0/theta=ny-1 Naht liegt,
    muss dank periodischer Verbindungsanalyse als EIN Stueck erkannt werden
    -- nicht als zwei separate (falsch erkannte) Inseln."""
    ny, nx = 60, 40
    hole = np.ones((ny, nx), dtype=bool)
    hole[0:4, 15:20] = False    # ein Teil oben...
    hole[ny - 4:ny, 15:20] = False  # ...ein Teil unten -> beruehren sich ueber die Naht

    islands, labels = find_material_islands(hole)
    assert len(islands) == 1, (
        f"Ueber die Naht laufende Insel wurde in {len(islands)} Teile zerrissen, erwartet 1"
    )
    print("PASS: Insel ueber die theta-Naht (Umlauf) wird als ein zusammenhaengendes Stueck erkannt")


def _test_end_to_end_mesh_smoke():
    """Baut aus einer Maske mit allen drei Fallen (Insel, Trennring,
    randlaufende Linie) tatsaechlich Schale+Kern und prueft, dass dabei
    kein Fehler auftritt und beide Meshes Flaechen haben."""
    ny, nx = 48, 32
    hole = np.zeros((ny, nx), dtype=bool)
    hole[10:38, 8:24] = True          # grosses Lochfeld
    hole[20:24, 14:18] = False        # Insel darin (Materialklotz mitten im Loch)
    hole[:, 5] = True                 # Trennring
    hole[40:44, 20:nx] = True         # Loch-Linie bis zum Rand (unkritisch)

    shell_mesh, core_mesh, report = build_dual_cylinder(
        hole, radius_mm=25.0, height_mm=40.0, wall_thickness_mm=1.5,
        axis_diameter_mm=None, cut_through=False,
    )
    assert len(shell_mesh.faces) > 0, "Schale hat keine Flaechen"
    assert len(core_mesh.faces) > 0, "Kern hat keine Flaechen"
    assert report["remaining_islands"] == 0
    assert report["remaining_severing_rings"] == []
    assert report["overlap_free"], f"Schale und Kern ueberlappen: {report['overlap_volume_mm3']} mm3"
    print(
        f"PASS: End-to-End Mesh-Erzeugung ohne Fehler "
        f"(Schale: {len(shell_mesh.faces)} Faces, Kern: {len(core_mesh.faces)} Faces)"
    )


def _test_shell_and_core_do_not_overlap():
    """Der eigentliche Kernpunkt der ganzen Diskussion: Schale und Kern
    duerfen im gleichen Bauraum liegen, aber niemals denselben Raum
    beanspruchen. Das wird hier NICHT ueber die Radius-Formeln unterstellt,
    sondern per echter Boolean-Schnittmenge nachgerechnet -- inklusive der
    Achsbohrung, also am realistischen End-Produkt."""
    ny, nx = 40, 24
    hole = np.zeros((ny, nx), dtype=bool)
    hole[8:32, 6:18] = True
    hole[16:20, 10:14] = False   # Insel

    shell_mesh, core_mesh, report = build_dual_cylinder(
        hole, radius_mm=20.0, height_mm=30.0, wall_thickness_mm=1.5,
        radial_clearance_mm=0.4, axis_diameter_mm=6.0, cut_through=True,
    )
    ok, vol = check_no_overlap(shell_mesh, core_mesh)
    assert ok, (
        f"Schale und Kern ueberlappen um {vol:.2f} mm3 -- die radiale "
        f"Zonierung ist NICHT geometrisch erzwungen worden"
    )
    print(f"PASS: Schale und Kern sind ueberlappungsfrei (Schnittvolumen: {vol:.4f} mm3)")


def _test_cap_triangulation_closes_star_shaped_ring():
    """Regressionstest fuer den urspruenglichen NaN-Overlap-Bug: die
    Endkappen-Ringe eines Schnittmusters koennen stark konkav/sternfoermig
    sein (viele unregelmaessige Zacken zwischen "Loch" und "Material"), wie
    es bei einem organischen Bildmuster (z.B. dem Puzzleteil-Muster aus dem
    Bugreport) am Kappen-Rand entsteht. Die alte Implementierung
    (unconstrained scipy-Delaunay + Zentroid-in-Polygon-Filter) hat dabei
    reproduzierbar legitime Randdreiecke verworfen und die Kappe nicht
    vollstaendig geschlossen (an genau diesem festen Seed/Profil verwarf
    sie 16 von 150 Randkanten) -- das musste als nicht-wasserdichtes Mesh
    enden. Hier direkt geprueft: die Kappe muss GENAU einen geschlossenen
    Randloop mit `ny` Kanten haben (keine Loecher, keine fehlenden
    Dreiecke)."""
    rng = np.random.default_rng(12345)
    ny = 150
    theta = np.linspace(0, 2 * np.pi, ny, endpoint=False)
    # Unregelmaessiges, aber deterministisches Radiusprofil (mehrere
    # ueberlagerte Frequenzen + Rauschen) -- deutlich realistischer als ein
    # rein periodischer Zacken-Ring und reproduzierbar fehlschlagend mit der
    # alten Delaunay/Zentroid-Implementierung.
    base = 20 + 8 * np.sin(theta * 5) + 4 * np.sin(theta * 17 + 1)
    noise = rng.normal(0, 3.0, ny)
    radius = np.clip(base + noise, 1.0, None)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(ny)
    vertices = np.column_stack([x, y, z])
    edge_indices = np.arange(ny)

    cap = _cap_mesh(vertices, edge_indices, is_bottom=True)
    assert not cap.is_empty, "Kappe fuer sternfoermigen Ring ist leer"

    boundary_edges = trimesh.grouping.group_rows(cap.edges_sorted, require_count=1)
    assert len(boundary_edges) == ny, (
        f"Kappen-Rand hat {len(boundary_edges)} offene Kanten, erwartet genau {ny} "
        f"(ein einziger geschlossener Loop) -- die Kappe hat Loecher"
    )
    print(
        f"PASS: Kappen-Triangulierung schliesst auch unregelmaessige, "
        f"sternfoermige Ringe vollstaendig ({len(cap.faces)} Dreiecke, {ny} Randkanten)"
    )


def _seamless_tileable_grid_mask(ny: int = 220, nx: int = 300) -> np.ndarray:
    """Baut eine Lochmaske, die einem nahtlos kachelbaren Muster (wie das
    Puzzleteil-Hintergrundbild, das den urspruenglichen Bug ausgeloest hat)
    nachempfunden ist: mehrere wellenfoermige Gitterlinien, die -- weil das
    Muster zum Kacheln gedacht ist -- den oberen/unteren UND linken/rechten
    Bildrand mehrfach mit Zacken durchqueren, statt sauber am Rand
    abzuschliessen."""
    xs = np.arange(nx)
    ys = np.arange(ny)
    img = np.full((ny, nx), 255, dtype=np.uint8)
    for row_base in (0, ny // 3, 2 * ny // 3, ny - 1):
        wave = (25 * np.sin(xs / 18.0)).astype(int)
        for x in xs:
            y0 = (row_base + wave[x]) % ny
            img[max(0, y0 - 2):y0 + 2, x] = 15
    for col_base in (0, nx // 4, nx // 2, 3 * nx // 4, nx - 1):
        wave = (25 * np.sin(ys / 18.0)).astype(int)
        for y in ys:
            x0 = min(max(col_base + wave[y], 0), nx - 1)
            img[y, max(0, x0 - 2):x0 + 2] = 15
    return image_to_cut_mask(img, threshold=128)


def _test_seamless_tile_pattern_produces_watertight_overlap_free_result():
    """End-to-End-Regressionstest fuer den gemeldeten Fehlerfall: ein
    nahtlos kachelbares Wellenmuster (Puzzleteil-artig) darf weder eine
    nicht-wasserdichte Schale/Kern noch ein NaN-Ueberlappungsvolumen
    erzeugen."""
    mask = _seamless_tileable_grid_mask()
    shell_mesh, core_mesh, report = build_dual_cylinder(
        mask, radius_mm=30.0, height_mm=40.0, wall_thickness_mm=2.0,
        radial_clearance_mm=0.4, axis_diameter_mm=6.0, cut_through=True,
    )
    assert shell_mesh.is_watertight, "Schale ist nach dem Kacheltest nicht wasserdicht"
    assert core_mesh.is_watertight, "Kern ist nach dem Kacheltest nicht wasserdicht"
    assert not np.isnan(report["overlap_volume_mm3"]), (
        "Ueberlappungsvolumen ist NaN -- genau der urspruenglich gemeldete Fehlerfall"
    )
    assert report["overlap_free"], (
        f"Schale und Kern ueberlappen beim Kachelmuster-Test: "
        f"{report['overlap_volume_mm3']} mm3"
    )
    print(
        "PASS: Nahtlos kachelbares Wellenmuster erzeugt wasserdichte, "
        "ueberlappungsfreie Schale+Kern (kein NaN)"
    )


def _test_shell_wall_is_full_thickness_and_core_is_solid():
    """Der zweite gemeldete Fehler: 'beide Zylinder hohl -> geringe
    Haltbarkeit'. Nachgerechnet am echten Volumen:

      * Die Schale muss ein Rohr mit VOLLER Wandstaerke sein -- ihr Volumen
        entspricht dem Kreisring (r_in..r_out) mal Hoehe, abzueglich der
        Loecher. Die alte Version hat unter der Wand zusaetzlich Material
        weggeschnitten und kam auf einen Bruchteil davon.
      * Der Kern muss MASSIV sein (nur die Achsbohrung ist hohl)."""
    ny, nx = 90, 60
    hole = np.zeros((ny, nx), dtype=bool)
    hole[20:40, 15:45] = True          # ein grosses Lochfeld
    hole[60:70, 10:30] = True

    radius, height, wall, clearance = 25.0, 40.0, 2.0, 0.4
    axis_d = 6.0
    shell, core, report = build_dual_cylinder(
        hole, radius_mm=radius, height_mm=height, wall_thickness_mm=wall,
        radial_clearance_mm=clearance, axis_diameter_mm=axis_d, cut_through=True,
    )

    r_out, r_in = radius, radius - wall
    hole_fraction = float(hole.mean())
    nominal_ring = np.pi * (r_out ** 2 - r_in ** 2) * height
    expected_shell = nominal_ring * (1.0 - hole_fraction)
    assert shell.volume > 0.75 * expected_shell, (
        f"Schalenvolumen {shell.volume:.0f} mm3 statt ~{expected_shell:.0f} mm3 -- "
        f"die Wand ist duenner als die geforderten {wall} mm"
    )
    assert report["shell_wall_thickness_mm"] == wall

    r_core = r_in - clearance
    nominal_core = np.pi * (r_core ** 2 - (axis_d / 2) ** 2) * height
    assert core.volume > 0.95 * nominal_core, (
        f"Kernvolumen {core.volume:.0f} mm3 statt ~{nominal_core:.0f} mm3 -- "
        f"der Kern ist hohl statt massiv"
    )
    assert report["core_fill_ratio"] > 0.95
    assert report["overlap_free"], report["overlap_volume_mm3"]
    print(
        f"PASS: Schale ist ein Rohr voller Wandstaerke "
        f"({shell.volume:.0f}/{expected_shell:.0f} mm3), Kern ist massiv "
        f"(Fuellgrad {report['core_fill_ratio']:.2f})"
    )


def _test_plugs_sit_in_the_holes_not_under_the_wall():
    """Die Stopfen des Kerns muessen IN den Loechern der Schale stehen (dort
    schieben sie beim Verdrehen den Teig heraus) -- nicht unter der Wand.
    Genau das war vertauscht: die Stopfen sassen auf der MATERIAL-Maske und
    haben damit die Schalenwand von innen aufgefressen.

    Geprueft am Radiusprofil der Kern-Vertices an einer Loch- und einer
    Materialposition."""
    ny, nx = 60, 48
    hole = np.zeros((ny, nx), dtype=bool)
    hole[20:40, 16:32] = True

    radius, height, wall, clearance, flush = 25.0, 40.0, 2.0, 0.4, 0.2
    _, core, report = build_dual_cylinder(
        hole, radius_mm=radius, height_mm=height, wall_thickness_mm=wall,
        radial_clearance_mm=clearance, flush_offset_mm=flush,
        axis_diameter_mm=None, cut_through=False,
    )
    r_core = radius - wall - clearance
    r_plug = radius - flush

    verts = core.vertices
    vert_r = np.linalg.norm(verts[:, :2], axis=1)
    vert_theta = np.mod(np.arctan2(verts[:, 1], verts[:, 0]), 2 * np.pi)
    vert_z = verts[:, 2]

    def band(theta_lo, theta_hi, z_lo, z_hi):
        sel = (
            (vert_theta >= 2 * np.pi * theta_lo / ny)
            & (vert_theta <= 2 * np.pi * theta_hi / ny)
            & (vert_z >= height * z_lo / (nx - 1))
            & (vert_z <= height * z_hi / (nx - 1))
        )
        return vert_r[sel]

    r_in_hole = band(24, 36, 20, 28).max()      # mitten im Lochfeld
    r_under_wall = band(2, 14, 20, 28).max()    # gleiche z-Hoehe, aber Material

    assert abs(r_in_hole - r_plug) < 1e-6, (
        f"Kern reicht am LOCH nur bis r={r_in_hole:.2f} mm, erwartet "
        f"{r_plug:.2f} mm -- der Stopfen fehlt dort"
    )
    assert r_under_wall <= radius - wall - clearance + 1e-6, (
        f"Kern reicht unter der SCHALENWAND bis r={r_under_wall:.2f} mm "
        f"(Wandinnenseite {radius - wall:.2f} mm) -- die Stopfen sitzen auf "
        f"der falschen Maske und hoehlen die Schale aus"
    )
    assert abs(r_under_wall - r_core) < 1e-6
    assert report["plug_pixels"] > 0
    print(
        f"PASS: Stopfen stehen im Loch (r={r_in_hole:.2f} mm) und nicht unter "
        f"der Wand (r={r_under_wall:.2f} mm)"
    )


def _test_plug_clearance_respects_the_theta_seam():
    """Regressionstest: die Erosion, die den Stopfen kleiner als das Loch
    macht, muss die UMLAUFENDE theta-Achse beruecksichtigen. Ohne
    periodisches Padding blieb an der Naht theta=0/theta=ny-1 ein Stopfen
    direkt neben der Schalenwand stehen -- Schale und Kern haben sich dort
    durchdrungen (im UI-Testlauf mit ~4 mm3)."""
    ny, nx = 64, 40
    hole = np.zeros((ny, nx), dtype=bool)
    # Lochfeld, das ueber die Naht laeuft und auf der anderen Seite direkt
    # an Material grenzt
    hole[0:12, 8:32] = True
    hole[ny - 12:ny, 8:32] = True

    radius, height, wall, clearance = 25.0, 40.0, 2.0, 0.4
    plug = _plug_mask_for_clearance(
        hole, clearance, 2 * np.pi * radius / ny, height / (nx - 1)
    )
    # Kein Stopfenpixel darf (periodisch in theta) an Material grenzen.
    material = ~hole
    dilated_material = ndimage.binary_dilation(
        np.pad(material, ((1, 1), (0, 0)), mode="wrap"),
        structure=np.ones((3, 3), dtype=bool),
    )[1:-1, :]
    assert not (plug & dilated_material).any(), (
        "Stopfen steht direkt neben der Schalenwand -- die Erosion hat die "
        "theta-Naht nicht beruecksichtigt"
    )

    shell, core, report = build_dual_cylinder(
        hole, radius_mm=radius, height_mm=height, wall_thickness_mm=wall,
        radial_clearance_mm=clearance, axis_diameter_mm=6.0, cut_through=True,
    )
    assert report["overlap_free"], (
        f"Schale und Kern ueberlappen an der theta-Naht: "
        f"{report['overlap_volume_mm3']} mm3"
    )
    print("PASS: Stopfen-Spiel gilt auch ueber die theta-Naht hinweg")


def _test_result_is_a_single_body_per_part():
    """Endgueltiger Beweis, dass keine losen Teile exportiert werden: Schale
    und Kern muessen aus je genau EINEM zusammenhaengenden Koerper bestehen --
    auch bei einem Muster mit eingeschlossenen Segmenten."""
    ny, nx = 72, 48
    hole = np.zeros((ny, nx), dtype=bool)
    for r in range(0, ny, 24):
        hole[r:r + 2, 6:42] = True
    for c in range(6, 43, 12):
        hole[:, c:c + 2] = True

    shell, core, report = build_dual_cylinder(
        hole, radius_mm=22.0, height_mm=35.0, wall_thickness_mm=2.0,
        axis_diameter_mm=6.0, cut_through=True,
    )
    assert report["shell_bodies"] == 1, (
        f"Schale zerfaellt in {report['shell_bodies']} lose Teile"
    )
    assert report["core_bodies"] == 1, (
        f"Kern zerfaellt in {report['core_bodies']} lose Teile"
    )
    assert shell.is_watertight and core.is_watertight
    print("PASS: Schale und Kern sind je genau ein zusammenhaengender Koerper")


def run_self_tests():
    print("=== dual_cylinder_ejector.py: Selbsttest Grenzfaelle ===")
    _test_island_is_detected_and_fixed()
    _test_enclosed_puzzle_segments_are_all_connected()
    _test_closed_loop_around_circumference_is_reconnected()
    _test_bridge_is_taken_to_the_nearest_neighbour_not_across_the_pattern()
    _test_boundary_touching_line_is_not_an_island()
    _test_severing_ring_is_detected_and_fixed()
    _test_wraparound_seam_is_one_component()
    _test_end_to_end_mesh_smoke()
    _test_shell_and_core_do_not_overlap()
    _test_cap_triangulation_closes_star_shaped_ring()
    _test_seamless_tile_pattern_produces_watertight_overlap_free_result()
    _test_shell_wall_is_full_thickness_and_core_is_solid()
    _test_plugs_sit_in_the_holes_not_under_the_wall()
    _test_plug_clearance_respects_the_theta_seam()
    _test_result_is_a_single_body_per_part()
    print("=== Alle Tests bestanden ===")


if __name__ == "__main__":
    run_self_tests()
