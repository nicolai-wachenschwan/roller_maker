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
Ein aus einem Bild per Schwellwert gewonnenes "Loch"-Muster kann zwei
Arten von strukturellen Fallen enthalten, die eine gedruckte Schale
zerstoeren wuerden:

- INSELN: ein Materialbereich, der vollstaendig von Loechern umgeben ist
  und weder den oberen noch den unteren Bildrand beruehrt (z.B. der
  Punkt ueber einem "i"). Ein solches Fragment haengt nach dem Schneiden
  in der Luft -- es faellt heraus bzw. lässt sich gar nicht erst drucken.
- TRENN-RINGE: eine Bildspalte (= eine z-Schicht = ein voller Umlauf),
  bei der ALLE Theta-Werte als "Loch" markiert sind. Das zerschneidet die
  Schale komplett in zwei unabhaengige Ringe (oben/unten), die nur noch
  ueber die Stirn-Kappen zusammenhaengen wuerden -- meist gar nicht.

Randberuehrende Linien ("Linien nach unten", die den oberen oder unteren
Bildrand erreichen) sind dagegen UNKRITISCH: sie sind ueber die
Stirnkappe des Zylinders verankert, genau wie ein Steg, der bis zum
Rand eines Stickmuster laeuft. Sie duerfen nicht faelschlich als Insel
markiert werden.

Dieses Modul erkennt beide Fallen (mit korrekter Beruecksichtigung des
Umlaufs in Theta-Richtung -- der Bildrand links/rechts ist periodisch!)
und repariert sie automatisch durch minimale Materialstege, bevor daraus
eine Geometrie erzeugt wird.
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


def fix_severing_rings(hole_mask: np.ndarray, severing_cols: list[int],
                        tab_width_px: int = 3, n_tabs: int = 2) -> np.ndarray:
    """Setzt an jeder Trenn-Spalte n_tabs schmale Materialstege (mask=False),
    gleichmaessig ueber den Umfang verteilt, um die Schale zusammenzuhalten."""
    repaired = hole_mask.copy()
    ny, nx = hole_mask.shape
    if ny == 0:
        return repaired
    half = tab_width_px // 2
    for col in severing_cols:
        for k in range(n_tabs):
            center = int(round(k * ny / n_tabs))
            for d in range(-half, half + 1):
                repaired[(center + d) % ny, col] = False
    return repaired


# ---------------------------------------------------------------------------
# 4. Material-Inseln (isolierte Materialfragmente ohne Rand-Verankerung)
# ---------------------------------------------------------------------------

def find_material_islands(hole_mask: np.ndarray):
    """Findet Materialfragmente (== ~hole_mask), die weder den oberen noch
    den unteren Bildrand (z=0 / z=nx-1) beruehren und daher nach dem
    Schneiden lose in der Luft haengen wuerden.

    Randberuehrende Fragmente sind sicher (ueber die Stirnkappe verankert)
    und werden NICHT als Insel gezaehlt -- das deckt genau den Fall
    "Linie laeuft nach unten bis zum Rand" ab.
    """
    material = ~hole_mask
    labels, num = label_periodic_theta(material)
    if num == 0:
        return {}, labels

    ny, nx = hole_mask.shape
    anchored = set(labels[:, 0][material[:, 0]].tolist()) | \
               set(labels[:, nx - 1][material[:, nx - 1]].tolist())
    anchored.discard(0)

    islands = {}
    for lbl in range(1, num + 1):
        if lbl in anchored:
            continue
        ys, xs = np.where(labels == lbl)
        if len(ys) == 0:
            continue
        islands[lbl] = {
            "size": len(ys),
            "pixels": (ys, xs),
            "centroid": (float(ys.mean()), float(xs.mean())),
        }
    return islands, labels


def bridge_islands(hole_mask: np.ndarray, islands: dict, labels: np.ndarray,
                    bridge_width_px: int = 2) -> np.ndarray:
    """Verbindet jede Insel ueber den kuerzesten Weg mit dem naechsten
    verankerten Materialbereich (periodisch in theta gedacht) durch einen
    duennen Materialsteg."""
    if not islands:
        return hole_mask.copy()

    repaired = hole_mask.copy()
    material = ~hole_mask
    ny, nx = hole_mask.shape

    # "Ziel"-Maske: verankertes Material (alles Material, das NICHT selbst
    # zu einer der gefundenen Inseln gehoert)
    island_label_set = set(islands.keys())
    is_island_pixel = np.isin(labels, list(island_label_set))
    anchored_material = material & ~is_island_pixel

    if not anchored_material.any():
        # Kein sicherer Zielbereich vorhanden -- degenerate Eingabe, nichts
        # Sinnvolles zu tun.
        return repaired

    # Fuer den periodischen Umlauf: Ziel-Maske dreifach uebereinander stapeln,
    # damit der Distanztransform auch "ueber die Naht" die kuerzeste
    # Verbindung findet.
    tiled_target = np.concatenate(
        [anchored_material, anchored_material, anchored_material], axis=0
    )
    dist, (idx_r, idx_c) = ndimage.distance_transform_edt(
        ~tiled_target, return_indices=True
    )

    for lbl, info in islands.items():
        ys, xs = info["pixels"]
        # In den gekachelten Raum verschieben (mittlere Kachel = Original)
        tys = ys + ny
        d_here = dist[tys, xs]
        best = np.argmin(d_here)
        src_r, src_c = int(ys[best]), int(xs[best])
        tgt_r_tiled = int(idx_r[tys[best], xs[best]])
        tgt_c = int(idx_c[tys[best], xs[best]])
        tgt_r = tgt_r_tiled % ny

        _draw_bridge(repaired, src_r, src_c, tgt_r, tgt_c, ny, bridge_width_px)

    return repaired


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

    for r, c in _bresenham(r0, c0, r1, c1):
        rr = r % ny
        half = width_px // 2
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
    """Repariert Trenn-Ringe und Material-Inseln. Gibt (reparierte Maske,
    Report) zurueck."""
    report = {
        "severing_rings_found": [],
        "severing_rings_fixed": False,
        "islands_found": 0,
        "island_details": [],
        "islands_bridged": 0,
    }

    working = hole_mask.copy()

    severing = find_severing_rings(working)
    report["severing_rings_found"] = severing
    if severing:
        working = fix_severing_rings(
            working, severing, severing_tab_width_px, severing_n_tabs
        )
        report["severing_rings_fixed"] = True

    islands, labels = find_material_islands(working)
    report["islands_found"] = len(islands)
    report["island_details"] = [
        {"size": v["size"], "centroid": v["centroid"]} for v in islands.values()
    ]
    if islands:
        working = bridge_islands(working, islands, labels, bridge_width_px)
        report["islands_bridged"] = len(islands)

    # Sicherheits-Check: nach der Reparatur duerfen keine Inseln/Trennringe
    # mehr uebrig sein.
    remaining_islands, _ = find_material_islands(working)
    remaining_severing = find_severing_rings(working)
    report["remaining_islands"] = len(remaining_islands)
    report["remaining_severing_rings"] = remaining_severing

    return working, report


# ---------------------------------------------------------------------------
# 6. Geometrieerzeugung (Konzept D: reine Rotation, keine Radial-Finger)
# ---------------------------------------------------------------------------

def _radial_erode_for_clearance(hole_mask: np.ndarray, clearance_mm: float,
                                 pixel_pitch_theta_mm: float,
                                 pixel_pitch_z_mm: float) -> np.ndarray:
    """Erodiert den Materialbereich (~hole_mask) leicht, damit der Kern-Stopfen
    kleiner als das Loch ist und sich reibungsfrei darin verdrehen laesst."""
    if clearance_mm <= 0:
        return ~hole_mask
    er_theta = max(1, int(round(clearance_mm / max(pixel_pitch_theta_mm, 1e-6))))
    er_z = max(1, int(round(clearance_mm / max(pixel_pitch_z_mm, 1e-6))))
    struct = np.ones((2 * er_theta + 1, 2 * er_z + 1), dtype=bool)
    return ndimage.binary_erosion(~hole_mask, structure=struct, border_value=0)


def _build_radius_field(hole_mask_shape_ny_nx, hole_or_plug_mask: np.ndarray,
                         base_r: float, raised_r: float, recessed_r: float
                         ) -> np.ndarray:
    """raised_r wo hole_or_plug_mask True ist, sonst recessed_r/base_r."""
    field = np.where(hole_or_plug_mask, raised_r, base_r)
    return field


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
    except Exception as exc:  # pragma: no cover - defensive
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
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Erzeugt Schale (mit Loechern) und Kern (mit passenden Stopfen) fuer
    das Rotations-Auswerfer-Konzept.

    cut_mask.shape == (ny, nx), True == hier wird geschnitten (Loch).
    axis 0 (ny) = Umfang/theta (periodisch), axis 1 (nx) = Achse/z (offen).

    Wichtig zur Kollisionsfreiheit: beide Teile werden zunaechst als volle
    Rotationskoerper von der Mittelachse aus aufgebaut (wie das bestehende
    Lithophane-Verfahren). Das allein GARANTIERT noch keinen Bauraum fuer
    den jeweils anderen Teil -- Kern und Schale wuerden sich sonst im
    gesamten Bereich von der Achse bis zu ihrem jeweiligen Musterradius
    ueberlappen. Deshalb wird aus der Schale explizit ein "Freiraum-Koerper"
    (Kern-Kontur + radial_clearance_mm) per Boolean-Differenz herausgeschnitten
    -- das erzwingt die radiale Zonierung geometrisch statt sie nur uebers
    Zahlenwerk zu unterstellen.
    """
    ny, nx = cut_mask.shape
    repaired_mask, report = repair_cut_mask(cut_mask, bridge_width_px=bridge_width_px)

    circumference_mm = 2 * np.pi * radius_mm
    pixel_pitch_theta_mm = circumference_mm / ny
    pixel_pitch_z_mm = height_mm / max(nx - 1, 1)

    # --- Kern: Stopfen an derselben Position wie die Loecher (Ruhelage,
    # KEINE Phasenverschiebung -- siehe Modul-Docstring), leicht erodiert
    # fuer Drehspiel ---
    plug_mask = _radial_erode_for_clearance(
        repaired_mask, radial_clearance_mm, pixel_pitch_theta_mm, pixel_pitch_z_mm
    )
    core_field = np.where(
        plug_mask,
        radius_mm - flush_offset_mm,               # Stopfen: fast buendig mit Schale
        radius_mm - wall_thickness_mm - radial_clearance_mm,  # Rest: zurueckgesetzt, Luft zur Schale
    )
    core_vertices = _vertices_from_radius_field(core_field, radius_mm, height_mm)
    core_mesh = _finish_mesh(core_vertices, ny, nx)

    # --- Freiraum-Koerper: Kern-Kontur + Sicherheitsabstand, dient NUR dazu,
    # die Schale radial auszusparen -- wird selbst nicht exportiert. ---
    clearance_field = core_field + radial_clearance_mm
    clearance_vertices = _vertices_from_radius_field(clearance_field, radius_mm, height_mm)
    clearance_mesh = _finish_mesh(clearance_vertices, ny, nx)

    # --- Schale: volle Musterkontur, danach um den Freiraum-Koerper
    # ausgespart -- das erzeugt die duenne Ring-Wand UND die Loch-Bereiche
    # (dort ist shell_field ohnehin sehr tief, siehe unten) in einem Schritt. ---
    shell_field = np.where(
        repaired_mask,
        radius_mm - wall_thickness_mm * 2.0,   # tief -> wird Loch
        radius_mm + wall_thickness_mm,          # normale Schneidenwand
    )
    shell_vertices = _vertices_from_radius_field(shell_field, radius_mm, height_mm)
    shell_mesh = _finish_mesh(shell_vertices, ny, nx)

    try:
        carved = shell_mesh.difference(clearance_mesh, engine="manifold")
        if not carved.is_empty:
            shell_mesh = carved
            report["clearance_carved"] = True
        else:
            report["clearance_carved"] = False
    except Exception as exc:  # pragma: no cover - defensive
        report["clearance_carved"] = False
        report["clearance_carve_error"] = str(exc)

    # --- Achsbohrung fuer die Handkurbel/Achse durch BEIDE Teile ---
    if axis_diameter_mm and cut_through:
        for name, mesh in (("shell", shell_mesh), ("core", core_mesh)):
            try:
                axis_cyl = _axis_cylinder(mesh, axis_diameter_mm)
                cut_result = mesh.difference(axis_cyl, engine="manifold")
                if not cut_result.is_empty:
                    if name == "shell":
                        shell_mesh = cut_result
                    else:
                        core_mesh = cut_result
                    report[f"axis_hole_cut_{name}"] = True
                else:
                    report[f"axis_hole_cut_{name}"] = False
            except Exception as exc:  # pragma: no cover - defensive
                report[f"axis_hole_cut_{name}"] = False
                report[f"axis_hole_error_{name}"] = str(exc)

    report["plug_pixels"] = int(plug_mask.sum())

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
    assert report["islands_found"] == 1
    assert report["remaining_islands"] == 0, "Insel nach Reparatur nicht mehr vorhanden sein"
    print("PASS: Insel wird erkannt und automatisch angebunden")


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


def run_self_tests():
    print("=== dual_cylinder_ejector.py: Selbsttest Grenzfaelle ===")
    _test_island_is_detected_and_fixed()
    _test_boundary_touching_line_is_not_an_island()
    _test_severing_ring_is_detected_and_fixed()
    _test_wraparound_seam_is_one_component()
    _test_end_to_end_mesh_smoke()
    _test_shell_and_core_do_not_overlap()
    print("=== Alle Tests bestanden ===")


if __name__ == "__main__":
    run_self_tests()
