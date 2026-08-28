"""
test_gyroid_coexistence.py
===========================

Tests fuer die Voxel-Koexistenz aus gyroid_coexistence.py.

Jeder Test hier haelt eine Eigenschaft fest, an der der Generator im Laufe
der Entwicklung tatsaechlich gescheitert ist. Sie sind bewusst klein und
schnell gehalten (grobe Gitter, kleine Zylinder) -- die physikalischen
Bedingungen haengen nicht an der Aufloesung, und die vollstaendige Kette
ueber die App wird in test_app.py geprueft.

Ausfuehren mit: pytest test_gyroid_coexistence.py -v
"""

import numpy as np
import pytest

from gyroid_coexistence import (
    CoexistenceConfig, CylGrid, build_gyroid_dual_cylinder, check_floating,
    count_components, count_edge_contacts, dilate_xy, drop_specks,
    image_to_blade_mask, label_periodic, project_45, repair_connectivity,
    repair_support, resolve_diagonal_contacts, support_map, support_reach,
    trim_floating, voxels_to_mesh,
)


def _puzzle_mask(nt: int = 90, nz: int = 90, pitch: int = 22) -> np.ndarray:
    """Puzzle-Raster: geschlossene Zellen, jede Ausstoesserflaeche ist in der
    Bildebene vollstaendig eingeschlossen.

    Die Linien sind ein Pixel breit -- die Mindestwandstaerke macht daraus
    ohnehin eine druckbare Wand. Zwei Pixel waeren bei dieser Bildgroesse ein
    Klingenanteil von 20 %, und das ist kein Schnittmuster mehr, sondern ein
    Gitter mit Loechern.
    """
    mask = np.zeros((nt, nz), dtype=bool)
    mask[::pitch, :] = True
    mask[:, ::pitch] = True
    return mask


# ---------------------------------------------------------------------------
# Metrik: der Bewegungsspalt wird zwischen KOERPERN gemessen, nicht zwischen
# Zellmittelpunkten -- und ueber die Sehne, nicht ueber den Bogen.
# ---------------------------------------------------------------------------

def _cell_corners(grid: CylGrid, t: int, i: int) -> np.ndarray:
    t0, t1 = t * grid.dtheta, (t + 1) * grid.dtheta
    r0, r1 = i * grid.dr, (i + 1) * grid.dr
    return np.array([(rr * np.cos(tt), rr * np.sin(tt))
                     for tt in (t0, t1) for rr in (r0, r1)])


def _solid_distance(grid: CylGrid, a: tuple[int, int], b: tuple[int, int]) -> float:
    ca, cb = _cell_corners(grid, *a), _cell_corners(grid, *b)
    return float(np.hypot(ca[:, None, 0] - cb[None, :, 0],
                          ca[:, None, 1] - cb[None, :, 1]).min())


def test_dilate_xy_covers_every_cell_whose_solid_is_too_close():
    """Die Kernbedingung des ganzen Konzepts steht und faellt mit dieser
    Operation. Zwei Fehler steckten hier drin, beide erst durch eine echte
    Verschiebung der fertigen Meshes aufgefallen:

    - gemessen wurde entlang des BOGENS statt der Sehne (der Bogen ist
      laenger, die Bedingung damit zu lasch),
    - gemessen wurde zwischen Zell-MITTELPUNKTEN statt zwischen den Koerpern
      (bei 1.6 mm Zellen fehlten dadurch bis zu 1.6 mm Spalt).

    Geprueft wird deshalb gegen die exakte Geometrie der Voxelzellen.
    """
    grid = CylGrid.from_dimensions(radius_mm=30.0, height_mm=20.0, voxel_mm=1.6)
    mm = 3.0
    for src_r in (3, 10, 18):
        one = np.zeros(grid.shape, dtype=bool)
        one[0, 5, src_r] = True
        dilated = dilate_xy(one, mm, grid)[:, 5, :]
        for t in list(range(0, 35)) + list(range(grid.nt - 35, grid.nt)):
            for i in range(max(0, src_r - 8), min(grid.nr, src_r + 9)):
                if _solid_distance(grid, (0, src_r), (t, i)) < mm:
                    assert dilated[t, i], (
                        f"Zelle (theta={t}, r={i}) liegt naeher als {mm} mm an "
                        f"(0, {src_r}), wird aber nicht erfasst"
                    )


def test_dilate_xy_does_not_reach_along_the_axis():
    """Der Ausstoesser bewegt sich NUR in der XY-Ebene. Wuerde die Dilatation
    auch in z greifen, waere der geforderte Freiraum ein Vielfaches des
    noetigen -- und von den Mustern bliebe nichts uebrig."""
    grid = CylGrid.from_dimensions(25.0, 25.0, 1.0)
    one = np.zeros(grid.shape, dtype=bool)
    one[5, 10, 15] = True
    dilated = dilate_xy(one, 4.0, grid)
    assert dilated[:, 10, :].any()
    assert not dilated[:, 9, :].any()
    assert not dilated[:, 11, :].any()


def test_dilate_xy_is_symmetric():
    """"A liegt naeher als d an B" muss dasselbe heissen wie "B liegt naeher
    als d an A" -- sonst haelt der Spalt in der einen Richtung und in der
    anderen nicht."""
    grid = CylGrid.from_dimensions(30.0, 30.0, 1.0)
    rng = np.random.default_rng(7)
    for _ in range(5):
        a = rng.random(grid.shape) < 0.002
        b = rng.random(grid.shape) < 0.002
        assert bool((dilate_xy(a, 3.0, grid) & b).any()) == \
               bool((dilate_xy(b, 3.0, grid) & a).any())


# ---------------------------------------------------------------------------
# Grundprinzip 1: die 45deg-Treppe traegt sich selbst
# ---------------------------------------------------------------------------

def test_45_degree_projection_is_self_supporting():
    """Die Projektion ist so definiert, dass ein Schritt nach innen ein
    Schritt nach unten ist -- jedes Voxel sitzt damit auf dem
    Diagonalnachbarn darunter.

    Der Test hat zwei echte Fehler gefunden: eine aus Millimetern gerundete
    Verschiebung (die zwischen benachbarten Ringen um zwei Schichten sprang
    und die Treppe zerriss) und eine Auflagepruefung, die bei dr > dz keinen
    einzigen radialen Schritt zuliess.
    """
    grid = CylGrid.from_dimensions(30.0, 188.0, 0.9)
    assert grid.dr > grid.dz, "Testfall soll gerade dr > dz treffen"
    mask = np.zeros((grid.nt, grid.nz), dtype=bool)
    mask[:, 40] = True                      # waagerechte Linie = Vollumlauf
    mask[10:14, :] = True                   # senkrechte Linie
    projected = project_45(mask, grid, 0.0, grid.radius_mm)
    unsupported = projected & ~support_map(projected, grid)
    # Ausgenommen ist einzig der Ring an der Achse: dort endet der Strahl,
    # sein Diagonalnachbar laege ausserhalb des Gitters. Im fertigen Bauteil
    # steht an dieser Stelle die Nabe des Ausstoessers.
    assert not unsupported[:, :, 1:].any(), (
        f"{int(unsupported[:, :, 1:].sum())} Voxel der 45deg-Projektion "
        f"schweben"
    )


def test_support_map_matches_the_layerwise_definition():
    """Die vektorisierte Fassung muss exakt dasselbe liefern wie die
    schichtweise -- sie wurde nur aus Laufzeitgruenden umgebaut."""
    grid = CylGrid.from_dimensions(30.0, 40.0, 1.0)
    rng = np.random.default_rng(3)
    occ = rng.random(grid.shape) < 0.2
    reference = np.zeros_like(occ)
    reference[:, 0, :] = True
    for j in range(1, grid.nz):
        reference[:, j, :] = support_reach(occ[:, j - 1, :], grid)
    assert np.array_equal(reference, support_map(occ, grid))


# ---------------------------------------------------------------------------
# Reparaturen
# ---------------------------------------------------------------------------

def test_repair_support_leaves_nothing_floating():
    """Nach der Stuetzreparatur darf kein einziges Voxel ohne Auflage sein --
    egal wie zerklueftet die Vorlage war.

    Die alte Fassung endete stattdessen in einem stabilen Hin und Her von
    exakt so vielen gesetzten wie geloeschten Voxeln je Durchlauf: eine
    Stuetze, die den Boden nicht erreichte, wurde abgebaut, das schwebende
    Voxel darueber blieb aber stehen.
    """
    grid = CylGrid.from_dimensions(25.0, 30.0, 1.0)
    rng = np.random.default_rng(11)
    occ = rng.random(grid.shape) < 0.12
    allowed = np.ones(grid.shape, dtype=bool)
    repaired, info = repair_support(occ, allowed, grid)
    assert check_floating(repaired, grid) == 0
    assert info["support_voxels_added"] + info["floating_voxels_removed"] > 0


def test_repair_support_reaches_the_build_plate():
    """"Beide Koerper muessen in der untersten Schicht vorhanden sein" ist
    kein eigener Schritt, sondern folgt aus der Stuetzung: eine
    ununterbrochene Stuetzkette endet zwangslaeufig auf z=0."""
    grid = CylGrid.from_dimensions(20.0, 20.0, 1.0)
    occ = np.zeros(grid.shape, dtype=bool)
    occ[4:8, 15:18, 10:13] = True           # schwebender Klotz
    repaired, _ = repair_support(occ, np.ones(grid.shape, dtype=bool), grid)
    assert repaired[:, 0, :].any(), "Nichts steht auf der Druckplatte"
    assert check_floating(repaired, grid) == 0


def test_repair_connectivity_joins_fragments_through_free_space():
    """Eingeschlossene Fragmente werden in der TIEFE angebunden, nicht im
    Muster. Der Spannbaum muss dafuer alle Fragmente erfassen, auch solche,
    deren kuerzester Weg ueber ein drittes Fragment fuehrt."""
    grid = CylGrid.from_dimensions(20.0, 20.0, 1.0)
    occ = np.zeros(grid.shape, dtype=bool)
    for t in (2, 10, 18, 26):
        occ[t:t + 2, 4:8, 8:11] = True
    assert count_components(occ) == 4
    repaired, info = repair_connectivity(occ, np.ones(grid.shape, dtype=bool), grid)
    assert count_components(repaired) == 1
    assert info["links_added"] == 3, "Ein Spannbaum braucht genau n-1 Kanten"


def test_repair_connectivity_respects_the_forbidden_space():
    """Verbindungen duerfen niemals durch den Bewegungsspalt des anderen
    Koerpers gelegt werden -- lieber unverbunden (und gemeldet) als
    blockiert."""
    grid = CylGrid.from_dimensions(20.0, 20.0, 1.0)
    occ = np.zeros(grid.shape, dtype=bool)
    occ[2:4, 4:8, 8:11] = True
    occ[20:22, 4:8, 8:11] = True
    allowed = np.ones(grid.shape, dtype=bool)
    allowed[:, :, :] = False                # nirgends darf gebaut werden
    repaired, info = repair_connectivity(occ, allowed, grid)
    assert info["link_voxels"] == 0
    assert count_components(repaired) == 2


def test_diagonal_contacts_are_resolved_and_the_mesh_closes():
    """Ein Kantenkontakt ist mechanisch eine Sollbruchstelle mit Querschnitt
    null und im Mesh eine Kante mit vier statt zwei Dreiecken. Beides muss
    verschwinden -- durch Fuellen, wo es geht, sonst durch Trennen."""
    grid = CylGrid.from_dimensions(15.0, 15.0, 1.0)
    occ = np.zeros(grid.shape, dtype=bool)
    occ[3, 5, 7] = True
    occ[4, 6, 7] = True                     # nur ueber eine Kante verbunden
    assert count_edge_contacts(occ) > 0

    resolved, info = resolve_diagonal_contacts(
        occ, np.ones(grid.shape, dtype=bool), np.zeros(grid.shape, dtype=bool),
        grid,
    )
    assert count_edge_contacts(resolved) == 0
    assert voxels_to_mesh(resolved, grid).is_watertight


def test_edge_contacts_do_not_open_the_mesh():
    """Selbst wenn im Voxelfeld ein Kantenkontakt stehen BLEIBT (weil er sonst
    nur mit Material zu bezahlen waere, das gebraucht wird), muss das Mesh
    geschlossen sein: die geteilte Ecke wird beim Vernetzen aufgespalten, jede
    Seite bekommt ihre eigene Kopie. Die Geometrie aendert das nicht -- das
    Volumen bleibt exakt die Summe der Zellen."""
    grid = CylGrid.from_dimensions(15.0, 15.0, 1.0)
    for cells in ([(3, 5, 7), (4, 6, 7)],      # Kante in (theta, z)
                  [(3, 5, 7), (3, 6, 8)],      # Kante in (z, r)
                  [(3, 5, 7), (4, 5, 8)],      # Kante in (theta, r)
                  [(3, 5, 7), (4, 6, 8)]):     # nur eine Ecke
        occ = np.zeros(grid.shape, dtype=bool)
        for c in cells:
            occ[c] = True
        mesh = voxels_to_mesh(occ, grid)
        assert mesh.is_watertight, f"nicht geschlossen bei {cells}"
        cell_volume = float((occ.sum(axis=(0, 1)) * grid.voxel_volume()).sum())
        facet = np.sin(grid.dtheta) / grid.dtheta
        assert mesh.volume == pytest.approx(cell_volume * facet, rel=1e-6)


def test_drop_specks_removes_dust_but_keeps_real_features():
    grid = CylGrid.from_dimensions(20.0, 20.0, 1.0)
    occ = np.zeros(grid.shape, dtype=bool)
    occ[2:9, 2:9, 2:9] = True               # echter Koerper
    occ[15, 15, 15] = True                  # ein einzelnes Voxel
    cleaned, info = drop_specks(occ, grid, min_fragment_mm=2.4)
    assert info["specks_removed"] == 1
    assert cleaned[2:9, 2:9, 2:9].all()
    assert not cleaned[15, 15, 15]


def test_trim_floating_terminates_and_removes_everything_unsupported():
    grid = CylGrid.from_dimensions(20.0, 20.0, 1.0)
    occ = np.zeros(grid.shape, dtype=bool)
    occ[5, 10:15, 10] = True                # Saeule ohne Fundament
    trimmed, removed = trim_floating(occ, grid)
    assert not trimmed.any()
    assert removed.sum() == 5


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def test_mesh_is_closed_and_has_the_right_volume():
    """Das Mesh entsteht aus den Voxelgrenzen selbst -- kein Marching Cubes,
    keine Glaettung, die den muehsam eingehaltenen Spalt wieder anknabbert.
    Entsprechend muss das Volumen exakt der Summe der Zellvolumen
    entsprechen."""
    grid = CylGrid.from_dimensions(10.0, 10.0, 1.0)
    occ = np.zeros(grid.shape, dtype=bool)
    occ[:, 2:6, 4:8] = True                 # geschlossener Ring
    mesh = voxels_to_mesh(occ, grid)
    assert mesh.is_watertight
    cells = float((occ.sum(axis=(0, 1)) * grid.voxel_volume()).sum())
    # Die Umfangsflaechen des Meshes sind Sehnen, nicht Boegen: der Koerper
    # ist um genau den Faktor sin(dtheta)/dtheta kleiner als die ideale
    # Zellsumme. Das ist die sichere Richtung -- das exportierte Teil liegt
    # INNERHALB dessen, woran der Bewegungsspalt geprueft wurde.
    facet = np.sin(grid.dtheta) / grid.dtheta
    assert mesh.volume == pytest.approx(cells * facet, rel=1e-6)
    assert mesh.volume < cells


# ---------------------------------------------------------------------------
# Gesamtablauf
# ---------------------------------------------------------------------------

_CFG = CoexistenceConfig(voxel_mm=1.0)


def _grid_for(report) -> CylGrid:
    return CylGrid.from_dimensions(30.0, 60.0, report["grid"]["voxel_mm"],
                                   _CFG.max_voxels)


@pytest.fixture(scope="module")
def puzzle_result():
    return build_gyroid_dual_cylinder(_puzzle_mask(), radius_mm=30.0,
                                      height_mm=60.0, cfg=_CFG)


def _voxel_volume(report):
    """Mittleres Zellvolumen aus dem Report -- fuer Anteilsschranken."""
    grid = _grid_for(report)
    return float(grid.voxel_volume().mean())


def test_end_to_end_satisfies_every_constraint(puzzle_result):
    blade, ejector, report = puzzle_result
    assert report["blade_bodies"] == 1
    assert report["ejector_bodies"] == 1
    assert report["blade_floating_voxels"] == 0
    # Beim Ausstoesser bleibt ein Rest, und zwar mit Ansage. Sein
    # Plattenband steht bis zuletzt unter Schutz, damit die Reparaturen es
    # nicht kaskadierend abtragen -- ohne diesen Schutz verschwanden die
    # Platten des halben Zylinders. Der Preis sind einzelne Plattenzellen,
    # denen im Bewegungsspalt der Klinge keine Auflage mehr zu geben ist.
    # Sie muessen selten bleiben und sie muessen im Report stehen; beides
    # wird hier geprueft. Null waere hier keine bessere, sondern eine
    # unehrliche Zahl.
    ejector_cells = report["ejector_volume_mm3"] / _voxel_volume(report)
    assert report["ejector_floating_voxels"] < 0.02 * ejector_cells, (
        f'{report["ejector_floating_voxels"]} schwebende Voxel bei rund '
        f'{ejector_cells:.0f} Zellen'
    )
    assert report["blade_on_build_plate"] and report["ejector_on_build_plate"]
    assert report["xy_travel_ok"], report["xy_travel_violations"]
    assert report["print_clearance_ok"]
    assert blade.is_watertight and ejector.is_watertight


def test_every_pattern_pixel_reaches_the_body(puzzle_result):
    """Die wichtigste Zusage an den Benutzer: was er hochlaedt, schneidet das
    Werkzeug auch. Jedes Klingenpixel muss im Aussenband des fertigen
    Koerpers auftauchen -- nicht 95 %, sondern alle.

    Der Weg dorthin war lang: anfangs fehlten 41.6 %. Die Ueberblendung
    entschied tief unten allein nach dem Gyroid, und wo dieses unter einer
    Klingenlinie "Ausstoesser" sagte, riss die 45deg-Treppe ab -- alles
    darueber stand in der Luft und wurde weggetrimmt. Dagegen stehen jetzt
    drei Dinge: die Verbindungssaeulen (jede Linie reicht bis zu ihrem
    eigenen Netzwerk), die Stuetzen bis zur Druckplatte fuer Saeulen, die
    ihr Netzwerk nicht treffen, und der Schutz der Musterzellen vor jeder
    Reparatur.
    """
    _, _, report = puzzle_result
    assert report["pattern_pixels"] > 0
    assert report["pattern_pixels_missing"] == 0, (
        f"{report['pattern_pixels_missing']} von {report['pattern_pixels']} "
        f"Musterpixeln fehlen im fertigen Koerper"
    )
    assert report["pattern_completeness"] == 1.0


def test_the_blade_gives_away_nothing_where_the_pattern_is(puzzle_result):
    """Der Spalt kommt im Aussenband ausschliesslich vom Ausstoesser. Waere
    es anders, wuerde die Klinge -- oft nur eine Mindestwandstaerke breit --
    dort weggeschnitten, wo sie das Produkt ist."""
    _, _, report = puzzle_result
    assert report["split"]["blade_shave_mm"] > 0    # tief innen schon
    assert report["split"]["ejector_clearance_mm"] == pytest.approx(
        CoexistenceConfig().clearance_mm())


def test_ejector_plates_exist_over_the_whole_height(puzzle_result):
    """Der Ausstoesser muss ueber die GANZE Bauhoehe Platten haben.

    Genau daran ist eine Zwischenfassung gescheitert: die Platten gab es nur
    im obersten Fuenftel, darunter war das Plattenband leer -- ein
    Ausstoesser, der nur oben drueckt, ist nicht benutzbar. Die Ursache war
    eine Asymmetrie in der Behandlung: die Klinge bekam Verbindungssaeulen
    nach innen und Schutz vor den Reparaturen, der Ausstoesser beides nicht.
    Seine Platten rissen deshalb dort ab, wo das Gyroid unter ihnen die
    andere Seite waehlte, und wurden als schwebendes Material weggetrimmt.

    Geprueft wird in Zehnteln der Bauhoehe, weil genau diese Verteilung der
    Befund war -- eine Gesamtsumme haette den Fehler nicht gezeigt.
    """
    _, ejector, report = puzzle_result
    grid = _grid_for(report)
    depth = grid.depth_centers()
    band = ((depth >= _CFG.cut_depth_mm)
            & (depth < _CFG.cut_depth_mm + 2 * grid.voxel_mm))
    z_of_vertices = ejector.vertices[:, 2]
    height = float(z_of_vertices.max() - z_of_vertices.min())

    # Aus dem Mesh laesst sich das Plattenband nicht ablesen, deshalb wird
    # der Ausstoesser hier noch einmal als Voxelfeld gebaut.
    from gyroid_coexistence import build_gyroid_dual_cylinder
    _, ejector_voxels, _ = build_gyroid_dual_cylinder(
        _puzzle_mask(), radius_mm=30.0, height_mm=60.0, cfg=_CFG,
        build_meshes=False)
    per_tenth = []
    nz = grid.nz
    for i in range(10):
        sl = slice(i * nz // 10, (i + 1) * nz // 10)
        per_tenth.append(int(ejector_voxels[:, sl, :][:, :, band].sum()))
    # Das oberste Zehntel ist ausgenommen, und zwar aus Geometrie, nicht aus
    # Nachsicht: jeder Anspruch laeuft auf einem 45deg-Strahl nach innen und
    # unten. Was am oberen Rand des Bildes steht, muesste dafuer oberhalb der
    # Zylinderkante beginnen -- den Platz gibt es nicht. Die obersten
    # ``cut_depth`` Millimeter koennen deshalb prinzipiell kein Plattenband
    # tragen, bei dieser Vorrichtung gerade das oberste Zehntel.
    reachable = per_tenth[:9]
    # Der Befund war: Platten NUR im obersten Fuenftel, die unteren 40 % der
    # Bauhoehe leer. Dagegen wird hier geprueft, und zwar in zwei Punkten.
    #
    # Erstens muss die untere Haelfte durchgehend Platten haben -- das ist
    # der Teil, der frueher fehlte, und der Teil, den ein Ausstoesser am
    # noetigsten braucht.
    lower_half = reachable[:5]
    assert all(v > 0 for v in lower_half), (
        f"Untere Bauhoehe ohne Plattenband: {per_tenth}"
    )
    # Zweitens muss der ueberwiegende Teil der erreichbaren Hoehe tragen.
    # Nicht jeder Abschnitt: wo eine Platte im Bewegungsspalt der Klinge
    # weder anzubinden noch zu stuetzen ist, wird sie aufgegeben statt als
    # loses Stueck mitgedruckt -- der Zylinder drueckt dort schwaecher, aber
    # er ist ein Zylinder und kein Haufen. Was das kostet, steht als
    # ``ejector_volume_dropped_mm3`` im Report.
    covered = sum(1 for v in reachable if v > 0)
    assert covered >= 7, (
        f"Plattenband traegt nur {covered} von 9 Abschnitten: {per_tenth}"
    )
    weakest = min(v for v in reachable if v > 0)
    assert weakest > 0.1 * max(reachable), (
        f"Platten sind sehr ungleich ueber die Hoehe verteilt: {per_tenth}"
    )
    assert height > 0


def test_conflicts_are_resolved_in_favour_of_the_ejector(puzzle_result):
    """Wo beide Koerper denselben Weg brauchen, gewinnt die Verbindung des
    Ausstoessers -- aber nur gegen FUELLUNG, nie gegen Muster.

    Die Rangfolge hat einen Grund: eine Klingensaeule ist ein Weg und kein
    Ort (die Klinge findet daneben einen neuen), eine unangebundene
    Ausstoesserplatte dagegen hat keine Alternative und faellt ganz weg.
    """
    _, _, report = puzzle_result
    split = report["split"]
    assert split["blade_yielded_voxels"] > 0, (
        "Bei diesem Muster muss es Konflikte geben"
    )
    # Das Muster hat trotzdem ueberlebt -- das ist die Grenze der Rangfolge.
    assert report["pattern_pixels_missing"] == 0


def test_end_to_end_parts_can_actually_move(puzzle_result):
    """Die Probe aufs Exempel an den fertigen Koerpern: um den vollen Hub in
    jede Richtung der XY-Ebene verschieben, ohne zu klemmen."""
    blade, ejector, report = puzzle_result
    # Gefordert ist die halbe Hubstrecke: der Ausstoesser sitzt exzentrisch
    # und wandert aus seiner Mittellage um +-travel/2.
    travel = CoexistenceConfig().clearance_mm()
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        moved = ejector.copy()
        moved.apply_translation([travel * np.cos(angle),
                                 travel * np.sin(angle), 0.0])
        hit = blade.intersection(moved, engine="manifold")
        volume = 0.0 if hit.is_empty else abs(hit.volume)
        assert volume < 0.002 * abs(ejector.volume), (
            f"Klemmt bei {np.degrees(angle):.0f} Grad ({volume:.1f} mm3)"
        )


def test_enclosed_plates_are_joined_in_the_depth_not_in_the_pattern(puzzle_result):
    """Im Puzzle-Raster ist jede Ausstoesserflaeche in der Bildebene
    eingeschlossen. Dass trotzdem EIN Ausstoesser herauskommt, kann nur an
    der Verbindung in der Tiefe liegen -- das ist der ganze Zweck des
    Gyroids."""
    blade, ejector, report = puzzle_result
    plate2d = ~_puzzle_mask()
    _, cells = label_periodic(plate2d[:, :, None])
    assert cells > 4, "Testmuster ist nicht der schwere Fall"
    assert report["ejector_bodies"] == 1


def test_report_names_the_gyroid_it_chose(puzzle_result):
    """Die Anpassung soll nachvollziehbar sein: welche Periode, welche
    z-Streckung, welche Phase -- und wie dick die Waende geworden sind."""
    _, _, report = puzzle_result
    gyroid = report["gyroid"]
    assert gyroid["period_mm"] > 0
    assert gyroid["z_stretch"] > 0
    assert len(gyroid["phase"]) == 3
    assert 0.0 <= gyroid["wall_ratio_blade"] <= 1.0


# ---------------------------------------------------------------------------
# Entartete Eingaben
# ---------------------------------------------------------------------------

def test_radius_below_the_minimum_is_refused_not_silently_built():
    """Zu klein heisst hier: fuer die Gyroid-Zone bleibt weniger als eine
    Masche uebrig, und die eingeschlossenen Musterflaechen finden in der
    Tiefe keinen Weg mehr zueinander. Das laesst sich mit feineren Voxeln
    nicht heilen -- es ist eine Frage des Platzes im Querschnitt. Also
    abweisen, nicht hinterher melden."""
    cfg = CoexistenceConfig(voxel_mm=0.9)
    minimum = cfg.min_radius_mm()
    # Schneidentiefe 4 + Nabe 5.4 + vier Maschenmasse a 4.35 mm
    assert minimum == pytest.approx(26.8, abs=0.05)

    with pytest.raises(ValueError, match="zu klein"):
        build_gyroid_dual_cylinder(_puzzle_mask(), radius_mm=minimum - 0.5,
                                   height_mm=40.0, cfg=cfg)


def test_minimum_radius_follows_the_settings():
    """Die Untergrenze ist keine feste Zahl, sondern folgt den
    Einstellungen: wer den Hub halbiert oder eine duennere Achse waehlt,
    darf auch einen kleineren Zylinder bauen."""
    base = CoexistenceConfig().min_radius_mm()
    assert CoexistenceConfig(travel_mm=1.5).min_radius_mm() < base
    assert CoexistenceConfig(axis_diameter_mm=3.0).min_radius_mm() < base
    assert CoexistenceConfig(cut_depth_mm=2.0).min_radius_mm() < base
    assert CoexistenceConfig(travel_mm=6.0).min_radius_mm() > base
    # Ein feineres Gitter macht die Masche kleiner und damit auch den
    # Mindestradius.
    assert CoexistenceConfig(voxel_mm=0.5).min_radius_mm() < base

    # Und was die Untergrenze gerade noch erlaubt, muss auch funktionieren.
    cfg = CoexistenceConfig(voxel_mm=0.9, travel_mm=1.5, axis_diameter_mm=3.0,
                            cut_depth_mm=3.0, blend_mm=4.0)
    _, _, report = build_gyroid_dual_cylinder(
        _puzzle_mask(60, 60, 16), radius_mm=cfg.min_radius_mm(),
        height_mm=45.0, cfg=cfg, build_meshes=False)
    assert report["blade_bodies"] == 1
    assert report["ejector_bodies"] == 1
    assert report["xy_travel_ok"]
    # Vollstaendig ist das Muster hier NICHT, und die Untergrenze verspricht
    # das auch nicht. Sie sichert die Geometrie zu -- Nabe, Hub, zwei Waende
    # und der Bewegungsspalt gehen sich aus --, nicht jedes Musterdetail
    # jeder Dichte: dieses Testmuster hat bei r = 21.3 mm 1-Pixel-Linien im
    # 16-Pixel-Raster, also den ungemuetlichsten Fall, den die Untergrenze
    # ueberhaupt zulaesst, und dort bleiben 1.5 % der Pixel im Spalt der
    # Nabe stecken. Was fehlt, sagt der Report als Warnung -- danach
    # entscheidet der Anwender ueber einen groesseren Radius.
    assert report["pattern_completeness"] > 0.98
    if report["pattern_pixels_missing"]:
        assert any("Musterpixel" in w for w in report["warnings"])


def test_degenerate_masks_are_reported():
    """Ein Muster ganz ohne Klinge (oder ganz aus Klinge) ergibt kein
    sinnvolles Werkzeug -- das darf nicht stillschweigend passieren."""
    cfg = CoexistenceConfig(voxel_mm=1.6)
    empty = np.zeros((40, 40), dtype=bool)
    _, _, report = build_gyroid_dual_cylinder(empty, 30.0, 40.0, cfg,
                                              build_meshes=False)
    assert any("keine Klingenlinien" in w for w in report["warnings"])

    full = np.ones((40, 40), dtype=bool)
    _, _, report = build_gyroid_dual_cylinder(full, 30.0, 40.0, cfg,
                                              build_meshes=False)
    assert any("vollstaendig Klinge" in w for w in report["warnings"])


def test_image_threshold_maps_dark_pixels_to_the_blade():
    img = np.array([[10, 200], [250, 5]], dtype=np.uint8)
    mask = image_to_blade_mask(img, 128)
    assert mask.tolist() == [[True, False], [False, True]]
