"""
test_app.py
============

Tests fuer app.py mit Streamlit's eigenem Testframework
(streamlit.testing.v1.AppTest). Simuliert Nutzerinteraktion (Bild-Upload,
Checkboxen, Button-Klick) ohne echten Browser und prueft, dass beide
Erzeugungspfade -- Einzel-Zylinder und der (nun voreingestellte) Zweiteiler
aus Schneide und Ausstoesser (gyroid_coexistence.py) -- fehlerfrei
durchlaufen.

Die Tests laufen bewusst ueber die APP-VOREINSTELLUNGEN und nicht ueber
handverlesene Parameter: Fehler dieses Generators haengen an den physischen
Massen (Radius, Hoehe, Hub, Voxelgroesse), nicht am Bildinhalt, und genau
die Standardkombination ist diejenige, die ein Benutzer als erstes trifft.
Ein Bild mit dem Seitenverhaeltnis 1:1 ergibt bei Radius 30 mm eine
Zylinderhoehe von 2*pi*30 = 188 mm -- der Fall, in dem frueher Fehler
auftraten, die keiner der Tests bemerkt hat.

Ausfuehren mit: pytest test_app.py -v
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

APP_TIMEOUT = 60
# Der Zweiteiler rechnet auf einem Voxelgitter mit Millionen Zellen; das
# dauert im Sekundenbereich statt im Millisekundenbereich.
EJECTOR_TIMEOUT = 900


def _set_coarse_voxels(at, voxel_mm: float = 1.6):
    """Voxelgitter vergroebern, damit ein Test in Sekunden statt Minuten
    laeuft. Die Voxelgroesse aendert die Physik nicht -- nur wie fein sie
    abgetastet wird."""
    at.slider(key="ejector_voxel_mm").set_value(voxel_mm)
    return at


def _square_image_bytes(size: int = 200) -> bytes:
    """Quadratisches Puzzle-Raster. Bei Radius 30 mm wird daraus ein
    Zylinder von 2*pi*30 = 188 mm Hoehe -- die Standardgeometrie der App."""
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    for y in range(0, size, 45):
        draw.rectangle([0, y, size, y + 4], fill=0)
    for x in range(0, size, 45):
        draw.rectangle([x, 0, x + 4, size], fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _assert_two_sound_bodies(at, context: str):
    """Alle harten Bedingungen an einem erzeugten Zweiteiler -- an einer
    Stelle, damit jeder Test sie vollstaendig prueft und nicht nur die,
    an die der Autor gerade gedacht hat."""
    assert not at.exception, f"{context}: unerwartete Exception: {at.exception}"
    blade = at.session_state["shell_mesh"]
    ejector = at.session_state["core_mesh"]
    report = at.session_state["ejector_report"]
    assert blade is not None and not blade.is_empty, context
    assert ejector is not None and not ejector.is_empty, context
    assert report is not None, context

    assert blade.is_watertight, f"{context}: Schneide ist nicht geschlossen"
    assert ejector.is_watertight, f"{context}: Ausstoesser ist nicht geschlossen"
    assert report["blade_bodies"] == 1, (
        f"{context}: Schneide besteht aus {report['blade_bodies']} Teilen")
    assert report["ejector_bodies"] == 1, (
        f"{context}: Ausstoesser besteht aus {report['ejector_bodies']} Teilen")
    assert report["blade_floating_voxels"] == 0, (
        f"{context}: {report['blade_floating_voxels']} Voxel der Schneide "
        f"haengen in der Luft")
    assert report["ejector_floating_voxels"] == 0, (
        f"{context}: {report['ejector_floating_voxels']} Voxel des "
        f"Ausstoessers haengen in der Luft")
    assert report["blade_on_build_plate"] and report["ejector_on_build_plate"], (
        f"{context}: nicht beide Koerper stehen auf der Druckplatte")
    assert report["xy_travel_ok"], (
        f"{context}: der Ausstoesser kann sich nicht ueberall um den vollen "
        f"Hub bewegen ({report['xy_travel_violations']} Voxel zu dicht)")
    assert report["print_clearance_ok"], f"{context}: Druckspiel verletzt"
    return blade, ejector, report

# Das tatsaechliche Bild aus dem Bugreport (nahtlos kachelbares
# Puzzleteil-Muster): "wenn ich dieses Bild ... verwende, zeigen die
# Erhoehungen im core nach aussen und der overlap wird nan". Als Fixture
# abgelegt, damit der genaue Ausloeser fuer kuenftige Aenderungen als
# Regressionstest greifbar bleibt statt nur ueber eine Annaeherung.
PUZZLE_PATTERN_BUGREPORT_PATH = (
    Path(__file__).parent / "test_assets" / "puzzle_pattern_bugreport.png"
)


def _synthetic_test_image_bytes() -> bytes:
    """Erzeugt ein kleines Testbild mit echten Formen (kein reines Rauschen
    oder Einheitsfarbe), damit beim Schwellwert-Schnitt ein nicht-triviales
    Lochmuster entsteht -- inkl. einer potenziellen Insel (Punkt) und einer
    Form, die bis zum Bildrand laeuft."""
    img = Image.new("L", (120, 160), color=255)  # weiss = Material
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 90, 60], fill=0)      # grosses Lochfeld
    draw.rectangle([45, 35, 60, 45], fill=255)    # Insel darin (Material-Klotz)
    draw.rectangle([30, 130, 45, 159], fill=0)    # Linie, die bis zum Rand laeuft
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _seamless_tileable_test_image_bytes() -> bytes:
    """Erzeugt ein Testbild, das (wie das Puzzleteil-Hintergrundbild aus dem
    urspruenglichen Bugreport) fuer nahtloses Kacheln gedacht ist: mehrere
    wellenfoermige Gitterlinien, die den Bildrand mehrfach mit Zacken
    durchqueren, statt sauber und einheitlich am Rand abzuschliessen. Zeile 0
    und die letzte Zeile sind dabei bewusst NICHT identisch (sie fuehren nur
    das Muster fort) -- genau der Fall, der den theta=2*pi==theta=0
    Rundungsfehler in map_image_to_vertices ausgeloest hat."""
    ny, nx = 160, 220
    img_array = np.full((ny, nx), 255, dtype=np.uint8)
    xs = np.arange(nx)
    for row_base in (0, ny // 3, 2 * ny // 3, ny - 1):
        wave = (18 * np.sin(xs / 14.0)).astype(int)
        for x in xs:
            y0 = (row_base + wave[x]) % ny
            img_array[max(0, y0 - 2):y0 + 2, x] = 15
    ys = np.arange(ny)
    for col_base in (0, nx // 3, 2 * nx // 3, nx - 1):
        wave = (18 * np.sin(ys / 14.0)).astype(int)
        for y in ys:
            x0 = min(max(col_base + wave[y], 0), nx - 1)
            img_array[y, max(0, x0 - 2):x0 + 2] = 15
    img = Image.fromarray(img_array, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _puzzle_pattern_bugreport_image_bytes(max_dim: int = 220) -> bytes:
    """Laedt das echte Bugreport-Bild und verkleinert es (Seitenverhaeltnis
    erhalten) auf max_dim px, damit der Test in vertretbarer Zeit laeuft --
    bei den App-Standardeinstellungen (DPI 150, kein Upscaling) wuerde die
    Originalaufloesung (780x549) sonst unveraendert durch die ganze
    Mesh-Erzeugung samt Delaunay-Kappen laufen und den Test stark
    verlangsamen. Das Fixture-PNG selbst bleibt in Originalaufloesung
    erhalten, siehe PUZZLE_PATTERN_BUGREPORT_PATH."""
    img = Image.open(PUZZLE_PATTERN_BUGREPORT_PATH).convert("L")
    scale = max_dim / max(img.size)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_uploaded_file(image_bytes: bytes):
    """Baut ein Streamlit-UploadedFile-Objekt aus rohen PNG-Bytes."""
    from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec

    rec = UploadedFileRec(
        file_id="test-image", name="test_pattern.png",
        type="image/png", data=image_bytes,
    )
    return UploadedFile(rec, None)


def _run_with_uploaded_image(at: AppTest, image_bytes: bytes | None = None) -> AppTest:
    data = image_bytes or _synthetic_test_image_bytes()
    # AppTest.file_uploader gibt es erst ab Streamlit 1.50; requirements.txt
    # pinnt <1.50 (pyvista-Kompatibilitaet). Fuer aeltere Versionen wird der
    # Upload deshalb direkt ueber den Widget-Key im Session-State gesetzt --
    # das ist derselbe Zustand, den das Widget nach einem echten Upload haette.
    uploader = getattr(at, "file_uploader", None)
    if uploader is not None:
        uploader(key="uploaded_file").set_value(
            ("test_pattern.png", data, "image/png")
        )
    else:
        # Der Widget-Wert wird bei jedem Rerun aus dem Widget-Baum neu
        # gesetzt (und waere dann wieder None), deshalb wird der Upload vor
        # jedem Lauf erneut injiziert.
        original_run = at.run

        def run_with_upload(*args, **kwargs):
            at.session_state["uploaded_file"] = _make_uploaded_file(data)
            return original_run(*args, **kwargs)

        at.run = run_with_upload
    at.run(timeout=APP_TIMEOUT)
    return at


def test_app_loads_without_upload():
    """Die App muss auch im Leerzustand (kein Bild hochgeladen) ohne
    Exception starten."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    assert not at.exception


def test_single_mesh_generation():
    """Klassischer Einzel-Zylinder-Pfad muss weiterhin funktionieren."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at)
    assert not at.exception

    at.checkbox(key="generate_ejector_system").set_value(False)
    at.run(timeout=APP_TIMEOUT)
    assert not at.exception

    at.button(key="generate_button").click()
    at.run(timeout=APP_TIMEOUT)

    assert not at.exception
    assert at.session_state["mesh"] is not None
    assert not at.session_state["mesh"].is_empty
    assert at.session_state["shell_mesh"] is None
    assert at.session_state["core_mesh"] is None


def test_ejector_system_generation():
    """Der Zweiteiler ist die Voreinstellung und muss ohne jedes Zutun
    durchlaufen: Bild hochladen, Knopf druecken, fertig."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at)
    assert not at.exception
    assert at.checkbox(key="generate_ejector_system").value is True, (
        "Der Zweiteiler soll die Voreinstellung sein"
    )

    _set_coarse_voxels(at)
    at.run(timeout=APP_TIMEOUT)
    at.button(key="generate_button").click()
    at.run(timeout=EJECTOR_TIMEOUT)

    _assert_two_sound_bodies(at, "Standardpfad")
    assert at.session_state["mesh"] is None


def test_ejector_parts_really_do_not_touch():
    """Die zentrale Bedingung, unabhaengig vom Report nachgerechnet: die
    beiden Meshes duerfen sich nicht durchdringen, und der Ausstoesser muss
    sich um den vollen Hub verschieben lassen, ohne die Schneide zu
    beruehren.

    Geprueft wird das hier nicht am Voxelfeld (das tut der Generator
    selbst), sondern an den EXPORTIERTEN Koerpern -- also an dem, was
    tatsaechlich im Slicer landet."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at)
    _set_coarse_voxels(at)
    at.run(timeout=APP_TIMEOUT)
    at.button(key="generate_button").click()
    at.run(timeout=EJECTOR_TIMEOUT)

    blade, ejector, report = _assert_two_sound_bodies(at, "Kollisionspruefung")

    overlap = blade.intersection(ejector, engine="manifold")
    assert overlap.is_empty or abs(overlap.volume) < 1e-6, (
        f"Schneide und Ausstoesser durchdringen sich um "
        f"{abs(overlap.volume):.3f} mm3"
    )

    # Verschiebung in acht Richtungen der XY-Ebene um den vollen Hub.
    travel = at.slider(key="ejector_travel").value
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        moved = ejector.copy()
        moved.apply_translation(
            [travel * np.cos(angle), travel * np.sin(angle), 0.0]
        )
        hit = blade.intersection(moved, engine="manifold")
        volume = 0.0 if hit.is_empty else abs(hit.volume)
        # Eine Voxelecke Ueberschneidung ist Diskretisierung, kein Klemmen;
        # gemessen wird gegen das Volumen der Koerper.
        assert volume < 0.002 * abs(ejector.volume), (
            f"Bei Verschiebung um {travel} mm in Richtung "
            f"{np.degrees(angle):.0f} Grad klemmt der Ausstoesser "
            f"({volume:.1f} mm3 Ueberschneidung)"
        )


def test_radius_slider_locks_out_cylinders_that_are_too_small():
    """Zu kleine Zylinder werden gar nicht erst angeboten: unterhalb der
    Untergrenze bleibt fuer die Gyroid-Zone weniger als eine Masche uebrig.
    Der Regler zieht seine Grenze aus denselben Einstellungen, mit denen
    spaeter gerechnet wird -- wer den Hub verkleinert, darf auch kleiner
    bauen."""
    from gyroid_coexistence import CoexistenceConfig

    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at)
    assert not at.exception

    expected = CoexistenceConfig().min_radius_mm()
    slider = at.slider(key="radius")
    assert slider.min == pytest.approx(expected, abs=0.5), (
        f"Untergrenze des Radius ist {slider.min}, erwartet ~{expected:.1f}"
    )
    assert slider.value >= slider.min

    # Kleinerer Hub -> kleinere Untergrenze.
    at.slider(key="ejector_travel").set_value(1.5)
    at.run(timeout=APP_TIMEOUT)
    assert not at.exception
    assert at.slider(key="radius").min < expected

    # Ohne Zweiteiler faellt die Beschraenkung ganz weg.
    at.checkbox(key="generate_ejector_system").set_value(False)
    at.run(timeout=APP_TIMEOUT)
    assert at.slider(key="radius").min == pytest.approx(10.0)


def test_ejector_system_requires_axis_hole():
    """Ohne Achsbohrung darf der Generate-Button fuer den Auswerfer-Pfad
    nicht aktiv sein (Validierung in der Sidebar)."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at)

    at.checkbox(key="create_axis_hole").set_value(False)
    at.checkbox(key="generate_ejector_system").set_value(True)
    at.run(timeout=APP_TIMEOUT)

    assert not at.exception
    assert at.button(key="generate_button").disabled is True


def test_single_mesh_watertight_for_seamless_tileable_image():
    """Regressionstest fuer den gemeldeten Bug: bei einem nahtlos
    kachelbaren Muster (Puzzleteil-artig) sorgte ein Rundungsfehler in
    map_image_to_vertices (theta=2*pi statt theta<2*pi fuer die letzte
    Bildzeile) fuer fast-deckungsgleiche Randpunkte mit unterschiedlichem
    Radius an der theta=0-Naht -- das Mesh war dort nicht wasserdicht."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at, _seamless_tileable_test_image_bytes())
    assert not at.exception

    at.checkbox(key="generate_ejector_system").set_value(False)
    at.run(timeout=APP_TIMEOUT)

    at.button(key="generate_button").click()
    at.run(timeout=APP_TIMEOUT)

    assert not at.exception, f"Unerwartete Exception: {at.exception}"
    mesh = at.session_state["mesh"]
    assert mesh is not None and not mesh.is_empty
    assert mesh.is_watertight, (
        "Mesh ist fuer ein nahtlos kachelbares Muster nicht wasserdicht "
        "(theta-Naht-Regression)"
    )


def test_ejector_system_watertight_for_seamless_tileable_image():
    """Regressionstest aus dem urspruenglichen Bugreport: bei einem nahtlos
    kachelbaren Muster (Puzzleteil-artig, mit Zacken ueber den Bildrand) kam
    frueher ein nicht geschlossenes Mesh heraus und damit ein NaN als
    Ueberlappungsvolumen statt einer echten Pruefung."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at, _seamless_tileable_test_image_bytes())
    assert not at.exception

    _set_coarse_voxels(at)
    at.run(timeout=APP_TIMEOUT)
    at.button(key="generate_button").click()
    at.run(timeout=EJECTOR_TIMEOUT)

    blade, ejector, report = _assert_two_sound_bodies(at, "Kachelmuster")
    overlap = blade.intersection(ejector, engine="manifold")
    volume = 0.0 if overlap.is_empty else abs(overlap.volume)
    assert not np.isnan(volume), (
        "Ueberlappungsvolumen ist NaN -- genau der urspruenglich gemeldete "
        "Fehlerfall"
    )
    assert volume < 1e-6, f"Koerper durchdringen sich um {volume} mm3"


def test_ejector_system_with_original_bugreport_image():
    """Das TATSAECHLICHE Bild aus dem Bugreport, komplett ueber die echte
    App-UI und mit den VOREINSTELLUNGEN der App erzeugt.

    Es ist der harte Fall: die Schnittlinien bilden lauter geschlossene
    Puzzlezellen, jede Ausstoesserplatte ist damit ringsum von Klinge
    umgeben. Ohne die Gyroid-Struktur in der Tiefe koennte keine dieser
    Platten mit den anderen verbunden werden, ohne das Muster zu
    zerschneiden."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at, _puzzle_pattern_bugreport_image_bytes())
    assert not at.exception

    _set_coarse_voxels(at)
    at.run(timeout=APP_TIMEOUT)
    at.button(key="generate_button").click()
    at.run(timeout=EJECTOR_TIMEOUT)

    _assert_two_sound_bodies(at, "Original-Bugreport-Bild")


def test_ejector_system_at_app_defaults_for_square_image():
    """Alles auf Voreinstellung, quadratisches Muster -- daraus wird bei
    Radius 30 mm ein Zylinder von 188 mm Hoehe.

    Diese Kombination ist mit Absicht getestet: Fehler dieses Generators
    haengen an den physischen Massen und am Verhaeltnis von Hoehe zu Radius,
    nicht am Bildinhalt. Beim schlanken, hohen Zylinder ist die
    Gyroid-Zone nur noch etwa eine Maschenweite dick, und genau dort sind
    frueher Fehler durchgerutscht, die mit handverlesenen Testparametern
    unsichtbar blieben (etwa dr > dz, wodurch die 45deg-Treppe als
    schwebend galt und das halbe Muster weggetrimmt wurde)."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at, _square_image_bytes())
    assert not at.exception

    # Bewusst KEINE Vergroeberung: das hier ist der Standardfall.
    at.button(key="generate_button").click()
    at.run(timeout=EJECTOR_TIMEOUT)

    blade, ejector, report = _assert_two_sound_bodies(at, "App-Voreinstellung")
    height = blade.bounds[1, 2] - blade.bounds[0, 2]
    assert height == pytest.approx(2 * np.pi * 30.0, rel=0.02), (
        f"Erwartet wurde die Standardgeometrie (Hoehe = Umfang = 188 mm), "
        f"gemessen {height:.0f} mm"
    )
    assert report["ok"], f"Report meldet Maengel: {report['warnings']}"


def _puzzle_grid_image_bytes(ny: int = 240, nx: int = 180) -> bytes:
    """Ein Puzzle-artiges Raster: die Schnittlinien bilden geschlossene
    Zellen, jedes Segment ist damit VOLLSTAENDIG EINGESCHLOSSEN. Ohne
    Verbindungsstege wuerde daraus eine Schale aus lauter losen Teilen."""
    img = Image.new("L", (nx, ny), color=255)
    draw = ImageDraw.Draw(img)
    for y in range(0, ny, 40):
        draw.rectangle([10, y, nx - 10, y + 3], fill=0)
    for x in range(10, nx - 9, 35):
        draw.rectangle([x, 0, x + 3, ny], fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_ejector_system_connects_enclosed_puzzle_segments():
    """Der Kern des Konzepts: bei einem Puzzle-Raster ist JEDE
    Ausstoesserflaeche vollstaendig von Klinge umschlossen. In der Bildebene
    gibt es keine Verbindung zwischen ihnen -- sie kann nur in der Tiefe
    entstehen, durch die Gyroid-Struktur.

    Der Test prueft deshalb beides: dass das Muster wirklich aus lauter
    eingeschlossenen Zellen besteht, und dass trotzdem genau ein
    Ausstoesser-Koerper herauskommt."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at, _puzzle_grid_image_bytes())
    assert not at.exception

    _set_coarse_voxels(at)
    at.run(timeout=APP_TIMEOUT)
    at.button(key="generate_button").click()
    at.run(timeout=EJECTOR_TIMEOUT)

    blade, ejector, report = _assert_two_sound_bodies(at, "Puzzle-Raster")

    # Gegenprobe, dass das Testmuster wirklich der schwere Fall ist: in der
    # BILDEBENE zerfaellt die Ausstoesserflaeche in viele Zellen.
    from gyroid_coexistence import image_to_blade_mask, label_periodic

    plate2d = ~image_to_blade_mask(
        np.array(Image.open(io.BytesIO(_puzzle_grid_image_bytes())).convert("L")),
        128,
    )
    _, cells = label_periodic(plate2d[:, :, None])
    assert cells > 4, (
        f"Testmuster sollte viele eingeschlossene Zellen haben, hat aber {cells}"
    )


def test_ejector_parts_are_durable_not_hollow():
    """Zweiter gemeldeter Fehler der Vorgaengerversion: 'beide Zylinder hohl
    -> geringe Haltbarkeit'. Nachgerechnet wird deshalb, dass beide Koerper
    einen nennenswerten Teil des Bauraums fuellen und dass ihr Material
    ueberwiegend dicker ist als die eingestellte Mindestwandstaerke."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at)
    _set_coarse_voxels(at)
    at.run(timeout=APP_TIMEOUT)
    at.button(key="generate_button").click()
    at.run(timeout=EJECTOR_TIMEOUT)

    blade, ejector, report = _assert_two_sound_bodies(at, "Haltbarkeit")

    height = blade.bounds[1, 2] - blade.bounds[0, 2]
    nominal = np.pi * 30.0 ** 2 * height
    together = report["blade_volume_mm3"] + report["ejector_volume_mm3"]
    assert together > 0.15 * nominal, (
        f"Beide Koerper zusammen fuellen nur {together / nominal * 100:.0f} % "
        f"des Zylinders -- das ist eher Gitter als Bauteil"
    )
    assert report["blade_volume_mm3"] > 0.02 * nominal, (
        "Die Schneide ist zu einer Haut zusammengeschrumpft"
    )
    assert report["ejector_volume_mm3"] > 0.05 * nominal, (
        "Der Ausstoesser ist zu duenn, um etwas herauszudruecken"
    )
    assert report["blade_wall_ratio"] > 0.2 and report["ejector_wall_ratio"] > 0.2, (
        f"Zu viel duennes Material: Wandanteile "
        f"{report['blade_wall_ratio']:.2f} / {report['ejector_wall_ratio']:.2f}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
