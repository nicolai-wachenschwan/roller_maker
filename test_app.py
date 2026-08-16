"""
test_app.py
============

Tests fuer app.py mit Streamlit's eigenem Testframework
(streamlit.testing.v1.AppTest). Simuliert Nutzerinteraktion (Bild-Upload,
Checkboxen, Button-Klick) ohne echten Browser und prueft, dass beide
Erzeugungspfade -- Einzel-Zylinder und das neue Zwei-Teile-Auswerfer-System
aus dual_cylinder_ejector.py -- fehlerfrei durchlaufen.

Ausfuehren mit: pytest test_app.py -v
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

APP_TIMEOUT = 60

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


def _run_with_uploaded_image(at: AppTest, image_bytes: bytes | None = None) -> AppTest:
    at.file_uploader(key="uploaded_file").set_value(
        ("test_pattern.png", image_bytes or _synthetic_test_image_bytes(), "image/png")
    )
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
    """Der neue Zwei-Teile-Pfad muss Schale+Kern erzeugen, ohne Fehler, und
    das Overlap-freie Ergebnis (siehe dual_cylinder_ejector.py) muss auch
    ueber die UI ankommen."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at)
    assert not at.exception

    at.checkbox(key="create_axis_hole").set_value(True)
    at.checkbox(key="generate_ejector_system").set_value(True)
    at.run(timeout=APP_TIMEOUT)
    assert not at.exception

    at.button(key="generate_button").click()
    at.run(timeout=APP_TIMEOUT)

    assert not at.exception, f"Unerwartete Exception: {at.exception}"
    assert at.session_state["mesh"] is None

    shell_mesh = at.session_state["shell_mesh"]
    core_mesh = at.session_state["core_mesh"]
    assert shell_mesh is not None and not shell_mesh.is_empty
    assert core_mesh is not None and not core_mesh.is_empty

    report = at.session_state["ejector_report"]
    assert report is not None
    assert report["overlap_free"] is True, (
        f"Schale und Kern ueberlappen ueber die App-UI: "
        f"{report['overlap_volume_mm3']} mm3"
    )
    assert report["remaining_islands"] == 0
    assert report["remaining_severing_rings"] == []


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


def test_ejector_system_watertight_and_overlap_free_for_seamless_tileable_image():
    """Regressionstest fuer den gemeldeten Bug: bei einem nahtlos
    kachelbaren Muster (Puzzleteil-artig) fuehrte eine nicht robuste
    Endkappen-Triangulierung zu einer nicht-wasserdichten Schale/Kern und
    damit zu einem NaN-Ueberlappungsvolumen statt einer echten Pruefung."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at, _seamless_tileable_test_image_bytes())
    assert not at.exception

    at.checkbox(key="create_axis_hole").set_value(True)
    at.checkbox(key="generate_ejector_system").set_value(True)
    at.run(timeout=APP_TIMEOUT)

    at.button(key="generate_button").click()
    at.run(timeout=APP_TIMEOUT)

    assert not at.exception, f"Unerwartete Exception: {at.exception}"
    shell_mesh = at.session_state["shell_mesh"]
    core_mesh = at.session_state["core_mesh"]
    assert shell_mesh is not None and not shell_mesh.is_empty
    assert core_mesh is not None and not core_mesh.is_empty
    assert shell_mesh.is_watertight, "Schale ist beim Kachelmuster nicht wasserdicht"
    assert core_mesh.is_watertight, "Kern ist beim Kachelmuster nicht wasserdicht"

    report = at.session_state["ejector_report"]
    assert report is not None
    assert not np.isnan(report["overlap_volume_mm3"]), (
        "Ueberlappungsvolumen ist NaN -- genau der urspruenglich gemeldete Fehlerfall"
    )
    assert report["overlap_free"] is True, (
        f"Schale und Kern ueberlappen ueber die App-UI: "
        f"{report['overlap_volume_mm3']} mm3"
    )


def test_ejector_system_with_original_bugreport_image():
    """Der wichtigste Regressionstest: das TATSAECHLICHE Bild aus dem
    Bugreport (test_assets/puzzle_pattern_bugreport.png), einmal komplett
    ueber die echte App-UI hochgeladen und mit dem Zwei-Teile-Auswerfer
    erzeugt. Muss watertight sein und darf kein NaN-Ueberlappungsvolumen
    liefern."""
    at = AppTest.from_file("app.py")
    at.run(timeout=APP_TIMEOUT)
    at = _run_with_uploaded_image(at, _puzzle_pattern_bugreport_image_bytes())
    assert not at.exception

    at.checkbox(key="create_axis_hole").set_value(True)
    at.checkbox(key="generate_ejector_system").set_value(True)
    at.run(timeout=APP_TIMEOUT)

    at.button(key="generate_button").click()
    at.run(timeout=APP_TIMEOUT)

    assert not at.exception, f"Unerwartete Exception: {at.exception}"
    shell_mesh = at.session_state["shell_mesh"]
    core_mesh = at.session_state["core_mesh"]
    assert shell_mesh is not None and not shell_mesh.is_empty
    assert core_mesh is not None and not core_mesh.is_empty
    assert shell_mesh.is_watertight, "Schale ist beim Original-Bugreport-Bild nicht wasserdicht"
    assert core_mesh.is_watertight, "Kern ist beim Original-Bugreport-Bild nicht wasserdicht"

    report = at.session_state["ejector_report"]
    assert report is not None
    assert not np.isnan(report["overlap_volume_mm3"]), (
        "Ueberlappungsvolumen ist NaN beim Original-Bugreport-Bild -- der "
        "urspruenglich gemeldete Fehlerfall"
    )
    assert report["overlap_free"] is True, (
        f"Schale und Kern ueberlappen beim Original-Bugreport-Bild: "
        f"{report['overlap_volume_mm3']} mm3"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
