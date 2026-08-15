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

import numpy as np
import pytest
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

APP_TIMEOUT = 60


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


def _run_with_uploaded_image(at: AppTest) -> AppTest:
    at.file_uploader(key="uploaded_file").set_value(
        ("test_pattern.png", _synthetic_test_image_bytes(), "image/png")
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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
