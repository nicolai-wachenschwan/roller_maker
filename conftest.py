"""
Pytest-Fixture-Datei: stubbt das UI-Rendering-Paket 'stpyvista', bevor
app.py importiert wird.

Hintergrund: stpyvista 0.2.1 (siehe requirements.txt) ist mit aktuellen
Streamlit-Versionen inkompatibel (Streamlit hat mit der "components v2"-API
ein Breaking Change eingefuehrt, das stpyvista noch nicht unterstuetzt --
weder aeltere noch neuere Streamlit-Versionen funktionieren mit dieser
stpyvista-Version). Das ist ein vorbestehendes Dependency-Problem, unabhaengig
von den hier getesteten Aenderungen.

Da stpyvista ohnehin nur einen interaktiven iframe im Browser rendert (im
headless AppTest-Lauf gibt es davon nichts zu pruefen), wird das Modul fuer
die Tests durch einen No-Op-Stub ersetzt. Die eigentliche Geometrie-Logik
(dual_cylinder_ejector.py, create_cylinder_mesh, ...) wird davon nicht
beruehrt und lueckenlos getestet.
"""

import sys
import types


def _install_stpyvista_stub():
    if "stpyvista" in sys.modules:
        return

    def _stpyvista(plotter, key=None, **kwargs):
        return None

    stpyvista_mod = types.ModuleType("stpyvista")
    stpyvista_mod.stpyvista = _stpyvista

    utils_mod = types.ModuleType("stpyvista.utils")
    utils_mod.start_xvfb = lambda *a, **kw: None
    stpyvista_mod.utils = utils_mod

    sys.modules["stpyvista"] = stpyvista_mod
    sys.modules["stpyvista.utils"] = utils_mod


_install_stpyvista_stub()
