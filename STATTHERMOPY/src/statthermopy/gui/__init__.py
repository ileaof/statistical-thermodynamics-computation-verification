"""Optional Qt GUI for StatThermoPy (requires PySide6).

This subpackage is **not** imported by ``import statthermopy``; install the optional dependency
with ``pip install statthermopy[gui]`` and launch via ``statthermopy-gui`` or
``python -m statthermopy.gui.app``. The GUI adds no physics — it wraps the pure
statistical-mechanics core.
"""

from .app import main
from .mainwindow import StatThermoPyWindow

__all__ = ["main", "StatThermoPyWindow"]
