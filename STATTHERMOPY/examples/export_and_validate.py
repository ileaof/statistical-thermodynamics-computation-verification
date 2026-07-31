"""Export demonstration: compute CH4 at 800 K/5 bar and write CSV/JSON/YAML/LaTeX/Excel."""

from __future__ import annotations

from pathlib import Path

from statthermopy import State, Thermodynamics, get
from statthermopy.io import Exporter


def main() -> None:
    ch4 = get("CH4")
    res = Thermodynamics(ch4, State(T=800.0, P=5e5)).compute()
    out = Path(__file__).parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    for fmt, ext in [("csv", "csv"), ("json", "json"), ("yaml", "yaml"),
                     ("latex", "tex"), ("excel", "xlsx")]:
        path = out / f"ch4.{ext}"
        getattr(Exporter(res), f"to_{fmt}")(path)
        print(f"  wrote {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()