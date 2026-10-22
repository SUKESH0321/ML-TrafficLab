"""Build the browser (Pygbag/WebAssembly) version into public/ for Vercel.

Usage:  python build_web.py
Requires:  pip install pygbag
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "web_src" / "adaptive-traffic-lab"
OUTPUT = ROOT / "public"
GAME_FILES = (
    "main.py", "simulation.py", "traffic_signal.py", "vehicle.py",
    "q_learning_agent.py", "training.py", "evaluation.py",
)


def main() -> None:
    shutil.rmtree(STAGE.parent, ignore_errors=True)
    STAGE.mkdir(parents=True)
    for name in GAME_FILES:
        shutil.copy2(ROOT / name, STAGE / name)

    subprocess.run([sys.executable, "-X", "utf8", "-m", "pygbag", "--build", str(STAGE)], check=True)

    built = STAGE / "build" / "web"
    if not (built / "index.html").exists():
        raise SystemExit(f"Pygbag did not produce {built / 'index.html'}")
    shutil.rmtree(OUTPUT, ignore_errors=True)
    shutil.copytree(built, OUTPUT)
    shutil.rmtree(STAGE.parent, ignore_errors=True)
    print(f"Done. Static site written to {OUTPUT}")


if __name__ == "__main__":
    main()
