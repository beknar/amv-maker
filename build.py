"""Build AMV Maker into a single executable using PyInstaller.

Usage:
    python build.py
"""

import subprocess
import sys


def main():
    print("Building AMV Maker executable...")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "amv_maker.spec", "--noconfirm"],
        cwd=".",
    )
    if result.returncode == 0:
        print("\nBuild complete! Executable is at: dist/AMV Maker.exe")
    else:
        print("\nBuild failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
