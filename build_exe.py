"""
Build Script for Industrial Quality Control HMI Installer
==========================================================
Creates a distributable folder with the application, all dependencies,
model, and dataset bundled into a standalone Windows application.

Usage:
    python build_exe.py

Output:
    dist/IndustrialQualityHMI/   -> Folder with the .exe and all dependencies
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(SCRIPT_DIR, "dist")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
SPEC_FILE = os.path.join(SCRIPT_DIR, "build_installer.spec")
APP_NAME = "IndustrialQualityHMI"


def clean_previous_builds():
    """Remove previous build artifacts."""
    print("[1/4] Cleaning previous builds...")
    for d in [BUILD_DIR, DIST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"       Removed: {d}")
    print("       Clean complete.")


def run_pyinstaller():
    """Run PyInstaller with the spec file."""
    print("[2/4] Running PyInstaller (this may take several minutes)...")
    print(f"       Spec file: {SPEC_FILE}")
    print(f"       Python:    {sys.executable}")
    print()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        SPEC_FILE,
        "--noconfirm",
        "--clean",
        "--distpath", DIST_DIR,
        "--workpath", BUILD_DIR,
    ]

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)

    if result.returncode != 0:
        print("\n[FAIL] PyInstaller failed. See errors above.")
        sys.exit(1)

    print("\n       PyInstaller completed successfully.")


def create_launcher_bat():
    """Create a simple launcher .bat in the dist folder."""
    print("[3/4] Creating launcher script...")
    app_dir = os.path.join(DIST_DIR, APP_NAME)
    bat_path = os.path.join(app_dir, f"Launch_{APP_NAME}.bat")

    with open(bat_path, "w") as f:
        f.write(f'@echo off\n')
        f.write(f'title Industrial Quality Control HMI\n')
        f.write(f'echo Starting Industrial Quality Control HMI...\n')
        f.write(f'start "" "%~dp0{APP_NAME}.exe"\n')

    print(f"       Created: {bat_path}")


def print_summary():
    """Print build summary with folder sizes."""
    print("[4/4] Build Summary")
    print("=" * 60)

    app_dir = os.path.join(DIST_DIR, APP_NAME)
    exe_path = os.path.join(app_dir, f"{APP_NAME}.exe")

    if os.path.isfile(exe_path):
        exe_size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"  EXE:           {exe_path}")
        print(f"  EXE Size:      {exe_size_mb:.1f} MB")
    else:
        print(f"  [WARN] EXE not found at: {exe_path}")

    # Calculate total folder size
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(app_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
            file_count += 1

    total_mb = total_size / (1024 * 1024)
    print(f"  Total Files:   {file_count}")
    print(f"  Total Size:    {total_mb:.1f} MB")
    print(f"  Output Folder: {app_dir}")
    print()
    print("  To distribute: Copy or ZIP the entire folder:")
    print(f"    {app_dir}")
    print()
    print("  To run on target machine:")
    print(f"    Double-click: {APP_NAME}.exe")
    print("=" * 60)


def main():
    print()
    print("=" * 60)
    print("  Industrial Quality Control HMI - Build Tool")
    print("=" * 60)
    print()

    clean_previous_builds()
    run_pyinstaller()
    create_launcher_bat()
    print_summary()


if __name__ == "__main__":
    main()
