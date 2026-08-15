"""
Shortcut Creator Utility
========================
Creates Windows shortcuts (.lnk) and batch launchers with a custom icon
on the Desktop and in the project folder.
"""

import os
import sys
import subprocess
from PIL import Image, ImageDraw

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHONW_PATH = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
GUI_APP_PATH = os.path.join(SCRIPT_DIR, "gui_app.py")
ICON_PATH = os.path.join(SCRIPT_DIR, "app_icon.ico")
BAT_PATH = os.path.join(SCRIPT_DIR, "Launch_Classifier.bat")

# 1. Generate Custom Industrial App Icon (.ico)
img = Image.new('RGBA', (256, 256), color=(0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Outer rounded rectangle (Dark industrial theme)
draw.rounded_rectangle([10, 10, 246, 246], radius=40, fill=(23, 31, 42, 255), outline=(0, 210, 211, 255), width=6)

# Camera / QC Inspector box
draw.rounded_rectangle([50, 80, 206, 180], radius=20, fill=(36, 51, 70, 255), outline=(52, 152, 219, 255), width=4)

# Camera lens outer & inner
draw.ellipse([98, 98, 158, 158], fill=(24, 33, 45, 255), outline=(0, 210, 211, 255), width=5)
draw.ellipse([112, 112, 144, 144], fill=(39, 174, 96, 255))

# Top green indicator light
draw.ellipse([180, 60, 210, 90], fill=(46, 204, 113, 255), outline=(255, 255, 255, 255), width=3)

# Save .ico file
img.save(ICON_PATH, format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print(f"[+] Custom icon saved: {ICON_PATH}")

# 2. Create Double-Clickable Batch File (.bat)
bat_content = f'''@echo off
start "" "{PYTHONW_PATH}" "{GUI_APP_PATH}"
'''
with open(BAT_PATH, "w") as f:
    f.write(bat_content)
print(f"[+] Batch launcher created: {BAT_PATH}")

# 3. Create Windows Shortcuts (.lnk) on Desktop using PowerShell
ps_script_path = os.path.join(SCRIPT_DIR, "make_shortcuts.ps1")
ps_code = f'''
$pythonw = "{PYTHONW_PATH}"
$script = "{GUI_APP_PATH}"
$workdir = "{SCRIPT_DIR}"
$icon = "{ICON_PATH}"

$wsh = New-Object -ComObject WScript.Shell

$destinations = @(
    [Environment]::GetFolderPath('Desktop'),
    "C:\\Users\\abrah\\Desktop",
    "C:\\Users\\abrah\\OneDrive\\Desktop",
    "{SCRIPT_DIR}"
)

foreach ($dest in $destinations) {{
    if (Test-Path $dest) {{
        $lnk = Join-Path $dest "Image Classifier HMI.lnk"
        $s = $wsh.CreateShortcut($lnk)
        $s.TargetPath = $pythonw
        $s.Arguments = "`"$script`""
        $s.WorkingDirectory = $workdir
        $s.IconLocation = $icon
        $s.Description = "Launch Industrial Image Classifier HMI"
        $s.Save()
        Write-Host "[+] Shortcut created at: $lnk"
    }}
}}
'''

with open(ps_script_path, "w") as f:
    f.write(ps_code)

res = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_script_path], capture_output=True, text=True)
print(res.stdout)
