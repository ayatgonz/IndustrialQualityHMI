
$pythonw = "C:\Users\abrah\.gemini\antigravity\scratch\image_classifier\classImage\Scripts\pythonw.exe"
$script = "C:\Users\abrah\OneDrive\Desktop\ImageClassifier\Classifier\gui_app.py"
$workdir = "C:\Users\abrah\OneDrive\Desktop\ImageClassifier\Classifier"
$icon = "C:\Users\abrah\OneDrive\Desktop\ImageClassifier\Classifier\app_icon.ico"

$wsh = New-Object -ComObject WScript.Shell

$destinations = @(
    [Environment]::GetFolderPath('Desktop'),
    "C:\Users\abrah\Desktop",
    "C:\Users\abrah\OneDrive\Desktop",
    "C:\Users\abrah\OneDrive\Desktop\ImageClassifier\Classifier"
)

foreach ($dest in $destinations) {
    if (Test-Path $dest) {
        $lnk = Join-Path $dest "Image Classifier HMI.lnk"
        $s = $wsh.CreateShortcut($lnk)
        $s.TargetPath = $pythonw
        $s.Arguments = "`"$script`""
        $s.WorkingDirectory = $workdir
        $s.IconLocation = $icon
        $s.Description = "Launch Industrial Image Classifier HMI"
        $s.Save()
        Write-Host "[+] Shortcut created at: $lnk"
    }
}
