; Inno Setup Script for Industrial Quality Control HMI
; ========================================================

[Setup]
AppName=Industrial Quality Control HMI
AppVersion=1.0.0
AppPublisher= ayatgonz
DefaultDirName={userappdata}\IndustrialQualityHMI
DefaultGroupName=Industrial Quality Control HMI
UninstallDisplayIcon={app}\IndustrialQualityHMI.exe
Compression=lzma2/max
SolidCompression=yes
OutputDir=C:\Users\abrah\OneDrive\Desktop\ImageClassifier\Classifier\dist_installer
OutputBaseFilename=IndustrialQualityHMI_Setup
SetupIconFile=C:\Users\abrah\OneDrive\Desktop\ImageClassifier\Classifier\app_icon.ico
WizardStyle=modern
DisableProgramGroupPage=yes
DisableWelcomePage=no

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Copy all files from the collection folder
Source: "C:\Users\abrah\OneDrive\Desktop\ImageClassifier\Classifier\dist\IndustrialQualityHMI\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Industrial Quality Control HMI"; Filename: "{app}\IndustrialQualityHMI.exe"
Name: "{group}\Uninstall Industrial Quality Control HMI"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Industrial Quality Control HMI"; Filename: "{app}\IndustrialQualityHMI.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\IndustrialQualityHMI.exe"; Description: "{cm:LaunchProgram,Industrial Quality Control HMI}"; Flags: nowait postinstall skipifsilent
