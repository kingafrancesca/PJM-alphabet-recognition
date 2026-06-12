' Launcher PJM - uruchamia rozpoznawanie alfabetu PJM bez okna terminala.
' Klikniecie tego pliku (lub skrotu na pulpicie) od razu otwiera podglad z kamery.
' Sciezki sa wzgledne wzgledem polozenia tego pliku, wiec mozna przeniesc caly folder.

Set fso = CreateObject("Scripting.FileSystemObject")
base = fso.GetParentFolderName(WScript.ScriptFullName)

' Uzywamy python.exe (nie pythonw.exe), bo camera.py loguje przez print() -
' pod pythonw nie ma stdout i skrypt sie wysypuje. Konsole chowamy stylem okna 0,
' a okno podgladu z kamery (OpenCV) i tak sie pokazuje normalnie.
python = base & "\.venv\Scripts\python.exe"
skrypt  = base & "\camera.py"

Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = base

If Not fso.FileExists(python) Then
    MsgBox "Nie znaleziono srodowiska Python (.venv)." & vbCrLf & _
           "Oczekiwano: " & python, vbCritical, "PJM - blad uruchomienia"
ElseIf Not fso.FileExists(skrypt) Then
    MsgBox "Nie znaleziono pliku camera.py w folderze projektu.", vbCritical, "PJM - blad uruchomienia"
Else
    ' 0 = ukryte okno konsoli, False = nie czekaj na zakonczenie
    sh.Run """" & python & """ """ & skrypt & """", 0, False
End If
