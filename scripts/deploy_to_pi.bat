@echo off
echo === Pi Transfer ===

set USER=fugaeus
set /p IP="Pi IP: "

if "%IP%"=="" exit /b

echo Klasorler olusturuluyor...
ssh %USER%@%IP% "mkdir -p ~/Balik_Projesi/{app,models}"

echo Dosyalar gonderiliyor...
scp -r ..\app\ %USER%@%IP%:~/Balik_Projesi/
scp -r ..\models\ %USER%@%IP%:~/Balik_Projesi/
scp install_pi.sh %USER%@%IP%:~/Balik_Projesi/

echo === Tamamlandi ===
echo Simdi: ./install_pi.sh && python3 app/main.py

ssh %USER%@%IP% "cd Balik_Projesi && bash"
