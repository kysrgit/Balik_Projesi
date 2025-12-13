@echo off
setlocal

echo ========================================================
echo      Pufferfish Project - Raspberry Pi Deployer
echo ========================================================

:: Configuration
set "USER=fugaeus"

:: 1. Input IP
echo.
set /p RPi_IP="Lutfen Raspberry Pi IP adresini girin (Orn: 192.168.1.35): "

if "%RPi_IP%"=="" (
    echo IP adresi girmediniz. Cikis yapiliyor.
    pause
    exit /b
)

echo.
echo Hedef: %USER%@%RPi_IP%
echo.

:: 2. Preparation
echo [1/3] Uzak klasor yapisi olusturuluyor...
ssh %USER%@%RPi_IP% "mkdir -p ~/Balik_Projesi/app/detections"

if %errorlevel% neq 0 (
    echo.
    echo [HATA] Baglanti saglanamadi!
    echo Lutfen IP adresini kontrol edin ve Pi'nin acik oldugundan emin olun.
    pause
    exit /b
)

:: 3. Transfer
echo.
echo [2/3] Dosyalar gonderiliyor...

:: Copy App folder (Recursively - includes model and scripts)
echo   - App klasoru gonderiliyor...
scp -r app/ %USER%@%RPi_IP%:~/Balik_Projesi/

:: Copy Installer
echo   - Installer gonderiliyor...
scp install_pi.sh %USER%@%RPi_IP%:~/Balik_Projesi/

echo.
echo ========================================================
echo      Transfer Tamamlandi!
echo ========================================================
echo.
echo Simdi Raspberry Pi'ye baglaniyoruz...
echo.
echo ========================================================
echo  GUVENLI BAGLANTI ICIN SSH KEY KULLANIN:
echo  --------------------------------------------------------
echo  1. Eger SSH key ayarlamadiyseniz:
echo     ssh-keygen -t ed25519
echo     ssh-copy-id %USER%@RPi_IP_ADRESINIZ
echo.
echo  2. Bu sayede sifre girmeden guvenli baglanti kurarsınız.
echo  3. Sifreyi asla script veya kod icinde saklamayin!
echo ========================================================
echo.
echo Baglandiktan sonra calistirmaniz gereken komutlar:
echo   cd Balik_Projesi
echo   chmod +x install_pi.sh
echo   ./install_pi.sh
echo   python3 app/main_headless.py
echo.

:: 4. Connect (using SSH key or prompting for password securely)
ssh %USER%@%RPi_IP% "cd Balik_Projesi && bash"
