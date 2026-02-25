@echo off
echo === GitHub Senkronizasyonu Basliyor ===
git add .
git commit -m "Pi 5 Native picamera2 Mimari Guncellemesi"
git push origin HEAD
echo === Senkronizasyon Tamamlandi ===
pause
