# setup_env.ps1 — Sanal ortam bootstrap scripti
# Kullanım: .\setup_env.ps1
# İdempotent: Tekrar tekrar çalıştırılabilir, var olan ortamı bozmaz.

$ErrorActionPreference = "Stop"
$venvPath = Join-Path $PSScriptRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$pipExe = Join-Path $venvPath "Scripts\pip.exe"
$reqFile = Join-Path $PSScriptRoot "requirements.txt"

Write-Host ""
Write-Host "=== Balik Projesi - Ortam Kurulumu ===" -ForegroundColor Cyan
Write-Host ""

# 1. Sanal ortam kontrolü ve oluşturma
if (Test-Path $pythonExe) {
    Write-Host "[OK] .venv zaten mevcut: $pythonExe" -ForegroundColor Green
} else {
    Write-Host "[...] .venv olusturuluyor..." -ForegroundColor Yellow
    python -m venv $venvPath
    if (-not (Test-Path $pythonExe)) {
        Write-Host "[HATA] .venv olusturulamadi!" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] .venv olusturuldu." -ForegroundColor Green
}

# 2. pip güncelleme
Write-Host "[...] pip guncelleniyor..." -ForegroundColor Yellow
& $pythonExe -m pip install --upgrade pip --quiet 2>$null
Write-Host "[OK] pip guncellendi." -ForegroundColor Green

# 3. Bağımlılık yükleme
if (Test-Path $reqFile) {
    Write-Host "[...] Bagimliliklar yukleniyor (requirements.txt)..." -ForegroundColor Yellow
    & $pipExe install -r $reqFile --quiet
    Write-Host "[OK] Bagimliliklar yuklendi." -ForegroundColor Green
} else {
    Write-Host "[!] requirements.txt bulunamadi, bagimlilik yuklemesi atlandi." -ForegroundColor Yellow
}

# 4. Doğrulama
Write-Host ""
Write-Host "=== Dogrulama ===" -ForegroundColor Cyan
$pyVersion = & $pythonExe --version 2>&1
Write-Host "Python: $pyVersion" -ForegroundColor White
Write-Host "Yol:    $pythonExe" -ForegroundColor White
Write-Host ""
Write-Host "Kurulum tamamlandi! VS Code'u yeniden baslatin." -ForegroundColor Green
Write-Host ""
