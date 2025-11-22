@ECHO OFF
ECHO Ortam temizlenip yeniden kuruluyor...

ECHO.
ECHO Mevcut .venv klasoru kontrol ediliyor...
IF EXIST .venv (
    ECHO ".venv" klasoru bulundu, kaldiriliyor...
    RMDIR /S /Q .venv
    IF EXIST .venv (
        ECHO.
        ECHO HATA: ".venv" klasoru kaldirilamadi. Klasorun bir program tarafindan kullanilmadigindan emin olun ve tekrar deneyin.
        ECHO Gerekirse bu komut istemini Administrator olarak calistirmayi deneyin.
        EXIT /B 1
    )
    ECHO ".venv" klasoru basariyla kaldirildi.
) ELSE (
    ECHO Mevcut bir .venv klasoru bulunamadi.
)

ECHO.
ECHO Yeni sanal ortam olusturuluyor...
python -m venv .venv || (
    ECHO HATA: Sanal ortam olusturulamadi.
    EXIT /B 1
)

ECHO.
ECHO Pip guncelleniyor...
.venv\Scripts\python.exe -m pip install --upgrade pip || (
    ECHO HATA: Pip guncellenemedi.
    EXIT /B 1
)

ECHO.
ECHO Gerekli kutuphaneler yukleniyor...
.venv\Scripts\pip.exe install -r requirements.txt || (
    ECHO HATA: Gerekli kutuphaneler yuklenemedi.
    EXIT /B 1
)

ECHO.
ECHO KURULUM BASARIYLA TAMAMLANDI! Simdi projeye baslayabilirsiniz.
