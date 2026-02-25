@echo off
echo Antigravity (Python/AI) Icin Gerekli Eklentiler Kuruluyor...

code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.black-formatter
code --install-extension ms-azuretools.vscode-docker
code --install-extension mikestead.dotenv
code --install-extension formulahendry.code-runner

echo Kurulum Tamamlandi! Lutfen VS Code'u yeniden baslatin.
pause
