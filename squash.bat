@echo off

rem Run after "python setup.py py2exe"
rem You will need 7-zip and UPX in your PATH:
rem   - http://www.7-zip.org/
rem   - http://upx.sourceforge.net/

cd dist

rem Decompress library.zip
7z -aoa x library.zip -olibrary\
del library.zip

rem Recompress library.zip
cd library\
7z a -tzip -mx9 ..\library.zip -r
cd ..
rd library /s /q

rem UPX-compress executables
upx --best *.*

cd ..
