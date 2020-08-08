# Fontmaker
Makes your own font so you can create your own personal handwritten (latex) letters.

## Usage Instructions
0. Print the pdf page with the qr codes from `/template_creating/main.pdf`
1. Scan the pages (or take pics of them with your phone).
2. Run the latest `/template_reading/readXX.py`.
3. That should be it, now you have your font in `/showing_your_font/`. (TODO: call the python script that makes the font once all symbols are found and compile the latex example to show the font in a pdf.)

## Advanced Usage Instructions
0. Fill the `/template_creating/symbols.txt` with the symbols that you want in your font.
1. Compile the main.tex (you can just import this github to overleaf and press "compile"), or install Texmaker, open `main.tex` and press F6 F7.
2. Print the handwriting templated called `main.pdf` file with the qr codes.
3. Fill in the template and scan it.
4. Put the scanned images or scanned pdf into the `template_reading/in/` folder.
5. Create a python 3.6 environment in anaconda (see `readXX.py`) and run the latest `readXX.py` file.
6. TODO: Scan the extracted font symbols and actually create the font.