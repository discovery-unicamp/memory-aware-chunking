$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 2;
$clean_ext = 'synctex.gz synctex.gz(busy) run.xml tex.bak bbl bcf fdb_latexmk run tdo %R-blx.bib';

# Output directory
$out_dir = 'out';

# Automatically open PDF after compilation on macOS
$pdf_previewer = 'open -a Preview %S';
$preview_continuous_mode = 0;

# Run pdflatex the required number of times
$max_repeat = 5;

# Custom dependency for acronym package
add_cus_dep('acn', 'acr', 0, 'makeglossaries');
sub makeglossaries {
    system("makeglossaries $_[0]");
} 