SOURCE = ijcai23

$(SOURCE).pdf: $(SOURCE).tex $(SOURCE).bib
	pdflatex $(SOURCE).tex
	bibtex $(SOURCE)
	pdflatex $(SOURCE).tex
	pdflatex $(SOURCE).tex

clean:
	rm -rf *.{aux,bbl,blg,log,out}
