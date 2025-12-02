# Makefile for NumGeom-Fair ICML paper

PAPER = paper
FIGURES_DIR = figures

.PHONY: all clean figures

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib $(wildcard $(FIGURES_DIR)/*.png)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf

figures:
	@echo "Generating figures from experimental data..."
	cd ../../.. && python3.11 scripts/generate_plots.py

watch:
	@echo "Watching for changes..."
	@while true; do \
		inotifywait -e modify $(PAPER).tex references.bib 2>/dev/null || sleep 2; \
		make all; \
	done

.PHONY: all clean figures watch
