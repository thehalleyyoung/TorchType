.PHONY: all test experiments plots paper clean help

# Default target
all: test experiments plots paper

# Run comprehensive tests
test:
	@echo "Running NumGeom-AD tests..."
	python3.11 tests/test_comprehensive.py

# Run all experiments
experiments: exp1 exp2 exp3

exp1:
	@echo "Running Experiment 1: Precision-guided training..."
	python3.11 experiments/exp1_precision_guided.py

exp2:
	@echo "Running Experiment 2: Instability detection..."
	python3.11 experiments/exp2_instability_detection.py

exp3:
	@echo "Running Experiment 3: Transformer attention analysis..."
	python3.11 experiments/exp3_transformer_attention.py

# Generate plots for paper
plots:
	@echo "Generating plots..."
	python3.11 scripts/generate_plots.py

# Compile paper
paper:
	@echo "Compiling paper..."
	cd docs && pdflatex -interaction=nonstopmode numgeom_ad_paper_simple.tex > /dev/null 2>&1 || true
	cd docs && bibtex numgeom_ad_paper_simple > /dev/null 2>&1 || true
	cd docs && pdflatex -interaction=nonstopmode numgeom_ad_paper_simple.tex > /dev/null 2>&1 || true
	cd docs && pdflatex -interaction=nonstopmode numgeom_ad_paper_simple.tex > /dev/null 2>&1 || true
	@echo "Paper compiled: docs/numgeom_ad_paper_simple.pdf"

# Clean generated files
clean:
	rm -f data/*.json data/*.pdf data/*.png
	rm -f docs/figures/*.pdf docs/figures/*.png
	rm -f docs/*.aux docs/*.log docs/*.bbl docs/*.blg docs/*.out
	@echo "Cleaned generated files"

# Show help
help:
	@echo "NumGeom-AD Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Run tests, experiments, generate plots, and compile paper"
	@echo "  test        - Run comprehensive test suite"
	@echo "  experiments - Run all experiments"
	@echo "  exp1        - Run precision-guided training experiment"
	@echo "  exp2        - Run instability detection experiment"
	@echo "  exp3        - Run transformer attention analysis"
	@echo "  plots       - Generate publication-quality plots"
	@echo "  paper       - Compile LaTeX paper to PDF"
	@echo "  clean       - Remove generated files"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Quick start:"
	@echo "  make test   # Verify implementation"
	@echo "  make all    # Full reproduction"
