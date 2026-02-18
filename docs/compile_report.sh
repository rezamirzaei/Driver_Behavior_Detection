#!/bin/bash
# Compile ABAX Technical Report to PDF
# Usage: ./compile_report.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "ABAX Technical Report Compilation"
echo "=============================================="

# Check for tectonic (preferred)
if command -v tectonic &> /dev/null; then
    echo "Using tectonic for compilation..."
    tectonic ABAX_Technical_Report.tex
    echo ""
    echo "✅ PDF generated successfully: ABAX_Technical_Report.pdf"
    exit 0
fi

# Check for pdflatex
if command -v pdflatex &> /dev/null; then
    echo "Using pdflatex for compilation..."
    pdflatex -interaction=nonstopmode ABAX_Technical_Report.tex
    pdflatex -interaction=nonstopmode ABAX_Technical_Report.tex  # Run twice for TOC

    # Clean up auxiliary files
    rm -f *.aux *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

    echo ""
    echo "✅ PDF generated successfully: ABAX_Technical_Report.pdf"
    exit 0
fi

# Check for latexmk
if command -v latexmk &> /dev/null; then
    echo "Using latexmk for compilation..."
    latexmk -pdf -interaction=nonstopmode ABAX_Technical_Report.tex
    latexmk -c  # Clean auxiliary files

    echo ""
    echo "✅ PDF generated successfully: ABAX_Technical_Report.pdf"
    exit 0
fi

# No LaTeX compiler found
echo ""
echo "❌ Error: No LaTeX compiler found!"
echo ""
echo "Please install one of the following:"
echo ""
echo "Option 1 - Tectonic (recommended, lightweight):"
echo "  brew install tectonic"
echo ""
echo "Option 2 - MacTeX (full LaTeX distribution):"
echo "  brew install --cask mactex"
echo ""
echo "Option 3 - BasicTeX (smaller):"
echo "  brew install --cask basictex"
echo "  Then: sudo tlmgr install collection-fontsrecommended"
echo ""
exit 1
