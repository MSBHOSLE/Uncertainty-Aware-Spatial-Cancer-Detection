#!/usr/bin/env bash
# ==============================================================================
# Setup Environment for M_2_Uncertainty-Aware Spatial Cancer Detection
# Reliable Tumor Microenvironment Mapping: Spatial Cell Segmentation with OoD Detection
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== Initializing Project Environment ==="
echo "Project Directory: $PROJECT_DIR"

# 1. Check Python Interpreter
PYTHON_BIN=""
if command -v python3.12 &> /dev/null; then
    PYTHON_BIN="python3.12"
elif command -v /opt/homebrew/bin/python3.12 &> /dev/null; then
    PYTHON_BIN="/opt/homebrew/bin/python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    echo "ERROR: No suitable Python 3 found!"
    exit 1
fi
echo "Using Python binary: $PYTHON_BIN"

# 2. Setup Virtual Environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment at .venv..."
    "$PYTHON_BIN" -m venv .venv
fi

# 3. Configure site-packages linking if central environment exists
CENTRAL_SITE_PACKAGES="/Users/mahesh_bhosle/Desktop/MB_Projects/2_AI That Knows When a Medical Image Cannot Be Trusted/.venv/lib/python3.12/site-packages"
VENV_SITE_PACKAGES=".venv/lib/python3.12/site-packages"

if [ -d "$CENTRAL_SITE_PACKAGES" ] && [ -d "$VENV_SITE_PACKAGES" ]; then
    echo "Linking central pre-installed packages via base.pth..."
    echo "$CENTRAL_SITE_PACKAGES" > "$VENV_SITE_PACKAGES/base.pth"
fi

# 4. Create directory structure
echo "Creating project directory structure..."
mkdir -p configs
mkdir -p data/pannuke/{train,val,test_id,test_ood_organs,test_ood_corrupted}
mkdir -p src/{models,uncertainty,spatial,ood,evaluation,human_in_the_loop,utils}
mkdir -p scripts
mkdir -p tests
mkdir -p reports/figures
mkdir -p experiments/checkpoints

# 5. Verify Core Imports
echo "Verifying Python packages..."
.venv/bin/python -c "
import torch, torchvision, numpy, scipy, sklearn, skimage, pandas, networkx, matplotlib, streamlit, pptx
print('PyTorch Version:', torch.__version__)
print('MPS Available:', torch.backends.mps.is_available())
print('NetworkX Version:', networkx.__version__)
print('All core dependencies successfully verified!')
"

echo "=== Environment Setup Completed Successfully! ==="
