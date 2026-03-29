#!/bin/bash
# phoneme_to_words_lm environment setup
# Target: Ubuntu 22.04 with NVIDIA GPU (CUDA 12.8)
#
# Prerequisites:
#   - miniconda3 installed at ~/miniconda3
#
# Usage:
#   ./env_setup.sh           # full install (default)
#   ./env_setup.sh --no-gpu  # skip causal-conv1d/flash-linear-attention

set -euo pipefail

ENV_NAME="phoneme_lm"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse args
INSTALL_GPU_EXTRAS=true
for arg in "$@"; do
    case $arg in
        --no-gpu) INSTALL_GPU_EXTRAS=false ;;
    esac
done

source ~/miniconda3/bin/activate

# Create fresh conda environment
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

# Disable user site-packages
conda env config vars set PYTHONNOUSERSITE=1 -n "${ENV_NAME}"
export PYTHONNOUSERSITE=1

# ── PyTorch (CUDA 12.8) ─────────────────────────────────────────────
pip install \
    torch==2.11.0+cu128 \
    torchvision==0.26.0+cu128 \
    torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Pinned pip dependencies ─────────────────────────────────────────
pip install -r "${REPO_DIR}/requirements.txt"

# ── Source-built packages ────────────────────────────────────────────

# KenLM (n-gram language model library)
pip install https://github.com/kpu/kenlm/archive/master.zip

# Flashlight-text (beam search decoder)
# The bundled copy at flashlight_text/ has the kTrieMaxLabel = 60 patch
# already applied (required for phoneme sequence homophones).
pip install "${REPO_DIR}/flashlight_text" --no-build-isolation

# ── Optional GPU acceleration (causal-conv1d, flash-linear-attention) ─
if [ "${INSTALL_GPU_EXTRAS}" = true ]; then
    echo "Installing GPU acceleration packages..."
    # causal-conv1d needs CUDA headers + compatible gcc to compile
    conda install -c nvidia cuda-toolkit=12.8 -y
    conda install -c conda-forge gxx_linux-64=13.3 gcc_linux-64=13.3 -y
    pip install flash-linear-attention==0.4.2
    pip install causal-conv1d==1.6.1 --no-build-isolation
else
    echo "Skipping GPU acceleration packages (--no-gpu)"
fi

# ── Install this package in editable mode ───────────────────────────
pip install -e "${REPO_DIR}"

echo ""
echo "Environment '${ENV_NAME}' ready."
echo "Activate with: conda activate ${ENV_NAME}"
