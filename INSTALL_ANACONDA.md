# Installation Guide for Anaconda Users

## Prerequisites
- Anaconda or Miniconda installed
- Python 3.8 or higher

## Installation Steps

### Step 1: Create a New Conda Environment (Recommended)

```bash
conda create -n hallucination python=3.10
conda activate hallucination
```

### Step 2: Install PyTorch

Install PyTorch based on your system:

**For CPU only:**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**For CUDA (GPU) - if you have NVIDIA GPU:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 3: Install Other Dependencies

```bash
pip install transformer-lens transformers sentence-transformers streamlit plotly scikit-learn matplotlib numpy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import transformer_lens; import streamlit; print('All dependencies installed successfully!')"
```

## Quick Start

### Activate Environment
```bash
conda activate hallucination
```

### Run Streamlit UI
```bash
streamlit run app.py
```

### Run Example
```bash
python example.py
```

### Run CLI
```bash
python main.py --prompt "What is the capital of France?"
```

## Troubleshooting

### ImportError: No module named 'transformer_lens'
```bash
pip install transformer-lens
```

### CUDA Out of Memory
Use CPU-only version or reduce batch size:
```bash
python main.py --prompt "..." --num-responses 3 --max-length 30
```

### Slow Performance on First Run
- First run downloads models (~500MB)
- Subsequent runs will be faster
- Models are cached in `~/.cache/huggingface/`

## Deactivate Environment

When done:
```bash
conda deactivate
```

## Uninstall

To remove the environment:
```bash
conda deactivate
conda env remove -n hallucination
```
