# 🚀 SETUP GUIDE - Anaconda Prompt

## Step-by-Step Installation

### Step 1: Open Anaconda Prompt
1. Press Windows key
2. Type "Anaconda Prompt"
3. Click to open

### Step 2: Navigate to Project
```bash
cd C:\Users\Sanjana\Desktop\gllm
```

### Step 3: Install PyTorch (CPU version)
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```
⏱️ This takes ~3-5 minutes

### Step 4: Install Other Dependencies
```bash
pip install transformer-lens transformers sentence-transformers streamlit plotly scikit-learn
```
⏱️ This takes ~2-3 minutes

### Step 5: Verify Installation
```bash
python -c "import torch; import transformer_lens; import streamlit; print('✅ All dependencies installed!')"
```

---

## Running the System

### 🎨 Option 1: Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```
Then open your browser to http://localhost:8501

### 📝 Option 2: Run Example
```bash
python example.py
```

### ⌨️ Option 3: Command Line
```bash
python main.py --prompt "What is the capital of France?"
```

---

## Quick Reference

### Try Different Prompts
```bash
python main.py --prompt "Who wrote Romeo and Juliet?"
python main.py --prompt "What is the speed of light?"
python main.py --prompt "What is the largest planet in our solar system?"
```

### Adjust Parameters
```bash
python main.py --prompt "Your question" --num-responses 5 --temperature 0.9 --max-length 50
```

### Save Entropy Plot
```bash
python main.py --prompt "Your question" --save-plot entropy_curve.png
```

---

## What Happens on First Run?

When you first generate text, the system will download:
- ✅ GPT-2 model (~500MB) - one time only
- ✅ Sentence transformer model (~80MB) - one time only

These are cached in `C:\Users\Sanjana\.cache\huggingface\`

Subsequent runs will be much faster!

---

## Troubleshooting

### If you get "No module named 'X'"
```bash
pip install X
```

### If Streamlit port is busy
```bash
streamlit run app.py --server.port 8502
```

### If you want GPU support (if you have NVIDIA GPU)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

---

## All Set! 🎉

You're ready to detect hallucinations. Start with:
```bash
streamlit run app.py
```
