# Quick Start Guide

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **First run will download models (~500MB):**
   - GPT-2 model (~500MB)
   - Sentence transformer model (~80MB)

## Running the System

### Option 1: Streamlit Web UI (Recommended)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Option 2: Run Example Script

```bash
python example.py
```

This will analyze the prompt "What is the capital of France?" and show all metrics.

### Option 3: Command Line

```bash
python main.py --prompt "What is the capital of France?"
```

## Testing Different Prompts

Try these prompts from the ground truth dataset:

```bash
python main.py --prompt "Who wrote Romeo and Juliet?"
python main.py --prompt "What is the speed of light?"
python main.py --prompt "When did World War II end?"
python main.py --prompt "What is the largest planet in our solar system?"
```

## Understanding the Output

### Internal Metrics
- **Mean Entropy**: Average uncertainty (0-10, higher = more uncertain)
- **Stability Score**: Layer consistency (0-1, higher = more stable)
- **Grounding Score**: Attention to prompt (0-1, higher = more grounded)
- **Internal Risk**: Combined internal score (0-1, higher = more risk)

### External Metrics
- **Similarity Scores**: How similar each response is to ground truth (0-1)
- **External Consistency**: Average similarity across responses (0-1)
- **External Risk**: 1 - consistency (0-1, higher = more risk)

### Final Risk Score
- **< 0.3**: ✅ LOW - Reliable response
- **0.3-0.6**: ⚠️ MEDIUM - Some uncertainties
- **> 0.6**: ❌ HIGH - Likely hallucination

## Customization

### Add Your Own Questions

Edit `ground_truth.json`:

```json
[
    {
        "question": "Your question",
        "answer": "The correct answer"
    }
]
```

### Adjust Parameters

In Streamlit UI: Use sidebar sliders

In CLI: Use command-line flags
```bash
python main.py --prompt "..." --temperature 0.9 --num-responses 10
```

## Troubleshooting

### Out of Memory
- Use smaller model: `--model gpt2` (default)
- Reduce responses: `--num-responses 3`
- Reduce length: `--max-length 30`

### Slow Performance
- First run downloads models (one-time)
- GPU recommended but not required
- Reduce `--num-responses` for faster analysis

### No Ground Truth Found
- System will still compute internal metrics
- External risk defaults to 0.5 (neutral)
- Add your prompt to `ground_truth.json`

## Next Steps

1. Try the Streamlit UI for interactive exploration
2. Add your own questions to the ground truth dataset
3. Experiment with different temperature settings
4. Compare results across different GPT-2 model sizes

Enjoy exploring hallucination detection! 🔍
