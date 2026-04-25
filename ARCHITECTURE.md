# System Architecture

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                   │
│                    "What is the capital of France?"                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL LOADER                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  TransformerLens GPT-2                                        │  │
│  │  • Load pretrained model                                      │  │
│  │  • Generate 5 stochastic responses (temp=0.8)                │  │
│  │  • Capture activations (logits, hidden states, attention)    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
┌───────────────────────────┐   ┌─────────────────────────────┐
│   INTERNAL METRICS        │   │   EXTERNAL VERIFIER         │
│                           │   │                             │
│  ┌─────────────────────┐ │   │  ┌───────────────────────┐ │
│  │ Entropy Metric      │ │   │  │ Ground Truth Loader   │ │
│  │ • Token-level       │ │   │  │ • Match question      │ │
│  │ • Mean & max        │ │   │  │ • Load answer         │ │
│  │ • Normalized        │ │   │  └───────────────────────┘ │
│  └─────────────────────┘ │   │                             │
│                           │   │  ┌───────────────────────┐ │
│  ┌─────────────────────┐ │   │  │ Sentence Transformer  │ │
│  │ Stability Metric    │ │   │  │ • Encode responses    │ │
│  │ • Layer similarity  │ │   │  │ • Encode ground truth │ │
│  │ • Cosine distance   │ │   │  │ • Compute similarity  │ │
│  │ • Averaged          │ │   │  └───────────────────────┘ │
│  └─────────────────────┘ │   │                             │
│                           │   │  ┌───────────────────────┐ │
│  ┌─────────────────────┐ │   │  │ Consistency Score     │ │
│  │ Grounding Metric    │ │   │  │ • Mean similarity     │ │
│  │ • Attention to      │ │   │  │ • External risk       │ │
│  │   prompt ratio      │ │   │  │   = 1 - consistency   │ │
│  │ • Across layers     │ │   │  └───────────────────────┘ │
│  └─────────────────────┘ │   │                             │
│                           │   └─────────────────────────────┘
│  ┌─────────────────────┐ │
│  │ Internal Risk       │ │
│  │ = w1*entropy +      │ │
│  │   w2*(1-stability)+ │ │
│  │   w3*(1-grounding)  │ │
│  └─────────────────────┘ │
└───────────┬───────────────┘
            │
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ANALYZER                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Hybrid Risk Score                                            │  │
│  │  = alpha * InternalRisk + beta * ExternalRisk                │  │
│  │  = 0.6 * InternalRisk + 0.4 * ExternalRisk                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • 5 Generated Responses                                       │  │
│  │ • Mean Entropy: 2.3456                                        │  │
│  │ • Max Entropy: 4.5678                                         │  │
│  │ • Stability Score: 0.8765                                     │  │
│  │ • Grounding Score: 0.7654                                     │  │
│  │ • Internal Risk: 0.3456                                       │  │
│  │ • Similarity Scores: [0.92, 0.89, 0.91, 0.88, 0.90]          │  │
│  │ • External Consistency: 0.9000                                │  │
│  │ • External Risk: 0.1000                                       │  │
│  │ • FINAL RISK: 0.2474 (LOW)                                   │  │
│  │ • Entropy Curve Visualization                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Interaction

```
┌──────────────┐
│   app.py     │  Streamlit UI
│   main.py    │  CLI Interface
│   example.py │  Demo Script
└──────┬───────┘
       │
       │ uses
       ▼
┌──────────────────┐
│   analyzer.py    │  Orchestrates everything
└──────┬───────────┘
       │
       │ coordinates
       ▼
┌──────────────────────────────────────────┐
│  model_loader.py                         │
│  internal_metrics.py                     │
│  external_verifier.py                    │
└──────────────────────────────────────────┘
       │
       │ uses
       ▼
┌──────────────────────────────────────────┐
│  TransformerLens (GPT-2)                 │
│  SentenceTransformer (MiniLM)            │
│  PyTorch, NumPy, Matplotlib              │
└──────────────────────────────────────────┘
```

## Risk Calculation Formula

```
InternalRisk = w1 × normalized_entropy
             + w2 × (1 - stability)
             + w3 × (1 - grounding)

where:
  w1 = 0.4 (entropy weight)
  w2 = 0.3 (stability weight)
  w3 = 0.3 (grounding weight)

ExternalRisk = 1 - ExternalConsistency
             = 1 - mean(similarities)

FinalRisk = α × InternalRisk + β × ExternalRisk

where:
  α = 0.6 (internal weight)
  β = 0.4 (external weight)

Risk Interpretation:
  FinalRisk < 0.3  → LOW RISK (reliable)
  0.3 ≤ FinalRisk < 0.6 → MEDIUM RISK (uncertain)
  FinalRisk ≥ 0.6  → HIGH RISK (hallucination)
```

## Module Dependencies

```
model_loader.py
  ├── transformer_lens (HookedTransformer)
  ├── torch
  └── numpy

internal_metrics.py
  ├── torch
  ├── torch.nn.functional
  └── numpy

external_verifier.py
  ├── sentence_transformers
  ├── sklearn.metrics.pairwise
  ├── json
  └── numpy

analyzer.py
  ├── model_loader
  ├── internal_metrics
  ├── external_verifier
  └── matplotlib

app.py
  ├── analyzer
  ├── streamlit
  ├── plotly
  └── matplotlib

main.py
  ├── analyzer
  └── argparse
```

## File Sizes

```
Core Modules:        ~25 KB
UI/CLI:             ~15 KB
Documentation:      ~15 KB
Data:               ~1.5 KB
Total Project:      ~56 KB

External Downloads (first run):
  GPT-2 model:      ~500 MB
  MiniLM model:     ~80 MB
  PyTorch:          ~200 MB
```
