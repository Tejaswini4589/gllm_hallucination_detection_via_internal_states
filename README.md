# 🧠 LLM Hallucination Detection using Internal States & Hybrid Analysis

## 📌 Overview
Large Language Models (LLMs) often generate fluent but incorrect responses, known as **hallucinations**.  
This project proposes a **hybrid hallucination detection system** that combines:

- Internal model signals (hidden states, embeddings)
- Statistical analysis using EigenScore
- Feature clipping for overconfidence handling
- External verification using datasets

The goal is to improve **trust, reliability, and interpretability** of LLM outputs.

---

## 🚀 Key Features

- 🔁 Multi-response generation for consistency analysis  
- 📊 EigenScore computation for semantic divergence detection  
- 🧠 Internal state analysis using TransformerLens  
- ✂️ Feature clipping to reduce overconfident hallucinations  
- 📉 Stability & grounding score calculation  
- 🌐 External verification using datasets  
- 📈 Visualization of:
  - Eigenvalue spectrum
  - Confidence scores
  - ROC curves
- 🎯 Final hallucination risk scoring system  

---

## 🏗️ System Architecture

User Input
↓
LLM (Generate K Responses)
↓
Embedding Extraction (Hidden States)
↓
Feature Clipping
↓
EigenScore Computation
↓
External Verification (Dataset Similarity)
↓
Risk Score Calculation
↓
Visualization + Output


---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:**  
  - PyTorch  
  - TransformerLens  
  - NumPy  
  - Matplotlib / Plotly  
- **Models:** GPT-2 / Pythia  
- **Dataset:** QA / Wikipedia-based datasets  

---

## 🧪 Methodology

### 1. Multi-Response Generation
Generate multiple responses for a given query to analyze consistency.

### 2. Internal State Analysis
Extract hidden states using TransformerLens.

### 3. Feature Clipping
Reduce extreme activations to handle overconfident hallucinations.

### 4. EigenScore Computation
- Compute covariance matrix of embeddings  
- Analyze eigenvalues  
- Detect semantic divergence  

### 5. External Verification
Compare responses with dataset (ground truth / similarity).

### 6. Final Risk Score
Combine:
- Internal score (EigenScore)
- Stability score
- Grounding score
- External verification

---

## 📊 Output

- Generated responses  
- Confidence score  
- Hallucination classification  
- Eigenvalue spectrum visualization  
- Final hallucination risk score  

---

## 🧩 Use Cases

- 📚 Educational tools  
- 🏥 Healthcare AI systems  
- 🤖 Chatbots & virtual assistants  
- 🔬 AI research & evaluation  

---

## ⚠️ Challenges

- Handling log(0) in eigen computations  
- Variability in LLM outputs  
- Detecting overconfident hallucinations  
- Balancing internal & external signals  

---

## 📈 Future Improvements

- Support for larger LLMs (LLaMA, Mistral)  
- Real-time deployment  
- Improved verification mechanisms  
- Performance optimization  

---

## 👩‍💻 Team

- Kanduri Abhignya  
- Budiga Sanjana  
- Annam Ritika  
- Karnakanti Pooja  
- Manuka Tejaswini  

---

## 🎓 Learning Outcomes

- Understanding LLM internal states  
- Eigenvalue-based analysis  
- Feature clipping techniques  
- TransformerLens usage  
- Building hybrid AI evaluation systems  

---

## 📌 Status

🚧 Prototype Stage  
✔ Core system implemented  
⏳ Optimization & UI improvements in progress  

---

## 📜 License
This project is for academic and research purposes.

---

## ⭐ Acknowledgements
- TransformerLens  
- PyTorch  
- Open-source QA datasets  
