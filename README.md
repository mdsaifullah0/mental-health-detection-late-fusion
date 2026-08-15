# 🧠 Mental Health Detection using Late Fusion Transformer Models with LLM Chat Integration

[![IEEE Xplore](https://img.shields.io/badge/IEEE-Paper_Published-00629B?style=for-the-badge&logo=ieee&logoColor=white)](https://doi.org/10.1109/WIECON-ECE69386.2025.11526093)
[![Conference](https://img.shields.io/badge/WIECON--ECE-2025-blue?style=for-the-badge)](https://wiecon-ece.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

Official implementation of the research paper: **"Mental Health Detection using Late Fusion Transformer Models with LLM Chat Integration"**, presented at the **2025 IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering (WIECON-ECE)**.

---

## 📌 Overview
This study presents a multilingual, security-conscious text-based framework built on a **dual-encoder late-fusion architecture**. It combines **DistilBERT** and **XLM-RoBERTa** models using a learnable softmax-based weighting mechanism at the logit level. Additionally, an integrated **LLaMA-3** module provides empathetic, non-clinical supportive guidance and resources.

### 🎯 Key Highlights:
- **7 Classification Classes:** Normal, Depression, Anxiety, Bipolar, Stress, Suicide, and Personality Disorder.
- **Dual-Encoder Late Fusion:** Combines English-centric (DistilBERT) and Multilingual (XLM-RoBERTa) representations.
- **Optimized Training:** Fine-tuned final layers using AdamW optimizer and gradient stability controls.
- **Explainability & LLM Support:** Integrated with LLaMA-3 for automated empathetic conversational assistance.
- **Performance:** Achieved **76.68% Accuracy** and **76.54% Weighted F1-score**.

---

## 📦 Model Weights
Due to GitHub's file size limit, the trained model weight (`fusion_model.pt`) is hosted on Google Drive:
- 🔗 **[Download Trained Fusion Model (Google Drive)](https://drive.google.com/file/d/1T6tRRtCLVey8jiSN3HV81YbD-ymG1fAV/view?usp=sharing)**

> **Note:** After downloading, place the `fusion_model.pt` file in the root directory of this project before running the application.

---

## 🏗️ System Architecture

```text
User Text Input
       │
       ├───► DistilBERT Tokenizer ────► DistilBERT Encoders ────► Logits A ──┐
       │                                                                      ├──► Learnable Softmax Fusion ──► Fused Logits ──► Classification
       └───► XLM-RoBERTa Tokenizer ──► XLM-RoBERTa Encoders ──► Logits B ──┘                                                          │
                                                                                                                                      ▼
                                                                                                                            LLaMA-3 LLM Support Chat
```

---

## 📊 Experimental Results

| Metric | Early Fusion | **Late Fusion (Proposed)** |
| :--- | :---: | :---: |
| **Accuracy** | 74.44% | **76.68%** |
| **Precision** | 75.69% | **76.90%** |
| **Recall** | 74.44% | **76.68%** |
| **F1 Score** | 74.52% | **76.54%** |
| **MCC** | 0.7037 | **0.7288** |
| **Cohen Kappa** | 0.7018 | **0.7279** |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/saifullah100/mental-health-detection-late-fusion.git](https://github.com/saifullah100/mental-health-detection-late-fusion.git)
cd mental-health-detection-late-fusion
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
Open `http://localhost:5000` in your web browser.

---

## 📜 Citation
If you find this work or codebase useful in your research, please cite our IEEE paper:

```bibtex
@inproceedings{wiecon_ece2025_mental_health,
  author    = {Riad Rahman and Md Saifullah and Natasha Bose and Rubaiya Hafiz},
  title     = {Mental Health Detection using Late Fusion Transformer Models with LLM Chat Integration},
  booktitle = {2025 IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering (WIECON-ECE)},
  year      = {2025},
  pages     = {783--788},
  doi       = {10.1109/WIECON-ECE69386.2025.11526093},
  publisher = {IEEE}
}
```
