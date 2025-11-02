# HuBERT-Dict — Personal Sound Dictionary (Assistive AI)

> ⚠️ **View-Only / Proprietary — All Rights Reserved**  
> This repository is provided **for viewing and evaluation only**.  
> **No permission** is granted to copy, use, modify, merge, publish, distribute, or sublicense any part of this code without prior written consent from the author. See **LICENSE** and **NOTICE**.

> **Language note (Hebrew-first):** This project is configured and documented primarily for **Hebrew** usage (labels, examples, CLI output).  
> The acoustic matcher itself is language-agnostic, but provided flows and examples target **Hebrew caregiving contexts**.

---

## ✨ What this build includes
This README matches the **minimal CLI** in `app.py` you shared:
- `add-word` — record N short examples for a label and build/update its profile
- `listen` — continuous fixed-window listening and classification
- Default window: **2.0s**; default mic sensitivity: **high**
- Per‑child JSON auto‑created as `child_<id>.json` (no folders to set up)

> Not included in this build: voice‑trigger / pre‑roll, list/delete commands, unknown logging.  
> JSON version used here: **simple-1.0**.

---

## 🔧 Requirements
- **Python 3.9+**
- Python packages: `torch`, `transformers`, `librosa`, `sounddevice`, `numpy`  
  (Optional but harmless in `requirements.txt`: `soundfile`)

Linux audio backend (if needed):
```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 libsndfile1
```
> HuBERT runs on **CPU**. First run downloads the model to your local HuggingFace cache.  
> **Windows UTF-8 tip:** If Hebrew text looks garbled, use a modern PowerShell or run `chcp 65001`.

---

## 🚀 Installation
```bash
git clone https://github.com/IZ1KG/HuBERT-Dict.git
cd HuBERT-Dict

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install --upgrade pip
pip install torch transformers librosa sounddevice numpy
# (optional) pip install soundfile
```

List audio devices (optional):
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

---

## ⚙️ Quick Start
**Train a word (collect 5 examples)** — defaults: `--seconds 2.0`, `--sensitivity high`
```bash
python app.py add-word --child 1 --label "מים" -n 5
```

**Live recognition (continuous windows)**
```bash
python app.py listen --child 1
```
Options:
- `--seconds 2.0` — fixed window length (seconds)  
- `--sensitivity low|med|high|ultra` (default `high`) or `--rms-min <float>`  
- `--device <index/name>` — specific microphone device
- `--pause 0.1` — short sleep between windows

---

## 🧠 How It Works (high level)
1. **Record** a fixed window (default 2.0s).  
2. **Preprocess**: trim silence; reject too‑quiet windows by **RMS threshold** (sensitivity).  
3. **Embed** with **HuBERT** (mean‑pool, L2‑normalize).  
4. **Compare** to each label’s **centroid** using **cosine distance** and accept only if `dist ≤ τ` (adaptive per‑label threshold).

---

## 🗂️ Data Format
Per-child file: `child_<id>.json`
```json
{
  "version": "simple-1.0",
  "model": "facebook/hubert-base-ls960",
  "child_id": "1",
  "words": [
    {
      "label": "מים",
      "vectors": [[0.01, -0.02, "..."], ["..."]],
      "centroid": [0.05, -0.01, "..."],
      "tau": 0.21
    }
  ]
}
```

---

## 🧰 Troubleshooting
- **No detections** → speak slightly louder; try `--sensitivity ultra` or lower `--rms-min` (e.g., `0.002`).  
- **Many false positives** → add more clean samples for each label (`-n 5` or more).  
- **Device errors (Linux)** → ensure `libportaudio2`/`libsndfile1`; pick a `--device` index.

---

## 🏷️ License
**Proprietary — All Rights Reserved.** View‑only. Any use beyond viewing requires explicit written permission.  
See **LICENSE** and **NOTICE**.

---

## © Acknowledgements
HuBERT Base LS‑960 (HuggingFace: `facebook/hubert-base-ls960`) · PyTorch · Transformers · librosa · sounddevice

© 2025 Itzik Galanti. All rights reserved.
