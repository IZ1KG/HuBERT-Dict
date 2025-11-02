# HuBERT-Dict — Personal Sound Dictionary (Assistive AI)

> ⚠️ **View-Only / Proprietary — All Rights Reserved**  
> This repository is provided **for viewing and evaluation only**.  
> **No permission** is granted to copy, use, modify, merge, publish, distribute, or sublicense any part of this code without prior written consent from the author. See **LICENSE** and **NOTICE**.

> **Language note (Hebrew-first):** This project is configured and documented primarily for **Hebrew** usage (labels, examples, CLI output).  
> The acoustic matcher itself is language-agnostic, but provided flows and examples target **Hebrew caregiving contexts**.


---

## ✨ Features
- 🎙️ Live microphone recognition (fixed window) + optional **voice-trigger with pre-roll** (avoid missed openings)
- 👤 Per-child local JSON storage (privacy by default)
- 🧠 Simple, fast matcher: HuBERT → vector → compare to word profiles with clear acceptance rules
- 🛠️ CLI-first workflow: add words, listen, list, delete, reset—no GUI needed
- 🧩 Cross-platform: Windows / Linux / macOS

---

## 🔧 Requirements
- **Python 3.9+**
- Python packages: `torch`, `transformers`, `librosa`, `sounddevice`, `soundfile`, `numpy`
- Audio backend (Linux only):
  ```bash
  sudo apt-get update
  sudo apt-get install -y libportaudio2 libsndfile1
  ```
> HuBERT runs on **CPU**. First run downloads the model to your local HuggingFace cache.  
> **Windows UTF-8 tip:** If Hebrew text looks garbled, use a modern PowerShell or run `chcp 65001`.

---

## 🚀 Installation (reviewers with permission)
```bash
# Clone
git clone https://github.com/IZ1KG/HuBERT-Dict.git
cd HuBERT-Dict

# Virtual env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Dependencies
pip install --upgrade pip
pip install torch transformers librosa sounddevice soundfile numpy
```

Optional: list audio devices to find your mic index
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

---

## ⚙️ Quick Start (with permission)
**Add a word (collect 5 examples)** — defaults: `--seconds 2.0`, `--sensitivity high`
```bash
python app.py add-word --child 1 --label "מים" -n 5
```

**Live recognition**
```bash
python app.py listen --child 1
```

**Voice-trigger mode (optional)**
```bash
python app.py listen --child 1 --voice-trigger --preroll-ms 400 --verbose
```

---

## 🧪 CLI Usage
### Add samples (train a word)
```bash
# New word with 5 samples
python app.py add-word --child 1 --label "טיול" -n 5

# Add more samples later (improves robustness)
python app.py add-word --child 1 --label "טיול" -n 3

# Re-record from scratch (keeps the label, wipes old samples)
python app.py add-word --child 1 --label "טיול" --reset-first -n 5
```

### Live listen / predict
```bash
# Basic
python app.py listen --child 1

# Use a specific mic device (see index from 'list devices')
python app.py listen --child 1 --device 2

# Voice-trigger with pre-roll
python app.py listen --child 1 --voice-trigger --preroll-ms 400 --verbose
```

### Manage dictionary
```bash
# List words and stats
python app.py list-words --child 1

# Delete a word
python app.py delete-word --child 1 --label "טיול"
```

### Sensitivity (microphone loudness)
- Presets: `--sensitivity low|med|high|ultra` (default `high`)
- Or explicit RMS threshold: `--rms-min 0.002`

Examples:
```bash
python app.py listen --child 1 --sensitivity ultra
python app.py add-word --child 1 --label "מים" -n 5 --rms-min 0.002
```

---

## 🧠 How It Works (high level)
1. **Record** a short mic window (default 2.0s). Optional **voice-trigger** waits for speech and includes ~400 ms **pre-roll**.  
2. **Preprocess** audio: trim silence, check minimum loudness (RMS), light normalization.  
3. **Embed** with **HuBERT** (mean-pool across time) → L2-normalized vector.  
4. **Compare** to each word’s profile (**centroid**) and accept only if:  
   - Best distance is within that word’s **confidence threshold**, and  
   - It’s clearly better than the **second best** (margin/ratio check).  
   → reduces false matches and stabilizes decisions.

---

## 🗂️ Data Format
Per-child file: `child_<id>.json`
```json
{
  "version": "simple-1.2",
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
- `vectors`: raw training embeddings (per sample)  
- `centroid`: per-word profile (mean vector, normalized)  
- `tau`: dynamic confidence threshold for this word

---

## 🧰 Troubleshooting
- **No output while listening** → speak a bit louder or try `--sensitivity ultra` / `--rms-min 0.002`.  
  You can also increase window: `--seconds 2.5`.  
- **Everything matches one word** → add more clean samples for that word or re-record with `--reset-first` (5–7 examples per word usually stabilizes thresholds).  
- **Device/sample-rate errors (Linux)** → ensure `libportaudio2` and `libsndfile1` are installed; choose a specific `--device` index if needed.  
- **Hebrew/Unicode on Windows** → use a modern PowerShell or `chcp 65001`.

---

## 🔐 Privacy & Hebrew Usage
- This repository’s docs and flows target **Hebrew caregiving contexts**; labels/examples and CLI output are in Hebrew.  
- All audio/embeddings are **local by default**. No cloud calls during inference.

---

## 🏷️ License
**Proprietary — All Rights Reserved.**  
This repository is for **view-only**. Any use beyond viewing requires **explicit written permission** from the copyright
holder. See **LICENSE** and **NOTICE**.

---

## © Acknowledgements
**HuBERT Base LS-960** (HuggingFace: `facebook/hubert-base-ls960`) · PyTorch · Transformers · librosa · sounddevice · soundfile

© 2025 Itzik Galanti. All rights reserved.
