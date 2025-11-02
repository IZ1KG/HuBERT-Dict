# Copyright (c) 2025 Itzik Galanti. All rights reserved.
# Personal Use — No Modification / No Redistribution. View-only; no use/copy/modify/distribute without written permission.


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, sys, time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import sounddevice as sd
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# ===== פרמטרים בסיס =====
SR = 16000
MODEL_NAME = "facebook/hubert-base-ls960"
DEVICE = "cpu"
JSON_VERSION = "simple-1.0"

# הקלטה
DEFAULT_SECONDS = 1.5     # משך כל הקלטה/חלון (שניות)
TOP_DB_TRIM = 35          # טרימינג שקט

# רגישות (סף RMS) — ניתן לשינוי ב-CLI
RMS_MIN_DEFAULT = 0.004   # MED
RMS_MIN = RMS_MIN_DEFAULT

SENS_TO_RMS = {
    "low":   0.008,  # פחות רגיש
    "med":   0.004,  # ברירת מחדל
    "high":  0.002,  # רגיש יותר
    "ultra": 0.001,  # רגיש מאוד (זהיר מרעש)
}

# ספי התאמה (cosine distance = 1 - cos)
TAU_MIN = 0.12            # תחתית סף (שמרני)
TAU_MAX = 0.35            # תקרה (נגד נדיבות)
TAU_SINGLE = 0.22         # כשיש רק דוגמה אחת למילה

_feature_extractor = None
_model = None

# ===== מודל =====
def load_model():
    global _feature_extractor, _model
    if _model is None:
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        _model = HubertModel.from_pretrained(MODEL_NAME).to(DEVICE)
        _model.eval()

def l2_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v) + eps)
    return (v / n).astype(np.float32)

def embed(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    אודיו → embedding (HuBERT mean-pooling + L2). מחזיר None אם שקט/רעש.
    """
    load_model()
    y = np.asarray(y, dtype=np.float32).flatten()
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # טרימינג שקט
    y, _ = librosa.effects.trim(y, top_db=TOP_DB_TRIM)
    if len(y) == 0:
        return None

    # RMS סף (רגישות)
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    if rms < RMS_MIN:
        return None

    # נרמול עוצמה עדין
    y = y / max(rms, 1e-2)

    with torch.no_grad():
        inputs = _feature_extractor(y, sampling_rate=SR, return_tensors="pt")
        x = inputs.input_values.to(DEVICE)
        h = _model(x).last_hidden_state.squeeze(0)   # [T,D]
        v = h.mean(dim=0).cpu().numpy().astype(np.float32)  # [D]
        return l2_normalize(v)

# ===== מאגר JSON פר-ילד =====
def child_path(child_id: str) -> Path:
    return Path(f"child_{child_id}.json")

def load_child(child_id: str) -> Dict:
    p = child_path(child_id)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    doc = {"version": JSON_VERSION, "model": MODEL_NAME, "child_id": str(child_id), "words": []}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return doc

def save_child(doc: Dict):
    if "version" not in doc: doc["version"] = JSON_VERSION
    if "model" not in doc: doc["model"] = MODEL_NAME
    with open(child_path(doc["child_id"]), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

def find_word(doc: Dict, label: str) -> Optional[Dict]:
    for w in doc["words"]:
        if w["label"] == label:
            return w
    return None

# ===== מתמטיקה =====
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cos_sim(a, b)  # קטן=דומה

def compute_centroid_and_tau(vectors: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    V = np.stack(vectors, axis=0).astype(np.float32)
    c = l2_normalize(V.mean(axis=0))
    if len(V) == 1:
        tau = TAU_SINGLE
    else:
        dists = np.array([cos_dist(v, c) for v in V], dtype=np.float32)
        mu, sd = float(dists.mean()), float(dists.std() + 1e-6)
        tau = mu + 2.0 * sd
        tau = float(np.clip(tau, TAU_MIN, TAU_MAX))
    return c, tau

# ===== הקלטה =====
def coerce_device(dev):
    if dev is None: return None
    s = str(dev).strip()
    if s.isdigit(): return int(s)
    return s

def record_seconds(seconds: float, sr: int = SR, device=None) -> Optional[Tuple[np.ndarray, int]]:
    """הקלטה חסימתית קצרה, Mono, float32."""
    try:
        if device is not None:
            sd.default.device = device
        n = int(sr * seconds)
        buf = sd.rec(frames=n, samplerate=sr, channels=1, dtype="float32", blocking=False)
        sd.wait()
        y = buf.reshape(-1)
        return y, sr
    except Exception as e:
        print(f"❌ שגיאת הקלטה: {e}")
        return None

# ===== עזר רגישות =====
def apply_sensitivity_args(args):
    """קובע RMS_MIN לפי --rms-min או --sensitivity."""
    global RMS_MIN
    if getattr(args, "rms_min", None) is not None:
        RMS_MIN = float(args.rms_min)
    elif getattr(args, "sensitivity", None):
        key = str(args.sensitivity).lower()
        RMS_MIN = SENS_TO_RMS.get(key, RMS_MIN_DEFAULT)
    else:
        RMS_MIN = RMS_MIN_DEFAULT
    print(f"[sensitivity] RMS_MIN set to {RMS_MIN:.6f}")

# ===== add-word =====
def cmd_add_word(args):
    apply_sensitivity_args(args)

    child = str(args.child)
    label = args.label.strip()
    n = int(args.n)
    seconds = float(args.seconds)
    device = coerce_device(args.device)

    doc = load_child(child)
    word = find_word(doc, label)
    if word is None:
        word = {"label": label, "vectors": [], "centroid": None, "tau": None}
        doc["words"].append(word)

    print(f"[add-word] child={child} label='{label}' n={n} seconds={seconds} device={device}")
    got, tries = 0, 0
    while got < n:
        i = got + 1
        print(f"אמירה {i}/{n} — דבר/י עכשיו...")
        rec = record_seconds(seconds, SR, device)
        if rec is None:
            tries += 1
            if tries > n * 5:
                print("ויתור אחרי יותר מדי נסיונות הקלטה.")
                break
            continue
        (y, sr) = rec
        v = embed(y, sr)
        if v is None:
            print("❌ חלש מדי/שקט. ננסה שוב.")
            tries += 1
            if tries > n * 5:
                print("ויתור אחרי יותר מדי נסיונות.")
                break
            continue
        word["vectors"].append(v.tolist())
        got += 1
        print("✅ נקלטה דוגמה.")

    if got >= 1:
        vecs = [np.array(u, dtype=np.float32) for u in word["vectors"]]
        c, tau = compute_centroid_and_tau(vecs)
        word["centroid"] = c.tolist()
        word["tau"] = float(tau)
        save_child(doc)
        print(f"✓ נשמר. child={child} label='{label}', דוגמאות={len(vecs)}, τ={tau:.3f}")
    else:
        print("❌ לא נשמרו דוגמאות.")

# ===== listen =====
def cmd_listen(args):
    apply_sensitivity_args(args)

    child = str(args.child)
    seconds = float(args.seconds)
    pause = float(args.pause)
    device = coerce_device(args.device)

    doc = load_child(child)
    if not doc["words"]:
        print("אין מילים במילון. הוסף קודם עם: python app.py add-word --child 1 --label \"...\" -n 3")
        return

    # ודא שלכל מילה יש centroid/tau
    changed = False
    for w in doc["words"]:
        if (not w.get("centroid")) or (w.get("tau") is None):
            vecs = [np.array(u, dtype=np.float32) for u in w.get("vectors", [])]
            if not vecs:
                continue
            c, tau = compute_centroid_and_tau(vecs)
            w["centroid"] = c.tolist()
            w["tau"] = float(tau)
            changed = True
    if changed:
        save_child(doc)

    print(f"[listen] child={child} seconds={seconds} device={device}  (Ctrl+C ליציאה)")
    try:
        while True:
            rec = record_seconds(seconds, SR, device)
            if rec is None:
                time.sleep(pause)
                continue
            (y, sr) = rec
            v = embed(y, sr)
            if v is None:
                time.sleep(pause)
                continue

            best_label, best_d, best_tau = None, float("inf"), None
            for w in doc["words"]:
                if not w.get("centroid") or (w.get("tau") is None):
                    continue
                c = np.array(w["centroid"], dtype=np.float32)
                d = cos_dist(v, c)  # קטן=טוב
                if d < best_d:
                    best_d = d
                    best_label = w["label"]
                    best_tau = float(w["tau"])

            if best_label is not None and best_d <= best_tau:
                print(f"✅ {best_label}  (dist={best_d:.3f} ≤ τ={best_tau:.3f})")

            time.sleep(pause)
    except KeyboardInterrupt:
        print("\nסיום.")

# ===== CLI =====
def main():
    p = argparse.ArgumentParser(description="Minimal semi-verbal matcher (HuBERT + centroid + tau)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("add-word", help="הוספת מילה מהמיקרופון")
    pa.add_argument("--child", default="1")
    pa.add_argument("--label", required=True)
    # לפני:
    # pa.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="משך כל דגימה (שניות)")
    # אחרי (ברירת־מחדל 2.0 שניות):
    pa.add_argument("--seconds", type=float, default=2.0, help="משך כל דגימה (שניות) — ברירת מחדל 2.0")
    pa.add_argument("-n", type=int, default=5, help="מספר דוגמאות (ברירת מחדל 5)")
    pa.add_argument("--device", default=None, help="שם/אינדקס התקן קלט")
    # לפני:
    # pa.add_argument("--sensitivity", choices=list(SENS_TO_RMS.keys()), help="רגישות המיקרופון: low/med/high/ultra")
    # אחרי (ברירת־מחדל high):
    pa.add_argument("--sensitivity", choices=list(SENS_TO_RMS.keys()), default="high",
                    help="רגישות המיקרופון: low/med/high/ultra (ברירת מחדל high)")
    pa.add_argument("--rms-min", type=float, help="סף RMS ידני (עוקף sensitivity)")
    pa.set_defaults(func=cmd_add_word)


    pl = sub.add_parser("listen", help="האזנה בלייב וזיהוי")
    pl.add_argument("--child", default="1")
    pl.add_argument("--seconds", type=float, default=2.0, help="משך כל חלון (שניות) — ברירת מחדל 2.0")
    pl.add_argument("--pause", type=float, default=0.1, help="הפסקה קצרה בין חלונות")
    pl.add_argument("--device", default=None, help="שם/אינדקס התקן קלט")
    pl.add_argument("--sensitivity", choices=list(SENS_TO_RMS.keys()), default="high", help="רגישות המיקרופון: low/med/high/ultra (ברירת מחדל high)")
    pl.add_argument("--rms-min", type=float, help="סף RMS ידני (עוקף sensitivity)")
    pl.add_argument("--verbose", action="store_true", help="הדפסת סטטוס בכל חלון")
    pl.set_defaults(func=cmd_listen)


    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    except Exception:
        pass
    main()



# Copyright (c) 2025 Itzik Galanti. All rights reserved.
# Personal Use — No Modification / No Redistribution. View-only; no use/copy/modify/distribute without written permission.
