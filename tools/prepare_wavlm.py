#!/usr/bin/env python3
"""
WavLM Model Preparation Tool (Milestone M6).
Resolves and pins an immutable Hugging Face revision, downloads weights to HF_HOME,
validates with trust_remote_code=False, and writes the persistent config/wavlm.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def prepare_wavlm(model_id: str, config_path: str, layer: int = 6) -> str:
    cfg_p = Path(config_path).resolve()
    cfg_p.parent.mkdir(parents=True, exist_ok=True)

    # Set HF_HOME to .build/huggingface if not set
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(REPO_ROOT / ".build" / "huggingface")

    print(f"[*] Preparing WavLM model: {model_id}")
    print(f"    HF_HOME: {os.environ['HF_HOME']}")
    print(f"    Target Config: {cfg_p}")

    # Check if existing config has a pinned revision
    existing_revision = None
    if cfg_p.exists():
        try:
            with open(cfg_p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model_id") == model_id and data.get("revision"):
                existing_revision = data["revision"]
                print(f"[+] Reusing existing pinned revision: {existing_revision}")
        except Exception:
            pass

    revision = existing_revision
    if not revision:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.model_info(model_id)
            revision = info.sha
            print(f"[+] Resolved immutable commit revision: {revision}")
        except Exception as e:
            print(f"[!] Warning: Could not resolve commit sha via HfApi ({e}). Falling back to 'main'")
            revision = "main"

    revision_str: str = str(revision) if revision else "main"

    # Download and validate model
    try:
        from transformers import AutoFeatureExtractor, WavLMModel
        print("[*] Validating and caching model weights (trust_remote_code=False)...")
        AutoFeatureExtractor.from_pretrained(
            model_id,
            revision=revision_str,
            trust_remote_code=False,
        )
        model = WavLMModel.from_pretrained(
            model_id,
            revision=revision_str,
            trust_remote_code=False,
        )
        print(f"[+] Successfully loaded and verified {model_id} @ {revision_str}")
    except Exception as e:
        raise RuntimeError(f"Failed to load and validate WavLM model: {e}")

    # Write persistent config
    config_data = {
        "model_id": model_id,
        "revision": revision,
        "layer": layer,
        "feature_dim": model.config.hidden_size if hasattr(model, "config") else 768,
        "sample_rate": 16000,
    }

    with open(cfg_p, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"[+] WavLM configuration saved to: {cfg_p}")
    return str(cfg_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and pin WavLM model checkpoint.")
    parser.add_argument("--model", default="microsoft/wavlm-base-plus", help="Hugging Face model ID")
    parser.add_argument("--config", default="./config/wavlm.json", help="Output config path")
    parser.add_argument("--layer", type=int, default=6, help="Target hidden layer profile")
    args = parser.parse_args()

    prepare_wavlm(args.model, args.config, layer=args.layer)
