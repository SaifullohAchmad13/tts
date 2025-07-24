import os
from typing import Generator
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import torch
from huggingface_hub import hf_hub_download
import json
from f5_tts.api import F5TTS
import torchaudio
import logging
from f5_tts.infer.utils_infer import (
    infer_batch_process,
    chunk_text,
    preprocess_ref_audio_text
)
from f5_tts.model.utils import seed_everything
from dotenv import load_dotenv

load_dotenv()
seed_everything(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

logger.info(f"Loading model on device: {device}")
target_folder = os.getenv("TTS_MODEL_DIR")
os.makedirs(target_folder, exist_ok=True)

def download_model(repo_id, files, target_folder):
    for filename in files:
        file_path = os.path.join(target_folder, filename)
        if os.path.exists(file_path):
            print(f"Skipping {filename}: already exists.")
            continue

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=target_folder,
            local_dir=target_folder,
            local_dir_use_symlinks=False
        )

download_model(
    "PapaRazi/Ijazah_Palsu_V2",
    [
        "model_last_v2.safetensors",
        "setting.json",
        "vocab.txt"
    ],
    target_folder
)
download_model(
    "charactr/vocos-mel-24khz",
    [
        "config.yaml",
        "pytorch_model.bin"
    ],
    target_folder
)
