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
model_path = target_folder + "/model_last_v2.safetensors"
config_path = target_folder + "/setting.json"
vocab_path = target_folder + "/vocab.txt"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
exp_name = config.get("exp_name", "F5TTS_v1_Base")

tts_model = F5TTS(
    model=exp_name,
    ckpt_file=str(model_path),
    vocab_file=str(vocab_path),
    device=device,
    use_ema=True,
    vocoder_local_path=target_folder
)
logger.info("Model loaded successfully")

voice_dir = "voices"
voices_lib = "voices.json"
with open(voices_lib, "r", encoding="utf-8") as f:
    supported_voices = json.load(f)
default_voice = list(supported_voices.keys())[0]

class TTSRequest(BaseModel):
    model: str = Field(default="dummy", description="TTS model to use")
    response_format: str = Field(default="wav", description="Response format")
    input: str = Field(default="Hai apa kabar?", description="Text to synthesize")
    voice: str = Field(default="", description="Voice to use for synthesis")

# FastAPI app
app = FastAPI(
    title="TTS Server",
    description="Text-to-speech API Server",
    version="1"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

voice_object = {}

def audio_generator(text, voice_path, voice_ref_text) -> Generator[bytes, None, None]:
    if voice_path in voice_object:
        ref_text = voice_object[voice_path]["ref_text"]
        audio = voice_object[voice_path]["audio"]
        sr = voice_object[voice_path]["sr"]
        max_chars = voice_object[voice_path]["max_chars"]
        few_chars = voice_object[voice_path]["few_chars"]
        min_chars = voice_object[voice_path]["min_chars"]
        
    else:
        ref_file, ref_text = preprocess_ref_audio_text(voice_path, voice_ref_text)
        audio, sr = torchaudio.load(ref_file)
        ref_audio_duration = audio.shape[-1] / sr
        ref_text_byte_len = len(ref_text.encode("utf-8"))
        max_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration))
        few_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration) / 2)
        min_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration) / 4)
        
        voice_object[voice_path] = {
            "ref_text": ref_text,
            "audio": audio,
            "sr": sr,
            "max_chars": max_chars,
            "few_chars": few_chars,
            "min_chars": min_chars
        }
        print('max_chars', max_chars, 'few_chars', few_chars, 'min_chars', min_chars)

    # Clean and normalize the input text
    text = text.strip()
    if not text:
        logger.info("Empty text input, skipping generation")
        return

    # More careful text chunking
    text_batches = chunk_text(text, max_chars=125)
    print('original', text_batches)
    
    if not text_batches:
        logger.info("No valid text batches after processing")
        return

    logger.info(f"Text: {text}")
    logger.info(f"Text batches {len(text_batches)}: {text_batches}")

    try:
        audio_stream = infer_batch_process(
            (audio, sr),
            ref_text,
            text_batches,
            tts_model.ema_model,
            tts_model.vocoder,
            tts_model.mel_spec_type,
            progress=None,
            device=device,
            streaming=True,
            speed=1
        )
    except:
        return

    for i, (audio_chunk, _) in enumerate(audio_stream):
        if len(audio_chunk) > 0:
            try:
                logger.info(f"Audio chunk {i} of size: {len(audio_chunk)}, {text_batches[i]}")
            except:
                pass

            try:
                wav = np.array(audio_chunk).reshape(1, -1)
                wav = (wav * 32767).astype(np.int16)
                chunk = wav.tobytes()
                yield chunk
            except:
                yield None
            

@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):    
    try:
        text = request.input
        voice = request.voice
        if voice not in supported_voices:
            voice = default_voice

        voice_path = voice_dir + "/" + supported_voices.get(voice).get("file_name")
        voice_ref_text = supported_voices.get(voice).get("transcript")

        return StreamingResponse(
            audio_generator(text, voice_path, voice_ref_text),
            media_type="audio/wav",
        )
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/voices")
async def list_voices():
    return {"voices": supported_voices, "default_voice": default_voice}

@app.post("/v1/voices/set")
async def set_default_voice(voice_name: str = Form(...)):
    global default_voice
    previous_voice = default_voice
    default_voice = voice_name
    return {"message": f"Default voice set to '{default_voice}', previous voice: '{previous_voice}'"}

@app.post("/v1/voices/upload")
async def upload_voice(
    voice_name: str = Form(...),
    file: UploadFile = File(...)
):
    global supported_voices
    global voices_lib

    # Validate file type
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")
    
    try:
        # Save the uploaded file
        voice_path = voice_dir + "/" + voice_name + ".wav"
        with open(voice_path, "wb") as f:
            content = await file.read()
            f.write(content)

        _, ref_text = preprocess_ref_audio_text(voice_path, "")

        supported_voices[voice_name] = {
            "file_name": voice_name + ".wav",
            "transcript": ref_text
        }
        
        with open(voices_lib, "w", encoding="utf-8") as f:
            json.dump(supported_voices, f, indent=4)
        
        return {"message": f"Voice '{voice_name}' uploaded successfully"}
        
    except Exception as e:
        logger.error(f"Error uploading voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/voices/{voice_name}")
async def get_voice_audio(voice_name: str):
    if voice_name not in supported_voices:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    
    try:
        voice_file_name = supported_voices[voice_name]["file_name"]
        voice_path = f"{voice_dir}/{voice_file_name}"
        
        if not os.path.exists(voice_path):
            raise HTTPException(status_code=404, detail=f"Audio file for voice '{voice_name}' not found")
        
        def file_stream():
            with open(voice_path, "rb") as file:
                while chunk := file.read(2048):
                    yield chunk
        
        return StreamingResponse(
            file_stream(),
            media_type="audio/wav"
        )
    except Exception as e:
        logger.error(f"Error retrieving voice audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


uvicorn.run(
    app,
    host="0.0.0.0",
    port=int(os.getenv("PORT_TTS")),
)
