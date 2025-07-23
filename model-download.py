import os
from huggingface_hub import hf_hub_download


target_folder = os.getenv("f5tts")
os.makedirs(target_folder, exist_ok=True)

files = [
    "model_last_v2.safetensors",
    "setting.json",
    "vocab.txt"
]
repo_id = "PapaRazi/Ijazah_Palsu_V2"

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
