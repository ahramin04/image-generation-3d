
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from typing import List
import os
import shutil
from utils.image_generator import generate_image_with_facefusion

app = FastAPI(title="Prompt-Based Scene + Face Fusion Generator")

MODELS_DIR = "models"
BASE_MODELS_DIR = os.path.join(MODELS_DIR, "base")
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

@app.get("/models/", response_model=List[str])
def get_models():
    if not os.path.exists(BASE_MODELS_DIR):
        return []
    return [f for f in os.listdir(BASE_MODELS_DIR) if f.endswith(".safetensors")]

@app.get("/loras/", response_model=List[str])
def get_loras():
    loras_path = os.path.join(MODELS_DIR, "loras")
    return [f for f in os.listdir(loras_path) if f.endswith(".safetensors")]

@app.post("/generate/")
async def generate_image(
    prompt: str = Form(...),
    model_name: str = Form(...),
    lora_name: str = Form(None),
    image: UploadFile = File(...)
):
    input_path = os.path.join(INPUTS_DIR, image.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    model_path = os.path.join(BASE_MODELS_DIR, model_name)
    lora_path = os.path.join(MODELS_DIR, "loras", lora_name) if lora_name else None
    output_path = generate_image_with_facefusion(prompt, input_path, model_path, lora_path)
    return FileResponse(output_path, media_type="image/png", filename=os.path.basename(output_path))
