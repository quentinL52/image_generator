import os
import json
import base64
from datetime import date
from io import BytesIO
from typing import Optional

import modal
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

app_api = FastAPI(title="Solle Flux Generation API")

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key")
app = modal.App("flux-solle-api")

# --- Configuration Modal ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", 
        "diffusers>=0.31.0", 
        "transformers>=4.44.0", 
        "accelerate",
        "safetensors", 
        "sentencepiece", 
        "protobuf",
        "huggingface_hub",
        "hf_transfer",
        "bitsandbytes>=0.43.0",
        "peft", 
        "fastapi"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HUB_CACHE": "/cache/huggingface",
    })
)

model_cache = modal.Volume.from_name("flux-model-cache", create_if_missing=True)
lora_vol = modal.Volume.from_name("flux-lora-storage", create_if_missing=True)
usage = modal.Dict.from_name("user-usage", create_if_missing=True)

# --- Authentification ---
def get_api_key(api_key: str = Security(api_key_header)):
    tokens_env = os.environ.get("VALID_TOKENS", "")
    
    # Check if the api_key string is simply present in the env var string
    # This handles both proper JSON ["TOKEN"] and stripped powershell strings [TOKEN]
    if api_key not in tokens_env:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# --- Modèles Pydantic ---
class GenerateRequest(BaseModel):
    prompt: str
    lora_scale: float = 0.8
    width: int = 1024
    height: int = 1024

jobs_state = modal.Dict.from_name("jobs-state", create_if_missing=True)

# --- CLASSE SERVEUR GPU (Worker) ---
@app.cls(
    image=image,
    gpu="A100",
    memory=32768,
    timeout=600,
    scaledown_window=300, # Garde le GPU chaud pendant 5 min après une requête
    min_containers=0,
    volumes={
        "/cache": model_cache,
        "/workspace": lora_vol,
    },
    secrets=[
        modal.Secret.from_name("hf-secret")
    ],
)
class FluxGenerator:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import FluxPipeline

        print("Chargement du modèle Flux.1-dev en bfloat16...")

        # Chargement en bfloat16 pur — idéal pour Dev
        self.pipe_txt2img = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )

        # Trouver le fichier LoRA principal et l'éventuel LoRA anti-flou
        import glob
        target_name = "solle_flux_v2.safetensors"
        antiblur_name = "antiblur.safetensors"
        loras = glob.glob("/workspace/**/*.safetensors", recursive=True)
        print(f"Fichiers .safetensors trouvés dans le volume : {loras}")

        lora_path = next((l for l in loras if os.path.basename(l) == target_name), None)
        antiblur_path = next((l for l in loras if os.path.basename(l) == antiblur_name), None)

        # Chargement des adaptateurs LoRA avec PEFT
        if lora_path:
            print(f"Chargement du LoRA principal depuis {lora_path}...")
            self.pipe_txt2img.load_lora_weights(lora_path, adapter_name="solle")
            print(f"LoRA '{target_name}' chargé avec succès.")
        else:
            print(f"ERREUR CRITIQUE : '{target_name}' introuvable dans le volume !")

        if antiblur_path:
            print(f"Chargement du LoRA Anti-Blur depuis {antiblur_path}...")
            self.pipe_txt2img.load_lora_weights(antiblur_path, adapter_name="antiblur")
            print("LoRA Anti-Blur chargé avec succès.")
            self.has_antiblur = True
        else:
            print("Aucun LoRA Anti-Blur détecté (optionnel).")
            self.has_antiblur = False

        # VAE slicing/tiling pour économiser la VRAM
        self.pipe_txt2img.vae.enable_slicing()
        self.pipe_txt2img.vae.enable_tiling()

        # Pas besoin d'offload sur A100 (40 Go), on charge tout en VRAM pour une vitesse maximale
        self.pipe_txt2img.to("cuda")

        print("Pipeline prêt (A100 VRAM).")

    @modal.method()
    def generate(self, job_id: str, req: GenerateRequest):
        import torch
        from PIL import Image

        try:
            # Mettre à jour le statut
            jobs_state[job_id] = {"status": "processing"}

            prompt = req.prompt
            # Force le trigger complet (nom + classe) comme dans le dataset
            if "sollechar" not in prompt.lower():
                prompt = f"sollechar, purple furry monster, {prompt}"
            elif "purple furry monster" not in prompt.lower():
                prompt = prompt.replace("sollechar", "sollechar, purple furry monster")

            # S'assurer que les dimensions sont des multiples de 64 (requis pour Flux)
            req.width = (req.width // 64) * 64
            req.height = (req.height // 64) * 64

            # Configurer dynamiquement les poids des LoRAs actifs
            if self.has_antiblur:
                # Appliquer les deux LoRAs en même temps (AntiBlur réduit à 0.8 pour ne pas déformer le personnage)
                self.pipe_txt2img.set_adapters(["solle", "antiblur"], adapter_weights=[req.lora_scale, 0.8])
                print(f"Inférence avec double LoRA : Solle={req.lora_scale}, AntiBlur=0.8")
            else:
                self.pipe_txt2img.set_adapters(["solle"], adapter_weights=[req.lora_scale])
                print(f"Inférence avec LoRA unique : Solle={req.lora_scale}")

            # Paramètres d'inférence (Dev = 25 steps, guidance=3.5 est le standard)
            kwargs = {
                "prompt": prompt,
                "num_inference_steps": 25,
                "guidance_scale": 3.5,
                "width": req.width,
                "height": req.height,
            }

            result = self.pipe_txt2img(**kwargs).images[0]

            # Convertir en JPEG
            buffer = BytesIO()
            result.save(buffer, format="JPEG", quality=90)
            buffer.seek(0)
            
            # Stocker le résultat et marquer comme terminé
            jobs_state[job_id] = {"status": "completed", "image": buffer.getvalue()}

        except Exception as e:
            import traceback
            traceback.print_exc()
            jobs_state[job_id] = {"status": "failed", "error": str(e)}


# --- CLASSE SERVEUR CPU (API) ---
@app.cls(
    image=image,
    cpu=0.25,
    memory=256,
    min_containers=1, # Toujours allumé pour masquer la latence de l'API
    secrets=[
        modal.Secret.from_name("api-tokens")
    ],
)
class ApiServer:
    @modal.asgi_app()
    def web(self):
        from fastapi import Response
        import uuid
        
        @app_api.get("/stats")
        async def stats():
            today = str(date.today())
            global_key = f"global:{today}"
            images_today = usage.get(global_key, 0)
            
            current_spend = 0.0
            try:
                current_spend = modal.Usage.get_current_month()
            except AttributeError:
                pass

            return {
                "monthly_spend": current_spend,
                "images_today": images_today,
                "avg_generation_time": "~8s",
                "remaining_budget": max(0.0, 30.0 - current_spend)
            }

        @app_api.post("/generate")
        def generate_endpoint(request: GenerateRequest, api_key: str = Depends(get_api_key)):
            # Rate limiting check
            user_key = f"quota_{api_key}"
            count = usage.get(user_key, 0)
            if count >= 50:
                raise HTTPException(status_code=429, detail="Quota journalier atteint (50 images max).")
            
            job_id = str(uuid.uuid4())
            jobs_state[job_id] = {"status": "pending"}
            
            # Exécution ASYNCHRONE sur le GPU via spawn()
            FluxGenerator().generate.spawn(job_id, request)
            
            # Mise à jour du compteur
            usage[user_key] = count + 1
            
            # Retourne le job_id immédiatement
            return {"job_id": job_id, "status": "pending"}

        @app_api.get("/status/{job_id}")
        def status_endpoint(job_id: str):
            state = jobs_state.get(job_id)
            if not state:
                raise HTTPException(status_code=404, detail="Job introuvable")
            
            # Ne pas renvoyer les bytes de l'image dans l'endpoint de statut
            return {
                "job_id": job_id,
                "status": state.get("status"),
                "error": state.get("error")
            }

        @app_api.get("/image/{job_id}")
        def image_endpoint(job_id: str):
            state = jobs_state.get(job_id)
            if not state:
                raise HTTPException(status_code=404, detail="Job introuvable")
            
            if state.get("status") != "completed":
                raise HTTPException(status_code=400, detail=f"Image non prête. Statut actuel: {state.get('status')}")
            
            image_bytes = state.get("image")
            if not image_bytes:
                raise HTTPException(status_code=500, detail="Image perdue")
                
            # Nettoyer pour libérer la mémoire du Dict
            del jobs_state[job_id]
            
            return Response(content=image_bytes, media_type="image/jpeg")

        return app_api
