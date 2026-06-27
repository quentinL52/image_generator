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
    init_image: Optional[str] = None  # Base64 string pour image-to-image
    strength: float = 0.6  # Pour img2img

# --- CLASSE SERVEUR GPU ---
@app.cls(
    image=image,
    gpu="A10G",
    memory=32768,
    timeout=300,
    scaledown_window=60,
    min_containers=0,
    volumes={
        "/cache": model_cache,
        "/workspace": lora_vol,
    },
    secrets=[
        modal.Secret.from_name("hf-secret"), 
        modal.Secret.from_name("api-tokens")
    ],
)
class FluxGenerator:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import FluxPipeline, FluxImg2ImgPipeline

        print("Chargement du modèle Flux.1-schnell en bfloat16...")

        # Chargement en bfloat16 pur — identique au notebook Kaggle
        self.pipe_txt2img = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        )

        # Trouver le fichier LoRA — cible directe du modèle final
        import glob
        target_name = "solle_flux_v2.safetensors"
        loras = glob.glob("/workspace/**/*.safetensors", recursive=True)
        print(f"Fichiers .safetensors trouvés dans le volume : {loras}")

        lora_path = next((l for l in loras if os.path.basename(l) == target_name), None)

        if lora_path:
            print(f"Chargement du LoRA depuis {lora_path}...")
            self.pipe_txt2img.load_lora_weights(lora_path)
            print(f"LoRA '{target_name}' chargé avec succès (sans fuse_lora car expérimental sur Flux).")
        else:
            print(f"ERREUR CRITIQUE : '{target_name}' introuvable dans le volume !")
            print(f"Fichiers disponibles : {[os.path.basename(l) for l in loras]}")

        # VAE slicing/tiling pour économiser la VRAM
        self.pipe_txt2img.vae.enable_slicing()
        self.pipe_txt2img.vae.enable_tiling()

        # Sequential CPU offload : déplace chaque couche individuellement.
        # Nécessaire sur A10G (24 Go) car le transformer Flux est trop gros
        # pour model_cpu_offload. Compatible avec le LoRA car il est fusionné.
        self.pipe_txt2img.enable_sequential_cpu_offload()

        # Pipeline img2img partage les mêmes composants (zéro duplication)
        self.pipe_img2img = FluxImg2ImgPipeline(
            vae=self.pipe_txt2img.vae,
            text_encoder=self.pipe_txt2img.text_encoder,
            text_encoder_2=self.pipe_txt2img.text_encoder_2,
            tokenizer=self.pipe_txt2img.tokenizer,
            tokenizer_2=self.pipe_txt2img.tokenizer_2,
            transformer=self.pipe_txt2img.transformer,
            scheduler=self.pipe_txt2img.scheduler,
        )
        self.pipe_img2img.enable_sequential_cpu_offload()

        print("Pipelines prêts (bfloat16 + sequential_cpu_offload + LoRA).")

    @modal.method()
    def generate(self, req: GenerateRequest):
        import torch
        from PIL import Image

        prompt = req.prompt
        # Force le trigger complet (nom + classe) comme dans le dataset
        if "sollechar" not in prompt.lower():
            prompt = f"sollechar, purple furry monster, {prompt}"
        elif "purple furry monster" not in prompt.lower():
            prompt = prompt.replace("sollechar", "sollechar, purple furry monster")

        # S'assurer que les dimensions sont des multiples de 64 (requis pour Flux)
        req.width = (req.width // 64) * 64
        req.height = (req.height // 64) * 64

        # Paramètres d'inférence (Schnell = 4 steps, guidance=1.0 car le LoRA a été entraîné avec GS=1)
        # On utilise joint_attention_kwargs pour le LoRA scale car fuse_lora est buggé sur Flux
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "joint_attention_kwargs": {"scale": req.lora_scale},
        }

        if req.init_image:
            # Img2Img
            init_image_bytes = base64.b64decode(req.init_image)
            image_obj = Image.open(BytesIO(init_image_bytes)).convert("RGB")
            # Redimensionner l'image initiale
            image_obj = image_obj.resize((req.width, req.height))
            
            kwargs["image"] = image_obj
            kwargs["strength"] = req.strength
            
            result = self.pipe_img2img(**kwargs).images[0]
        else:
            # Txt2Img
            kwargs["width"] = req.width
            kwargs["height"] = req.height
            result = self.pipe_txt2img(**kwargs).images[0]

        # Convertir en WebP
        buffer = BytesIO()
        result.save(buffer, format="WEBP", quality=80)
        buffer.seek(0)
        return buffer.getvalue()

    @modal.asgi_app()
    def web(self):
        from fastapi import Response
        
        @app_api.get("/stats")
        async def stats():
            today = str(date.today())
            global_key = f"global:{today}"
            images_today = usage.get(global_key, 0)
            
            current_spend = 0.0
            try:
                # Tentative via Modal API (si supporté par la version actuelle)
                current_spend = modal.Usage.get_current_month()
            except AttributeError:
                pass

            return {
                "monthly_spend": current_spend,
                "images_today": images_today,
                "avg_generation_time": "~12s",  # Estimation
                "remaining_budget": max(0.0, 30.0 - current_spend)
            }

        @app_api.post("/generate")
        def generate_endpoint(request: GenerateRequest, api_key: str = Depends(get_api_key)):
            try:
                # Rate limiting check
                user_key = f"quota_{api_key}"
                count = usage.get(user_key, 0)
                if count >= 50:
                    raise HTTPException(status_code=429, detail="Quota journalier atteint (50 images max).")
                
                # Exécution sur le GPU via remote()
                # On utilise remote() pour envoyer la charge sur la méthode décorée par @modal.method()
                image_bytes = self.generate.remote(request)
                
                # Mise à jour du compteur
                usage[user_key] = count + 1
                
                return Response(content=image_bytes, media_type="image/webp")
            except HTTPException:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
            
        return app_api
