"""
Génère une image avec le LoRA "solle" (sollechar) entraîné sur
**Stable Diffusion XL 1.0 Base** (XL Lora Trainer Hollowstrawberry,
captions en LANGAGE NATUREL).

  - Trigger word   : sollechar  (toujours en tête du prompt)
  - Modèle de base : Stable Diffusion XL 1.0 Base  (même base que l'entraînement)

Usage :
    python generate_lora.py "drawn in a comic illustration style, full body wearing an astronaut suit, floating in space with the earth behind"
    python generate_lora.py "<scene>" 0.8        # 2e arg optionnel = LoRA scale

Le script ajoute automatiquement le trigger + l'ancre d'identité (les mêmes
qu'à l'entraînement), puis ta description de scène en langage naturel.
Optimisé pour 8 Go de VRAM (offload CPU + fp16 + VAE slicing/tiling).

⚠️ SDXL 1.0 Base : la VAE par défaut donne des images NOIRES en fp16 -> on
charge la VAE corrigée `madebyollin/sdxl-vae-fp16-fix`.
"""

import os
import sys
import glob
import random
from datetime import datetime

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,      # DPM++ 2M Karras
    DPMSolverSinglestepScheduler,     # DPM++ SDE Karras
)

# --- Modèle de base : tous compatibles avec ton LoRA (architecture SDXL) ------
# Change MODEL pour tester. "sdxl" = base d'origine de l'entraînement ;
# les autres = finetunes (meilleure cohérence mains/objets/anatomie, gratuit).
# Chaque modèle se télécharge 1 fois (~6,9 Go) puis est mis en cache HF.
MODEL = "juggernaut"                 # "sdxl" | "juggernaut" | "realvis" | "dreamshaper"
MODELS = {
    "sdxl":        "stabilityai/stable-diffusion-xl-base-1.0",
    "juggernaut":  "RunDiffusion/Juggernaut-XL-v9",   # réalisme + cohérence (reco)
    "realvis":     "SG161222/RealVisXL_V5.0",          # photoréalisme poussé
    "dreamshaper": "Lykon/dreamshaper-xl-1-0",         # polyvalent cartoon/réel
}
BASE_REPO = MODELS[MODEL]            # repo diffusers (download Xet-aware, pas d'aria2c)
# Optionnel : un .safetensors local (ex. Civitai). Si défini -> prioritaire sur MODEL.
BASE_LOCAL = None
# VAE corrigée fp16 (évite les images noires en demi-précision).
VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"

# --- LoRA --------------------------------------------------------------------
# Mets ici le nom EXACT du fichier produit par l'entraînement (souvent
# solle_sdxl.safetensors ou un epoch précis solle_sdxl-08.safetensors).
LORA_PATH = "solle_img-10.safetensors"
LORA_SCALE = 0.9            # 0.8–1.0 : LoRA entraîné sur la MÊME base -> pas besoin de forcer

# --- Prompt ------------------------------------------------------------------
# PREFIX LONG = baseline qui marchait (ancre identité, perso bien tenu).
# Pour une scène réaliste TRÈS longue, tu peux raccourcir à
# "sollechar, purple furry monster, mismatched yellow eyes" (gagne des tokens).
PREFIX = ("sollechar, a purple furry monster with shaggy fur, two big round "
          "mismatched eyes with yellow irises, and large pink lips")

# Negatif allégé (SDXL base, pas de tags booru). On garde l'anti-watermark/sparkle
# par sécurité (l'ancien dataset Gemini avait un watermark losange).
NEGATIVE = ("low quality, worst quality, blurry, jpeg artifacts, "
            "deformed, bad anatomy, extra limbs, extra fingers, mutated, "
            "malformed feet, fused fingers, deformed hands, deformed feet, "
            "floating object, broken umbrella, disconnected object, "
            "watermark, signature, sparkle, diamond shape, star symbol, logo, "
            "text, gibberish, unreadable, duplicate tools, holding two tools, solid yellow eyes")

# Le LoRA est entraîné à ~90% sur du comic -> il tire tout le rendu vers le cartoon.
# Ces termes sont AJOUTÉS au negatif uniquement si le prompt demande du réaliste
# (mot "realistic"/"photo"), pour laisser SDXL base ramener un fond réel.
# (On ne les met pas par défaut, sinon ça casserait les rendus comic voulus.)
ANTI_CARTOON = ("drawing, anime, 2d, "
                "flat colors, cel shading, sketch, painterly")

STEPS = 40                 # baseline qui marchait
GUIDANCE = 4                # baseline (CFG 7 = sur-sature, fait baver le violet en néon)
GUIDANCE_RESCALE = 0.5      # OFF = baseline
FREEU = False               # OFF = baseline (FreeU = cause n°1 de la teinte néon)
WIDTH, HEIGHT = 1024, 1024   # portrait ; mets 1024x1024 pour un carré
SEED = None                 # None = aléatoire ; un entier pour reproduire un rendu précis
NUM_IMAGES = 1              # nb d'images par run (seeds différents) -> on cueille la meilleure
OUTPUT_DIR = "generated_images"

DEFAULT_BODY = ("drawn in a comic illustration style, full-body shot standing in a "
                "sunny city park with green trees and a blue sky")


def _resolve_lora() -> str:
    """Renvoie le chemin du LoRA ou explique quoi mettre s'il est introuvable."""
    if os.path.isfile(LORA_PATH):
        return LORA_PATH
    found = sorted(glob.glob("*.safetensors"))
    found = [f for f in found if "sd_xl_base" not in f]   # exclut le modèle de base
    raise SystemExit(
        f"[X] LoRA introuvable : {LORA_PATH}\n"
        f"    Edite LORA_PATH en haut du script. .safetensors detectes ici : "
        f"{found or 'aucun'}"
    )


def main():
    body = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BODY
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else LORA_SCALE
    n_images = int(sys.argv[3]) if len(sys.argv) > 3 else NUM_IMAGES
    lora = _resolve_lora()
    prompt = f"{PREFIX}, {body}"

    # Mode réaliste : si le prompt parle de réel/photo, on pousse l'anti-cartoon.
    negative = NEGATIVE
    if any(k in body.lower() for k in ("realistic", "photo", "photoreal", "realism")):
        negative = f"{NEGATIVE}, {ANTI_CARTOON}"
        print("[i] mode réaliste détecté -> anti-cartoon ajouté au negatif "
              "(pense aussi à baisser le scale, ex. 0.65)")

    print(f"[i] prompt complet : {prompt}")

    # 1) VAE corrigée fp16 (indispensable pour SDXL base)
    print(f"[1/4] VAE fp16-fix : {VAE_REPO}")
    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=torch.float16)

    # 2) Modèle de base (checkpoint SDXL choisi via MODEL)
    if BASE_LOCAL and os.path.isfile(BASE_LOCAL):
        print(f"[2/4] Modèle (local) : {BASE_LOCAL}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            BASE_LOCAL, torch_dtype=torch.float16, use_safetensors=True, vae=vae,
        )
    else:
        print(f"[2/4] Modèle '{MODEL}' (HF diffusers) : {BASE_REPO}")
        # Certains finetunes ne publient QUE la variante fp16 (ex. Juggernaut),
        # d'autres les poids "plein" -> on tente fp16, puis on retombe sur None.
        pipe = None
        for variant in ("fp16", None):
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    BASE_REPO, torch_dtype=torch.float16, use_safetensors=True,
                    vae=vae, variant=variant,
                )
                break
            except (OSError, ValueError):
                continue
        if pipe is None:
            raise SystemExit(f"[X] Impossible de charger {BASE_REPO} (ni fp16 ni plein).")

    # DPM++ 2M Karras : rendu doux = baseline qui marchait (palette naturelle).
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++",
    )

    # 3) LoRA
    print(f"[3/4] LoRA : {lora}  (scale={scale})")
    pipe.load_lora_weights(lora, adapter_name="solle")
    pipe.set_adapters(["solle"], adapter_weights=[scale])

    # FreeU : rééquilibre les skip-connections de l'UNet (valeurs SDXL reco)
    if FREEU:
        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        print("[i] FreeU activé")

    # 8 Go VRAM : offload CPU + VAE slicing/tiling
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4) Génération en lot : N seeds différents, on garde la meilleure
    print(f"[4/4] Génération de {n_images} image(s)…")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(n_images):
        seed = SEED if SEED is not None else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            width=WIDTH,
            height=HEIGHT,
            generator=generator,
        ).images[0]
        # seed dans le nom -> pour reproduire/retoucher la bonne image
        out_path = os.path.join(OUTPUT_DIR, f"solle_{stamp}_seed{seed}.png")
        image.save(out_path)
        print(f"  [{i+1}/{n_images}] {out_path}")

    print(f"\n[OK] {n_images} image(s) dans {OUTPUT_DIR}/ — choisis la plus propre.")


if __name__ == "__main__":
    main()
