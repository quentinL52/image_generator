import json


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s}


cells = []

cells.append(md(
"""# Entraînement LoRA FLUX.1 Schnell — Solle — **RunPod (single GPU)**

Version **RunPod** : un seul gros GPU (≥24 Go, ex. RTX 4090 / A100 / RTX PRO 6000 96 Go).
**Aucun patch, aucun `split_model_over_gpus`** — FLUX tient sur une carte, ai-toolkit actuel marche direct.

Config « qualité » (exploite la grosse VRAM) : **bf16 sans quantisation, résolution 1024, batch 4, 1500 steps**.

## Prérequis
1. **Template PyTorch ≥ 2.7 / CUDA ≥ 12.8** (obligatoire si carte Blackwell type PRO 6000).
2. **Token HF** (Read) : accepte d'abord la licence https://huggingface.co/black-forest-labs/FLUX.1-schnell
   puis définis la variable d'env `HF_TOKEN` sur le pod, ou colle-le dans la cellule 3.
3. **Dataset** : uploade tes **79 PNG + 79 TXT** dans `/workspace/train_data` (glisser-déposer Jupyter).

Durée estimée : ~1h pour 1500 steps sur un GPU récent. Lance et laisse tourner."""))

cells.append(md("## 1. Vérifier le GPU"))
cells.append(code("!nvidia-smi"))

cells.append(md("## 2. Installer ai-toolkit (version ACTUELLE, sans patch)"))
cells.append(code(
"""%cd /workspace
!git clone https://github.com/ostris/ai-toolkit
%cd ai-toolkit
!git submodule update --init --recursive
!pip install -q -r requirements.txt
!pip install -q bitsandbytes>=0.43.0
print('Installation terminée.')"""))

cells.append(md("## 3. Connexion HuggingFace — OBLIGATOIRE (Schnell est gated)"))
cells.append(code(
"""import os
from huggingface_hub import login

token = os.environ.get('HF_TOKEN') or 'hf_PASTE_TON_TOKEN_ICI'
if token.startswith('hf_PASTE'):
    raise RuntimeError(
        "Token HF manquant. Accepte la licence FLUX.1-schnell, crée un token Read, "
        "puis définis HF_TOKEN sur le pod OU colle-le ici."
    )
login(token=token)
print('Connecté à HuggingFace.')"""))

cells.append(md(
"""## 4. Vérifier le dataset

Uploade tes 79 PNG + 79 TXT dans **`/workspace/train_data`** (même nom de base, ex. `img1.png` + `img1.txt`)
avant de lancer cette cellule."""))
cells.append(code(
"""import glob
DST = '/workspace/train_data'
imgs = len(glob.glob(DST+'/*.png')) + len(glob.glob(DST+'/*.jpg')) + len(glob.glob(DST+'/*.jpeg'))
txts = len(glob.glob(DST+'/*.txt'))
print(f'images={imgs} | captions={txts}')
assert imgs > 0, f'AUCUNE image dans {DST} — uploade tes PNG+TXT dedans.'
assert imgs == txts, f'Mismatch ! {imgs} images vs {txts} captions'
print('Exemple caption :', open(sorted(glob.glob(DST+'/*.txt'))[0]).read())"""))

cells.append(md(
"""## 5. Configuration (single GPU, qualité 96 Go)

- **PAS de split, PAS de quantisation** (`quantize: False`, full bf16)
- `resolution: [1024]`, `batch_size: 4` → exploite la grosse VRAM
- `steps: 1500`, `save_every: 250` (checkpoints réguliers de secours)
- `gradient_checkpointing: True` (sûr ; passe à `False` pour + de vitesse si pas d'OOM)"""))
cells.append(code(
"""import os, yaml

config = {
    'job': 'extension',
    'config': {
        'name': 'solle_flux',
        'process': [{
            'type': 'sd_trainer',
            'training_folder': '/workspace/output',
            'performance_log_every': 100,
            'device': 'cuda:0',

            'network': {'type': 'lora', 'linear': 16, 'linear_alpha': 16},

            'save': {'dtype': 'float16', 'save_every': 250, 'max_step_saves_to_keep': 6},

            'datasets': [{
                'folder_path': '/workspace/train_data',
                'caption_ext': '.txt',
                'caption_dropout_rate': 0.0,
                'shuffle_tokens': False,
                'cache_latents_to_disk': True,
                'resolution': [1024],
            }],

            'train': {
                'batch_size': 4,
                'steps': 1500,
                'gradient_accumulation_steps': 1,
                'train_unet': True,
                'train_text_encoder': False,
                'gradient_checkpointing': True,
                'noise_scheduler': 'flowmatch',
                'optimizer': 'adamw8bit',
                'lr': 1e-4,
                'dtype': 'bf16',
            },

            'model': {
                'name_or_path': 'black-forest-labs/FLUX.1-schnell',
                'assistant_lora_path': 'ostris/FLUX.1-schnell-training-adapter',
                'is_flux': True,
                'quantize': False,
            },

            'sample': {
                'sampler': 'flowmatch',
                'sample_every': 250,
                'width': 1024, 'height': 1024,
                'prompts': [
                    'sollechar, comic illustration style, full body, standing, looking at viewer',
                    'sollechar standing in London, red phone booth, holding a sign that says BUY SOL, photorealistic background',
                    'sollechar, cartoon style, skateboarding in New York, dynamic pose, urban street',
                ],
                'neg': '',
                'seed': 42, 'walk_seed': True,
                'guidance_scale': 1,
                'sample_steps': 4,
            },
        }],
    },
    'meta': {'name': '[name]', 'version': '1.0'},
}

os.makedirs('/workspace/configs', exist_ok=True)
cfg_path = '/workspace/configs/solle_flux.yaml'
with open(cfg_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
print('Config écrite :')
print(open(cfg_path).read())"""))

cells.append(md(
"""## 6. Lancer l'entraînement

Logs à surveiller : `loss` (bruité, c'est normal pour FLUX — juge aux **samples**) et les images
auto toutes les 250 steps dans `output/.../samples/` (regarde surtout le prompt « BUY SOL »)."""))
cells.append(code(
"""%cd /workspace/ai-toolkit
!python run.py /workspace/configs/solle_flux.yaml"""))

cells.append(md(
"""## 7. Récupérer le LoRA

Fichiers dans `/workspace/output/`. Télécharge `solle_flux.safetensors` (final, step 1500)
+ un checkpoint intermédiaire en backup. Juge les samples pour choisir le meilleur step."""))
cells.append(code(
"""import os, glob
output_dir = '/workspace/output'
loras = sorted(glob.glob(f'{output_dir}/**/*.safetensors', recursive=True))
samples = sorted(glob.glob(f'{output_dir}/**/samples/*.png', recursive=True))
print(f'LoRA ({len(loras)}) :')
for f in loras:
    print(f'  {os.path.basename(f)} — {os.path.getsize(f)/1024/1024:.1f} Mo')
print(f'\\nSamples ({len(samples)}), 6 derniers :')
for f in samples[-6:]:
    print(' ', f)"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open('runpod_train_solle_flux.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

again = json.load(open('runpod_train_solle_flux.ipynb', encoding='utf-8'))
print('OK notebook ecrit :', len(again['cells']), 'cellules')
