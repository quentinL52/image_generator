import os
import subprocess
import modal

# 1. Configuration de l'environnement Modal
app = modal.App("flux-1-schnell-lora")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "bzip2")
    .pip_install(
        "torch",
        "torchvision", "torchaudio",
        "transformers",
        "diffusers",
        "huggingface_hub",
        "bitsandbytes>=0.43.0",
        "pydantic",
        "pyyaml",
    )
    .run_commands(
        "git clone https://github.com/ostris/ai-toolkit /root/ai-toolkit",
        "cd /root/ai-toolkit && git submodule update --init --recursive",
        "cd /root/ai-toolkit && pip install -r requirements.txt"
    )
)

volume = modal.Volume.from_name("flux-lora-storage")

# 2. Fonction d'entraînement exécutée sur le GPU
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/workspace": volume},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=54000 # Timeout de 1h30
)
def run_training():
    import yaml
    
    os.makedirs("/workspace/output", exist_ok=True)
    os.makedirs("/workspace/configs", exist_ok=True)

    # Configuration de l'entraînement
    config = {
        'job': 'extension',
        'config': {
            'name': 'solle_flux_v2',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': '/workspace/output',
                'performance_log_every': 100,
                'device': 'cuda:0',
                'network': {'type': 'lora', 'linear': 32, 'linear_alpha': 32},
                'save': {'dtype': 'float16', 'save_every': 250, 'max_step_saves_to_keep': 8},
                'datasets': [{
                    'folder_path': '/workspace/train_data',
                    'caption_ext': '.txt',
                    'caption_dropout_rate': 0.05,
                    'shuffle_tokens': False,
                    'cache_latents_to_disk': True,
                    'resolution': [1024],
                }],
                'train': {
                    'batch_size': 2,
                    'steps': 2000,
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
                        'sollechar, purple furry monster, comic style, full body, standing, looking at viewer',
                        'sollechar, purple furry monster, photorealistic style, standing in London, red phone booth, holding a sign that says BUY SOL',
                        'sollechar, purple furry monster, 3d render, skateboarding in New York, dynamic pose, urban street',
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

    cfg_path = '/workspace/configs/solle_flux.yaml'
    with open(cfg_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Connexion HF
    from huggingface_hub import login
    token = os.environ.get('HF_TOKEN')
    if token:
        login(token=token)
    else:
        print("Attention : HF_TOKEN non trouvé.")

    # Lancement de ai-toolkit
    os.chdir("/root/ai-toolkit")
    subprocess.run(["python", "run.py", cfg_path], check=True)

@app.local_entrypoint()
def main():
    print("Déploiement sur Modal en cours...")
    run_training.remote()
