import os
import mimetypes

def read_system_prompt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_prompt_template(base_image_path: str, solana_logo_path: str, user_instruction: str) -> str:
    base_prompt = read_system_prompt('prompt.txt')
    
    image_descriptions = f"Image de référence 1 (personnage): {base_image_path}\n"
    if solana_logo_path:
        image_descriptions += f"Image de référence 2 (logo Solana): {solana_logo_path}\n"
        
    enriched_prompt = base_prompt.format(
        base_image_paths=image_descriptions,
        user_instruction=user_instruction
    )
    return enriched_prompt.strip()
"""
def save_image(file_name: str, data: bytes):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"Fichier sauvegardé : {file_name}")"""

def save_image(user_text, file_name, data):
    sanitized_user_text = user_text[:10].replace(" ", "_").lower()
    new_file_name = f"genere_pour_{sanitized_user_text}_{file_name}"
    file_path = os.path.join("generated_image", new_file_name)
    os.makedirs("generated_image", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(data)
    print(f"Image générée : {file_path}")