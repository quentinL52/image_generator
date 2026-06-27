from src.solle_img_generation import imggenrator
from src.config import save_image
from prompt_from_tweet.main import generate_image_prompt_from_tweet

if __name__ == "__main__":
    input_image_path = "solle_base.jpg"
    solana_logo_path = "solana_logo.png" 
    user_text_input = input("your tweet here")
    prompt_img = generate_image_prompt_from_tweet(user_text_input)
    model = imggenrator()
    generated_images = model.generate(input_image_path, solana_logo_path, prompt_img)
    
    if generated_images:
        for i, img_data in enumerate(generated_images):

            save_image(user_text_input, f"image_{i}.png", img_data)
        
    else:
        print("Aucune image générée.")