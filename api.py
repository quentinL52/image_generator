from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from src.solle_img_generation import imggenrator
from prompt_from_tweet.main import generate_image_prompt_from_tweet
import io
from PIL import Image
import base64

app = FastAPI(title="Solle Image Generation API")

INPUT_IMAGE_PATH = "solle_base.jpg"
SOLANA_LOGO_PATH = "solana_logo.png"


class TweetRequest(BaseModel):
    tweet: str

@app.post("/generate-image")
def generate_image(request: TweetRequest):
    tweet_text = request.tweet.strip()
    if not tweet_text:
        raise HTTPException(status_code=400, detail="Le tweet ne peut pas être vide.")

    try:
        prompt_img = generate_image_prompt_from_tweet(tweet_text)
        model = imggenrator()
        generated_images = model.generate(INPUT_IMAGE_PATH, SOLANA_LOGO_PATH, prompt_img)

        if not generated_images:
            raise HTTPException(status_code=500, detail="Aucune image générée.")

        # ⚡ Bolt: Return the generated raw bytes directly to save decoding/re-encoding time (~40ms) and memory.
        img_data = generated_images[0]

        # Dynamically determine the image format based on magic bytes
        media_type = "image/jpeg"
        if img_data.startswith(b'\x89PNG\r\n\x1a\n'):
            media_type = "image/png"
        elif img_data.startswith(b'\xff\xd8\xff'):
            media_type = "image/jpeg"
        elif img_data.startswith(b'RIFF') and img_data[8:12] == b'WEBP':
            media_type = "image/webp"

        return Response(content=img_data, media_type=media_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération : {str(e)}")
