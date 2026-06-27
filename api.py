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
        img_data = generated_images[0]
        image = Image.open(io.BytesIO(img_data))

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération : {str(e)}")
