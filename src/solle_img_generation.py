import os
import mimetypes
from google import genai
from google.genai import types
from .config import load_prompt_template

class imggenrator:
    def __init__(self):
        self.api_key = os.getenv("IMG_GEN_API_KEY")
        self.model = os.getenv("model")
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, base_image_path: str, solana_logo_path: str, user_input_text: str):
        contents_parts = []
        with open(base_image_path, "rb") as f:
            image_data = f.read()
        image_mime_type = mimetypes.guess_type(base_image_path)[0]
        contents_parts.append(types.Part.from_bytes(data=image_data, mime_type=image_mime_type))
        if solana_logo_path:
            with open(solana_logo_path, "rb") as f:
                image_data = f.read()
            image_mime_type = mimetypes.guess_type(solana_logo_path)[0]
            contents_parts.append(types.Part.from_bytes(data=image_data, mime_type=image_mime_type))

        full_prompt = load_prompt_template(base_image_path, solana_logo_path, user_input_text)
        contents_parts.append(types.Part.from_text(text=full_prompt))

        contents = [
            types.Content(
                role="user",
                parts=contents_parts,
            ),
        ]
        config = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
        generated_images_data = []

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        generated_images_data.append(part.inline_data.data)
                    elif hasattr(part, "text") and part.text:
                        print(part.text)
        return generated_images_data