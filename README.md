# Solle Image Generation Toolkit

A comprehensive toolkit for generating, modifying, and interacting with images of "Solle" (a purple furry monster associated with the Solana ecosystem). This project includes local scripts, cloud-based APIs, prompt engineering tools, and a frontend interface.

## 🚀 Features

- **Cloud Image Generation API**: Deployable text-to-image and image-to-image API using Flux.1-schnell and a custom LoRA on Modal.com.
- **Local Image Generation**: Generate images locally using Stable Diffusion XL 1.0 and custom LoRAs.
- **Tweet-to-Prompt Generation**: AI agents (powered by CrewAI and GPT-4o-mini) that analyze tweets and generate optimized image prompts.
- **Comic Builder**: Add dynamic comic-style dialogue bubbles to any generated image.
- **LoRA Training Pipeline**: Scripts and notebooks for training custom LoRAs on Modal, Kaggle, and RunPod.
- **Frontend Interface**: A React/Vite-based web interface for easy interaction.

## 📁 Project Structure

- **`main.py` & `api.py`**: Local/API integration using Gemini and CrewAI to generate images directly from tweet text.
- **`generate_api_modal.py`**: Code to deploy a high-performance, auto-scaling inference API on Modal using Flux and LoRA. (See `API_README.md` for details).
- **`generate_lora.py`**: Local generation script using SDXL and HuggingFace Diffusers.
- **`comic_builder.py`**: Utility to add stylized text bubbles to images.
- **`train_modal.py`**: Script to launch LoRA training runs on Modal A100 GPUs.
- **`prompt_from_tweet/`**: CrewAI pipeline for analyzing tweets and crafting detailed prompts.
- **`src/`**: Core utilities, including system prompts and Gemini API wrappers.
- **`frontend/`**: React + TypeScript + Vite frontend application.
- **`training/`**: Dataset preparation scripts, captioning tools, and Jupyter notebooks for training models.

## 🛠 Getting Started

### Prerequisites

- Python 3.10+
- Node.js (for the frontend)
- API Keys: Modal (for cloud deployment), OpenAI (for CrewAI), Google Gemini (optional, for Gemini generation).

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Set up the Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your API keys (e.g., `OPENAI_API_KEY`, `IMG_GEN_API_KEY`).

4. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   ```

## 📖 Usage

### Local SDXL Generation
```bash
python generate_lora.py "drawn in a comic illustration style, full body wearing an astronaut suit"
```

### Tweet-to-Image Generation
Run the main script to enter a tweet and generate a corresponding image:
```bash
python main.py
```

### Deploying the Modal API
To deploy the Flux inference API to Modal:
```bash
modal deploy generate_api_modal.py
```
*For detailed API usage instructions, please refer to the `API_README.md`.*

### Comic Builder
You can add comic bubbles to your generated images:
```bash
python comic_builder.py
```

## 📝 License
[Specify License Here]
