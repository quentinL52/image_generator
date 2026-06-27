import React, { useState, useRef } from 'react';
import { Upload, Send, Loader2, Download } from 'lucide-react';

interface GeneratorProps {
  apiKey: string;
}

export const Generator: React.FC<GeneratorProps> = ({ apiKey }) => {
  const [prompt, setPrompt] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const removeImage = () => {
    setImageFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        let encoded = reader.result as string;
        // Remove data URL prefix (e.g., data:image/png;base64,)
        encoded = encoded.replace(/^data:image\/(png|jpeg|jpg|webp);base64,/, '');
        resolve(encoded);
      };
      reader.onerror = error => reject(error);
    });
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setIsGenerating(true);
    setError(null);
    setResultImage(null);

    try {
      let initImageBase64 = undefined;
      if (imageFile) {
        initImageBase64 = await fileToBase64(imageFile);
      }

      const response = await fetch('https://contact-4061--flux-solle-api-fluxgenerator-web.modal.run/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey
        },
        body: JSON.stringify({
          prompt: prompt,
          init_image: initImageBase64,
          width: 1024,
          height: 1024,
          lora_scale: 0.8,
          strength: 0.6
        })
      });

      if (!response.ok) {
        throw new Error(`Erreur lors de la génération: ${response.statusText}`);
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setResultImage(imageUrl);

    } catch (err: any) {
      setError(err.message || "Une erreur est survenue.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (resultImage) {
      const a = document.createElement('a');
      a.href = resultImage;
      a.download = `solle_${Date.now()}.webp`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  return (
    <div className="generator-container">
      <header className="header">
        <h1 className="logo-text-small">SOLLE</h1>
      </header>

      <main className="main-content">
        <div className="generation-box">
          
          <form onSubmit={handleGenerate} className="prompt-form">
            <div className="input-wrapper">
              <button 
                type="button" 
                className="btn-upload"
                onClick={() => fileInputRef.current?.click()}
                title="Ajouter une image"
              >
                <Upload size={20} />
              </button>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleImageUpload} 
                accept="image/jpeg, image/png, image/webp"
                style={{ display: 'none' }} 
              />
              
              <input 
                type="text" 
                className="prompt-input"
                placeholder="Décrivez votre image (ex: un garçon faisant du skate cyberpunk...)" 
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />

              <button 
                type="submit" 
                className="btn-generate" 
                disabled={isGenerating || !prompt.trim()}
              >
                {isGenerating ? <Loader2 className="spinner" size={20} /> : <Send size={20} />}
              </button>
            </div>
            
            {previewUrl && (
              <div className="image-preview-container">
                <div className="image-preview">
                  <img src={previewUrl} alt="Preview" />
                  <button type="button" className="btn-remove-image" onClick={removeImage}>&times;</button>
                </div>
                <span className="preview-label">Image d'initialisation ajoutée</span>
              </div>
            )}
          </form>

          {error && <div className="error-message">{error}</div>}

          {resultImage && (
            <div className="result-container glass">
              <img src={resultImage} alt="Image générée" className="result-image" />
              <button onClick={handleDownload} className="btn-download">
                <Download size={20} />
                TÉLÉCHARGER
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};
