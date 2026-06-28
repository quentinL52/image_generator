import React, { useState } from 'react';
import { Send, Loader2, Download } from 'lucide-react';

interface GeneratorProps {
  apiKey: string;
}

export const Generator: React.FC<GeneratorProps> = ({ apiKey }) => {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationStatus, setGenerationStatus] = useState('');
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const API_BASE_URL = 'https://contact-4061--flux-solle-api-apiserver-web.modal.run';

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setIsGenerating(true);
    setGenerationStatus('Démarrage...');
    setError(null);
    setResultImage(null);

    try {
      // 1. Submit Job
      const submitResponse = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey
        },
        body: JSON.stringify({
          prompt: prompt,
          width: 1024,
          height: 1024,
          lora_scale: 0.8
        })
      });

      if (!submitResponse.ok) {
        throw new Error(`Erreur de soumission: ${submitResponse.statusText}`);
      }

      const { job_id } = await submitResponse.json();
      
      // 2. Poll Status
      let isDone = false;
      while (!isDone) {
        setGenerationStatus('Génération en cours...');
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const statusResponse = await fetch(`${API_BASE_URL}/status/${job_id}`, {
          headers: { 'X-API-Key': apiKey }
        });
        
        if (!statusResponse.ok) throw new Error('Erreur lors de la vérification du statut');
        
        const statusData = await statusResponse.json();
        
        if (statusData.status === 'completed') {
          isDone = true;
        } else if (statusData.status === 'failed') {
          throw new Error(statusData.error || 'La génération a échoué');
        }
      }

      // 3. Fetch Image
      setGenerationStatus('Téléchargement de l\'image...');
      const imageResponse = await fetch(`${API_BASE_URL}/image/${job_id}`, {
        headers: { 'X-API-Key': apiKey }
      });

      if (!imageResponse.ok) throw new Error('Erreur lors de la récupération de l\'image');

      const blob = await imageResponse.blob();
      const imageUrl = URL.createObjectURL(blob);
      setResultImage(imageUrl);

    } catch (err: any) {
      setError(err.message || "Une erreur est survenue.");
    } finally {
      setIsGenerating(false);
      setGenerationStatus('');
    }
  };

  const handleDownload = () => {
    if (resultImage) {
      const a = document.createElement('a');
      a.href = resultImage;
      a.download = `solle_${Date.now()}.jpg`;
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
              <input 
                type="text" 
                className="prompt-input"
                placeholder="Décrivez votre image (ex: sollechar faisant du skate cyberpunk...)" 
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
            {isGenerating && <div className="status-text">{generationStatus}</div>}
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
