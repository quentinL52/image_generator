import React, { useState } from 'react';
import { Send, Loader2, Download, Sliders, Image as ImageIcon } from 'lucide-react';

interface GeneratorProps {
  apiKey: string;
}

export const Generator: React.FC<GeneratorProps> = ({ apiKey }) => {
  const [prompt, setPrompt] = useState('');
  const [loraScale, setLoraScale] = useState<number>(0.85);
  const [sharpBackground, setSharpBackground] = useState<boolean>(false);
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
      let finalPrompt = prompt;
      if (sharpBackground) {
        finalPrompt += ", deep depth of field, f/16, sharp background, everything in focus, wide angle lens";
      }

      // 1. Submit Job
      const submitResponse = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey
        },
        body: JSON.stringify({
          prompt: finalPrompt,
          width: 1024,
          height: 1024,
          lora_scale: loraScale
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
            
            <div className="settings-wrapper" style={{ marginTop: '0.5rem', padding: '0 0.5rem', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '1.5rem', color: 'var(--color-text-dim)', fontSize: '0.9rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: '1 1 200px' }}>
                <Sliders size={16} />
                <span>Force (LoRA) : {loraScale.toFixed(2)}</span>
                <input 
                  type="range" 
                  min="0.1" 
                  max="1.2" 
                  step="0.05" 
                  value={loraScale} 
                  onChange={(e) => setLoraScale(parseFloat(e.target.value))}
                  style={{ flex: 1, accentColor: 'var(--color-primary)' }}
                  disabled={isGenerating}
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <ImageIcon size={16} />
                <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input 
                    type="checkbox" 
                    checked={sharpBackground}
                    onChange={(e) => setSharpBackground(e.target.checked)}
                    disabled={isGenerating}
                    style={{ accentColor: 'var(--color-primary)', width: '16px', height: '16px', cursor: 'pointer' }}
                  />
                  Forcer le fond net (Paysage)
                </label>
              </div>
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
