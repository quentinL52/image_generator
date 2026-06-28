import React, { useState } from 'react';
import { Send, Loader2, Download, Sliders, Image as ImageIcon, LayoutTemplate } from 'lucide-react';

interface GeneratorProps {
  apiKey: string;
}

export const Generator: React.FC<GeneratorProps> = ({ apiKey }) => {
  const [prompt, setPrompt] = useState('');
  const [loraScale, setLoraScale] = useState<number>(0.85);
  const [stylePreset, setStylePreset] = useState<string>('none');
  const [aspectRatio, setAspectRatio] = useState<string>('square');
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
      if (stylePreset === 'realistic') {
        finalPrompt = `perfectly sharp background, crisp details throughout the entire image, photo of ${prompt}, wide-angle shot, f/11 aperture, no bokeh, clear view`;
      } else if (stylePreset === 'cartoon') {
        finalPrompt += ", 3d animated movie style, cartoon, flat shading, clear background, vibrant colors, illustration";
      }

      let width = 1024;
      let height = 1024;
      if (aspectRatio === 'landscape') {
        width = 1280;
        height = 768;
      } else if (aspectRatio === 'portrait') {
        width = 768;
        height = 1280;
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
          width: width,
          height: height,
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
                <LayoutTemplate size={16} />
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  Format :
                  <select 
                    value={aspectRatio} 
                    onChange={(e) => setAspectRatio(e.target.value)}
                    disabled={isGenerating}
                    style={{ 
                      background: 'rgba(0,0,0,0.4)', 
                      color: 'var(--color-text)', 
                      border: '1px solid var(--glass-border)', 
                      borderRadius: '8px', 
                      padding: '4px 8px',
                      outline: 'none',
                      fontFamily: 'inherit',
                      cursor: 'pointer'
                    }}
                  >
                    <option value="square">Carré (1024x1024)</option>
                    <option value="landscape">Paysage (1280x768)</option>
                    <option value="portrait">Portrait (768x1280)</option>
                  </select>
                </label>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <ImageIcon size={16} />
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  Style :
                  <select 
                    value={stylePreset} 
                    onChange={(e) => setStylePreset(e.target.value)}
                    disabled={isGenerating}
                    style={{ 
                      background: 'rgba(0,0,0,0.4)', 
                      color: 'var(--color-text)', 
                      border: '1px solid var(--glass-border)', 
                      borderRadius: '8px', 
                      padding: '4px 8px',
                      outline: 'none',
                      fontFamily: 'inherit',
                      cursor: 'pointer'
                    }}
                  >
                    <option value="none">Aucun (Défaut)</option>
                    <option value="cartoon">100% Cartoon (Original)</option>
                    <option value="realistic">Photoréaliste (Fond net)</option>
                  </select>
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
