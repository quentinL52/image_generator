import React, { useState } from 'react';

interface LoginProps {
  onLogin: () => void;
}

export const Login: React.FC<LoginProps> = ({ onLogin }) => {
  const [password, setPassword] = useState('');
  const [error, setError] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (password === 'SOLLE2026') {
      onLogin();
    } else {
      setError(true);
      setTimeout(() => setError(false), 2000);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h1 className="logo-text">SOLLE</h1>
        <p className="subtitle">Accès Générateur d'Images</p>
        
        <form onSubmit={handleSubmit} className="login-form">
          <input 
            type="password" 
            placeholder="Mot de passe..." 
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className={error ? 'error-input' : ''}
          />
          <button type="submit" className="btn-primary">
            SE CONNECTER
          </button>
        </form>
      </div>

      <div className="bg-images">
        <img src="/solle_cartoon.jpg" alt="Solle Cartoon" className="bg-img cartoon" />
        <img src="/solle_skating.jpg" alt="Solle Skating" className="bg-img skating" />
      </div>
      <div className="overlay"></div>
    </div>
  );
};
