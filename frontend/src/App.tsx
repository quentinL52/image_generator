import { useState } from 'react'
import { Login } from './Login'
import { Generator } from './Generator'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  // Store the password as the API key to pass to the backend
  const apiKey = 'SOLLE2026';

  return (
    <>
      {!isAuthenticated ? (
        <Login onLogin={() => setIsAuthenticated(true)} />
      ) : (
        <Generator apiKey={apiKey} />
      )}
    </>
  )
}

export default App
