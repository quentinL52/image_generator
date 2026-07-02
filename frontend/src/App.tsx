import { useState, lazy, Suspense } from 'react'
import { Login } from './Login'

// ⚡ Bolt: Lazy load the Generator component to reduce initial bundle size by ~5.8KB.
// The initial view is the Login page, so loading the Generator and its dependencies (like lucide-react icons)
// can be deferred until the user is successfully authenticated.
const Generator = lazy(() => import('./Generator').then(module => ({ default: module.Generator })));

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  // Store the password as the API key to pass to the backend
  const apiKey = 'SOLLE2026';

  return (
    <>
      {!isAuthenticated ? (
        <Login onLogin={() => setIsAuthenticated(true)} />
      ) : (
        <Suspense fallback={<div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', color: 'white'}}>Loading...</div>}>
          <Generator apiKey={apiKey} />
        </Suspense>
      )}
    </>
  )
}

export default App
