import React, { useState, useEffect } from 'react';
import Dashboard from './Dashboard';
import AuthPage from './components/AuthPage';
import { TranslationProvider } from './context/TranslationProvider';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check for stored user on mount
  useEffect(() => {
    const userId = localStorage.getItem('userId');
    const username = localStorage.getItem('username');
    
    if (userId && username) {
      setUser({ 
        id: userId,
        username: username 
      });
    }
    setLoading(false);
  }, []);

  const handleAuthComplete = (userData) => {
    console.log('Auth completed with user data:', userData);
    setUser(userData);
  };

  const handleSignOut = () => {
    localStorage.removeItem('userId');
    localStorage.removeItem('username');
    setUser(null);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (!user) {
    return <AuthPage onAuthComplete={handleAuthComplete} />;
  }

  return (
    <TranslationProvider>
      <Dashboard 
        currentUserId={user.id} 
        username={user.username}
        onSignOut={handleSignOut}
      />
    </TranslationProvider>
  );
}

export default App;