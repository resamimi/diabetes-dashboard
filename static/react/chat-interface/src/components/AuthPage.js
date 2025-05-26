import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Alert, AlertDescription } from './ui/alert';
import { Info } from 'lucide-react';

const AuthPage = ({ onAuthComplete }) => {
  // Start with signup mode by default
  const [isSignIn, setIsSignIn] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    const endpoint = isSignIn ? 'signin' : 'signup';

    try {
      const response = await fetch(`/api/auth/${endpoint}`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ username, password }),
      });
      
      let data;
      try {
        const textResponse = await response.text();
        data = JSON.parse(textResponse);
      } catch (parseError) {
        throw new Error('Invalid response from server');
      }

      if (!response.ok) {
        throw new Error(data.error || 'Authentication failed');
      }

      localStorage.setItem('userId', data.id);
      localStorage.setItem('username', data.username);

      onAuthComplete({
        id: data.id,
        username: data.username
      });
      
    } catch (err) {
      setError(err.message || 'Failed to authenticate. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 flex items-center justify-center p-6">
      <div className="w-full max-w-md space-y-4">
        <Alert className="bg-blue-50 border-blue-200">
          <div className="flex items-center gap-2">
            <Info className="h-4 w-4 text-blue-500" />
            <AlertDescription className="text-blue-800">
              Please sign up with your Prolific ID. You can choose any password you wish.
            </AlertDescription>
          </div>
        </Alert>

        <Card className="backdrop-blur-sm bg-white/95">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl font-bold text-center">
              {isSignIn ? 'Welcome Back' : 'Create Account'}
            </CardTitle>
            <p className="text-sm text-center text-gray-600">
              {isSignIn ? 'Sign in to continue to the dashboard' : 'Sign up to get started'}
            </p>
          </CardHeader>
          
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="p-3 text-sm text-red-600 bg-red-50 rounded-lg border border-red-100">
                  {error}
                </div>
              )}
              
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">
                  {isSignIn ? 'Username (Prolific ID)' : 'Prolific ID'}
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full px-3 py-2 text-gray-700 border rounded-lg 
                           focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                           bg-white/50 backdrop-blur-sm transition-all duration-200"
                  placeholder={isSignIn ? "Enter your Prolific ID" : "Enter your Prolific ID"}
                  required
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3 py-2 text-gray-700 border rounded-lg 
                           focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                           bg-white/50 backdrop-blur-sm transition-all duration-200"
                  placeholder="Choose a password"
                  required
                />
              </div>

              <button
                type="submit"
                className="w-full py-2.5 text-white bg-gradient-to-r from-blue-500 to-blue-600
                         rounded-lg hover:from-blue-600 hover:to-blue-700
                         focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                         transition-all duration-200 font-medium"
              >
                {isSignIn ? 'Sign In' : 'Sign Up'}
              </button>

              <p className="text-sm text-center text-gray-600">
                {isSignIn ? "Don't have an account? " : "Already have an account? "}
                <button
                  type="button"
                  onClick={() => {
                    setIsSignIn(!isSignIn);
                    setError('');
                    setUsername('');
                    setPassword('');
                  }}
                  className="text-blue-500 hover:text-blue-700 font-medium focus:outline-none"
                >
                  {isSignIn ? 'Sign Up' : 'Sign In'}
                </button>
              </p>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AuthPage;