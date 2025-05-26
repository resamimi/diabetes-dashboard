import React from 'react';
import { Alert, AlertTitle, AlertDescription } from './ui/alert';
import { X } from 'lucide-react';

const ErrorPopup = ({ message, onClose }) => {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/50 z-50">
      <div className="relative w-full max-w-md mx-4">
        <Alert className="bg-white border-red-200 shadow-lg">
          <AlertTitle className="text-red-600 flex items-center justify-between">
            Error
            <button 
              onClick={onClose}
              className="p-1 hover:bg-red-50 rounded-full transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </AlertTitle>
          <AlertDescription className="text-gray-600 mt-2">
            {message}
          </AlertDescription>
        </Alert>
      </div>
    </div>
  );
};

export default ErrorPopup;