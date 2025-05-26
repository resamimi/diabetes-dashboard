import React from 'react';
import { Loader2 } from 'lucide-react';

const LoadingMessage = () => {
  return (
    <div className="chat-message left">
      <div className="message-bubble flex items-center gap-3">
        <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
        <span data-translate="true" className="text-sm text-gray-600">
          Analyzing your question and preparing a thoughtful response...
        </span>
      </div>
    </div>
  );
};

export default LoadingMessage;