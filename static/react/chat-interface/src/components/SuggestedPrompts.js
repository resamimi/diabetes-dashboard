import React from 'react';

const SuggestedPrompts = ({ questions, onQuestionClick }) => {
  if (!questions || questions.length === 0) return null;

  return (
    <div className="w-full max-w-[80%] mx-auto my-2 p-3 bg-gray-50 rounded-lg border border-gray-100 shadow-sm">
      <div className="text-xs font-medium text-gray-500 mb-2">
        Suggested questions:
      </div>
      <div className="space-y-1">
        {questions.map((question, index) => (
          <button
            key={index}
            onClick={() => onQuestionClick(question)}
            className="w-full px-2.5 py-1.5 text-left text-sm text-gray-700 
                     hover:bg-white rounded-md transition-colors duration-200
                     border border-transparent hover:border-gray-200
                     hover:shadow-sm active:scale-[0.99]"
            data-translate="true"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
};

export default SuggestedPrompts;