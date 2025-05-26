import React, { useEffect, useRef } from 'react';
import { LineChart } from 'lucide-react';

const ChatMessage = React.forwardRef(({ 
  message, 
  onFeedback, 
  onWrongAnswer, 
  onVisualization,
  highlightFeature,
  messageIndex,
  totalMessages,
  currentVisualizationType,
  className,
  showFeedback = true
}, ref) => {
  const messageRef = useRef(null);
  const timeoutRef = useRef(null);
  
  const getVisualizationData = () => {
    if (!message?.figureData?.data || !message?.figureData?.type) {
      return null;
    }
    
    try {
      const visualizationData = {
        type: message.figureData.type,
        data: message.figureData.data
      };
      return visualizationData;
    } catch (error) {
      console.error('Error processing visualization data:', error);
      return null;
    }
  };

  const visualizationData = getVisualizationData();

  useEffect(() => {
    if (visualizationData) {
      onVisualization(visualizationData);
    }
  }, []);

  // Clear previous timeout if component updates
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const findFeatureSection = (headers, featureText) => {
    const allHeaders = Array.from(headers);
    for (let i = 0; i < allHeaders.length; i++) {
      const header = allHeaders[i];
      const headerText = header.textContent.toLowerCase();
      
      if (headerText.includes(featureText.toLowerCase())) {
        const content = [];
        let currentElement = header.nextElementSibling;
        const nextHeader = allHeaders[i + 1];

        while (currentElement && 
               currentElement !== nextHeader && 
               !currentElement.matches('h1, h2, h3, h4')) {
          if (currentElement.matches('p, ul, ol, div:not(.message-actions):not(.message-content)')) {
            content.push(currentElement);
          }
          currentElement = currentElement.nextElementSibling;
        }

        return { header, content };
      }
    }
    return null;
  };

  // Handle highlighting and cleanup
  useEffect(() => {
    if (!messageRef.current) return;

    // Clear any existing highlights in this message
    const clearHighlights = () => {
      if (messageRef.current) {
        messageRef.current.querySelectorAll('.highlight-feature').forEach(el => {
          el.classList.remove('highlight-feature', 'bg-yellow-100');
        });
      }
    };

    // If this is not the latest message or visualization types don't match, clear highlights
    if (messageIndex !== totalMessages - 1 || 
        (message?.figureData?.type && message.figureData.type !== currentVisualizationType)) {
      clearHighlights();
      return;
    }

    if (highlightFeature && messageRef.current) {
      clearHighlights();

      const headers = messageRef.current.querySelectorAll('h1, h2, h3, h4');
      const section = findFeatureSection(headers, highlightFeature);

      if (section) {
        const { header, content } = section;

        // Add highlight classes
        header.classList.add('highlight-feature', 'bg-yellow-100');
        content.forEach(el => {
          el.classList.add('highlight-feature', 'bg-yellow-100');
        });

        // Scroll the header into view
        header.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center'
        });
      }
    }
  }, [highlightFeature, messageIndex, totalMessages, currentVisualizationType]);

  return (
    <div ref={messageRef} className={`message ${message?.side || ''} ${className || ''}`}>
      <div className="message-content">
        {message?.text && (
          <div 
            className="prose prose-blue max-w-none"
            dangerouslySetInnerHTML={{ __html: message.text }} 
          />
        )}
        
        {visualizationData && (
        <div className="mt-6 flex justify-center">
          <button
            onClick={() => onVisualization(visualizationData)}
            className="inline-flex items-center gap-2 px-6 py-2.5 
                      bg-white border border-gray-200 rounded-full
                      shadow-sm hover:shadow-md
                      text-sm font-medium text-gray-700
                      hover:bg-gray-50 hover:border-gray-300
                      transition-all duration-200 ease-in-out
                      group"
          >
            <LineChart className="w-4 h-4 text-blue-500 group-hover:text-blue-600" />
            <span>
              {visualizationData.type === "classification_scatter" ? "View Classification Plot" :
              visualizationData.type === "feature_importance" ? "View Feature Importance" :
              visualizationData.type === "feature_range" ? "View Feature Range" :
              visualizationData.type === "typical_mistakes" ? "View Error Patterns" :
              visualizationData.type === "counterfactual_explanation" ? "View Recommendations" :
              "View Visualization"}
            </span>
          </button>
        </div>
      )}
      </div>
      
      {showFeedback && message?.side === 'left' && (
        <div className="message-actions mt-2">
          <button onClick={() => onFeedback(message.id, 'Positive')} title="Positive">👍</button>
          <button onClick={() => onFeedback(message.id, 'Negative')} title="Negative">👎</button>
          <button onClick={() => onFeedback(message.id, 'Insightful')} title="Insightful">💡</button>
          <button onClick={() => onFeedback(message.id, 'Confused')} title="Confused">😕</button>
          <button onClick={() => onWrongAnswer(message.id)} title="Wrong Answer">❌</button>
        </div>
      )}
    </div>
  );
});

export default ChatMessage;