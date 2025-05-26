import { useState, useEffect, useCallback } from 'react';
import { categories } from '../categories';

export const useChatMessages = (datasetObjective) => {
  const [messages, setMessages] = useState([]);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [lastMessageFeatureDetails, setLastMessageFeatureDetails] = useState(null);

  // Add this new function to handle feature details
  const updateLastMessageWithFeatureDetails = useCallback((feature, report, visualizationType) => {
    console.log("updateLastMessageWithFeatureDetails called with:", {
      feature,
      report: report?.substring(0, 100), // Log just the start of the report to keep console clean
      visualizationType
    });
  
    if (!report) {
      console.log("No report provided");
      return;
    }
  
    // Make the regex more flexible to handle potential whitespace
    const featureRegex = new RegExp(`<h4>\\s*${feature}\\s*</h4><p>.*?</p>`, 's');
    const match = report.match(featureRegex);
    
    console.log("Regex match result:", match ? match[0].substring(0, 100) : null);
  
    if (match) {
      const featureDetails = match[0];
      setLastMessageFeatureDetails({
        feature,
        details: featureDetails,
        visualizationType
      });
      
      setMessages(prevMessages => {
        const newMessages = [...prevMessages];
        const lastBotMessageIndex = newMessages
          .map((msg, index) => msg.side === 'left' ? index : -1)
          .filter(index => index !== -1)
          .pop();
          
        if (lastBotMessageIndex !== undefined) {
          const baseMessage = newMessages[lastBotMessageIndex].text.split('<div class="feature-details">')[0];
          const updatedText = `${baseMessage}<div class="feature-details">${featureDetails}</div>`;
          console.log("Updating message with:", updatedText.substring(0, 100));
          
          newMessages[lastBotMessageIndex] = {
            ...newMessages[lastBotMessageIndex],
            text: updatedText
          };
        }
        
        return newMessages;
      });
    }
  }, []);

  
  // Helper function to get questions from categories based on context
  const getContextualQuestions = useCallback((messageText) => {
    let relevantQuestions = [];
    const messageTextLower = messageText.toLowerCase();
    
    // If message mentions "more description"
    if (messageTextLower.includes('more description')) {
      // Always set "more details" as the first question
      relevantQuestions = [
        "Tell me more details",
        // Get two additional relevant questions
        ...categories
          .filter(cat => ['Data Visualization', 'Individual Explanations', 'Global Explanations']
          .includes(cat.name))
          .flatMap(cat => cat.questions)
          .filter(q => !q.toLowerCase().includes('more detail'))  // Avoid duplicate "more details" questions
          .slice(0, 2)
      ];
    }
    // If message contains feature importance or analysis
    else if (messageTextLower.includes('feature') || messageTextLower.includes('importance')) {
      relevantQuestions = categories
        .filter(cat => ['Feature Importance', 'Measurement Relationships', 'Individual Explanations']
        .includes(cat.name))
        .flatMap(cat => cat.questions)
        .slice(0, 3);
    }
    // If message shows visualization
    else if (messageTextLower.includes('visualization') || messageTextLower.includes('plot')) {
      relevantQuestions = categories
        .filter(cat => ['Data Visualization', 'Assessment Results', 'What-If Analysis']
        .includes(cat.name))
        .flatMap(cat => cat.questions)
        .slice(0, 3);
    }
    // If message discusses predictions or assessment
    else if (messageTextLower.includes('predict') || messageTextLower.includes('assessment')) {
      relevantQuestions = categories
        .filter(cat => ['Assessment Results', 'Result Probability', 'Potential Errors']
        .includes(cat.name))
        .flatMap(cat => cat.questions)
        .slice(0, 3);
    }
    // Default questions if no specific context is matched
    else {
      relevantQuestions = [
        categories.find(cat => cat.name === 'Global Explanations')?.questions[0],
        categories.find(cat => cat.name === 'Individual Explanations')?.questions[0],
        categories.find(cat => cat.name === 'Data Visualization')?.questions[0],
      ].filter(Boolean);
    }

    // Ensure we have exactly 3 questions and they're unique
    return [...new Set(relevantQuestions)].slice(0, 3);
  }, []);

  useEffect(() => {
    addMessage('left', `Hello! I'm here to help analyze ${datasetObjective}.\n\nYou can ask me questions directly or click the menu (☰) to see example questions.`, '');
  }, [datasetObjective]);

  const addMessage = useCallback((side, text, logText, figureData = null) => {
    const id = Date.now().toString();
    const newMessage = { id, side, text, logText, figureData };
    
    setMessages(prevMessages => [...prevMessages, newMessage]);
    
    // Generate suggested questions only for bot responses
    if (side === 'left') {
      const contextualQuestions = getContextualQuestions(text);
      setSuggestedQuestions(contextualQuestions);
    }
    
  }, [getContextualQuestions]);

  const clearSuggestedQuestions = useCallback(() => {
    setSuggestedQuestions([]);
  }, []);

  return {
    messages,
    addMessage,
    suggestedQuestions,
    clearSuggestedQuestions,
    updateLastMessageWithFeatureDetails  // Add this to the return object
  };
};