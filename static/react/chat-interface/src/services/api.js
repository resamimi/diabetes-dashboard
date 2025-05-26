

export const sendBotResponse = async (rawText, currentUserId) => {
    const dataPackage = JSON.stringify({ 
        userInput: rawText, 
        userId: currentUserId  // Changed from userName to userId
    });
    try {
      const response = await fetch('/get_bot_response', {
        method: 'POST',
        body: dataPackage,
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.text();
      return data.split('<>');
    } catch (error) {
      console.error('Error in botResponse:', error);
      throw error;
    }
  };
  
  export const logFeedback = async (messageId, feedbackType, currentUserId) => {
    const feedback = `MessageID: ${messageId} || Feedback: ${feedbackType} || Username: ${currentUserId}`;
    try {
      await fetch('/log_feedback', {
        method: 'POST',
        body: feedback,
        headers: { 'Content-Type': 'text/plain' },
      });
    } catch (error) {
      console.error('Error logging feedback:', error);
      throw error;
    }
};