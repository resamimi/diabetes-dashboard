import React, { useState, useRef, useContext } from 'react';
import { Card, CardHeader, CardTitle } from './components/ui/card';
import { LineChart } from 'lucide-react';
import { sendBotResponse } from './services/api';
import VisualizationRenderer from './components/VisualizationRenderer';
import RiskAssessmentCard from './components/RiskAssessmentCard';
import ErrorPopup from './components/ErrorPopup';
import { DashboardLayout, WelcomeScreen, PatientSection, ChatSection } from './DashboardUtils';
import './styles/dashboard-messages.css';
import { TranslationContext } from './context/TranslationProvider';
import useActivityTracking from './hooks/useActivityTracking';

const Dashboard = ({ currentUserId, username, onSignOut }) => {
  // State for patient data
  const [patientId, setPatientId] = useState('');
  const [patientData, setPatientData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [originalPrediction, setOriginalPrediction] = useState(null);
  const chatRef = useRef(null);
  const [error, setError] = useState(null);
  const { language, setLanguage, translateContent } = useContext(TranslationContext);
  const { trackChatInteraction, trackVisualizationInteraction } = useActivityTracking(currentUserId);

  // State for visualization
  const [selectedVisualization, setSelectedVisualization] = useState('feature_importance');
  const [visualizationData, setVisualizationData] = useState(null);
  
  // State for chat
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([{
    id: 'intro',
    side: 'left',
    text: "<p class='text-sm text-gray-600'>Hello! I'm here to help analyze diabetes risk predictions. You can ask me questions about risk factors, predictions, and patient data analysis.</p>"
  }]);
  const [suggestedQuestions, setSuggestedQuestions] = useState([
    "What are typical patterns for diabetes risk?",
    "How does the model make predictions?",
    "What are the key factors in diabetes risk assessment?"
  ]);

  // What-if mode state
  const [isWhatIfMode, setIsWhatIfMode] = useState(false);
  const [isProcessingWhatIf, setIsProcessingWhatIf] = useState(false);

  const visualizationOptions = [
    { value: 'feature_importance', label: 'Factor Importance' },
    { value: 'feature_range', label: 'Factor Range Analysis' },
    { value: 'counterfactual_explanation', label: 'Recommendations' },
    // { value: 'classification_scatter', label: 'Classification Results' },
    // { value: 'typical_mistakes', label: 'Typical Mistakes' }
  ];

  const getVisualizationQuestions = (vizType, prediction) => {
    const questionMap = {
      'feature_importance': [
        prediction?.result === 'High Risk' 
          ? "Can you explain why this patient has been flagged as high risk for diabetes?"
          : "Can you explain why this patient has been flagged as low risk for diabetes?",
        "Why BMI is more important than BloodPressure for diabetes risk?",
        "Why is skin thickness in the list of factors?"
      ],
      'feature_range': [
        "What is the meaning of showing AI-observed and Scientific ranges alongside each other?",
        "Why the AI-observed range doesn't exactly align with the Scientific range?",
        "How do these ranges compare to medical guidelines?"
      ],
      'counterfactual_explanation': [
        "Can you provide more specific dietary recommendations for reducing BMI levels?",
        "How long would it take to do all the steps?",
        "What are the most achievable changes to start with?"
      ],
      'classification_scatter': [
        "What patterns do you see in the classification results?",
        "How accurate is the model overall?",
        "What causes misclassifications?"
      ],
      'typical_mistakes': [
        "Why does the model make these specific mistakes?",
        "How can we avoid these misclassifications?",
        "What patterns exist in the incorrect predictions?"
      ]
    };
    return questionMap[vizType] || [
      "What can you tell me about this visualization?",
      "How should I interpret these results?",
      "What are the key insights here?"
    ];
  };

  const additionalQuestionPool = {
    'feature_importance': [
      "How do these factors influence each other?",
      "What lifestyle changes would impact these factors the most?",
      "Can you explain how BMI affects diabetes risk?",
      "What role does age play in these predictions?",
      "How do these factors compare to medical standards?",
      "Which factors can be modified through lifestyle changes?",
      "How do these factors interact with medications?"
    ],
    'feature_range': [
      "What happens when a patient's values fall in these ranges?",
      "How do these ranges compare to medical guidelines?",
      "What causes values to fall outside normal ranges?",
      "How often do healthy patients fall within these ranges?",
      "What lifestyle factors affect these ranges?",
      "How quickly can these values change?",
      "What are the risks of being outside these ranges?",
      "How do medications affect these ranges?"
    ],
    'counterfactual_explanation': [
      "Can you provide more specific dietary recommendations?",
      "How long would it take to see improvements?",
      "What are the most achievable changes to start with?",
      "What are the expected outcomes of these changes?",
      "Which lifestyle modifications are most effective?",
      "How do these changes affect other health metrics?",
      "What support systems help with these changes?",
      "How sustainable are these recommendations?"
    ]
  };

  // Handler functions
  const handlePatientSearch = async (e) => {
    e.preventDefault();
    if (!patientId) return;
  
    try {
      // Fetch patient data
      const patientResponse = await fetch(`/api/patient/${patientId}`);
      const patientResult = await patientResponse.json();
      trackChatInteraction('patient_search', `Searched for patient ID: ${patientId}`);
      
      // Check for error in response
      if (patientResult.error) {
        trackChatInteraction('patient_search_error', `No health record found for Patient ID: ${patientId}`);
        setError(`No health record found for Patient ID: ${patientId}`);
        return;
      }

      // Track successful patient data retrieval
      trackChatInteraction('patient_search_success', `Retrieved data for patient ID: ${patientId}`);
      // If successful, update patient data
      setPatientData(patientResult);
  
      // Fetch prediction
      const predictionResponse = await fetch(`/api/prediction/${patientId}`);
      const predictionResult = await predictionResponse.json();
      
      // Check for error in prediction response
      if (predictionResult.error) {
        setError('Error retrieving prediction data');
        return;
      }
      
      setPrediction(predictionResult);
      setOriginalPrediction(predictionResult);

      // Special handling for current visualization type
      if (selectedVisualization === 'counterfactual_explanation' && predictionResult.result === 'Low Risk') {
        setVisualizationData({
          type: 'counterfactual_explanation',
          data: { noRecommendationsNeeded: true }
        });
      } else {
        // Fetch normal visualization data
        const vizResponse = await fetch(`/api/visualization/${selectedVisualization}/${patientId}`);
        const vizResult = await vizResponse.json();

        if (vizResult.error) {
          setError('Error retrieving visualization data');
          return;
        }

        setVisualizationData(vizResult);
      }
  
      // Update suggested questions
      setSuggestedQuestions(getVisualizationQuestions(selectedVisualization, predictionResult));
      
      // Rest of your existing code...
    } catch (error) {
      trackChatInteraction('patient_search_error', error.message);
      setError('Error: Unable to retrieve patient data. Please check the ID and try again.');
    }
  };

  // Add error close handler
  const handleErrorClose = () => {
    setError(null);
  };

  const handleVisualizationChange = async (newType) => {
    setSelectedVisualization(newType);
    setSuggestedQuestions(getVisualizationQuestions(newType, prediction));
    
    if (patientId) {
      try {
        const response = await fetch(`/api/visualization/${newType}/${patientId}`);
        const result = await response.json();
        trackVisualizationInteraction(newType, 'data_loaded', { 
          success: true,
          patientId 
        });
  
        if (newType === 'counterfactual_explanation' && prediction?.result === 'Low Risk') {
          setVisualizationData({
            type: 'counterfactual_explanation',
            data: { noRecommendationsNeeded: true }
          });
          return;
        }
        
        setVisualizationData(result);
  
      } catch (error) {
        trackVisualizationInteraction(newType, 'error', { 
          error: error.message,
          patientId 
        });
        console.error('Error fetching visualization:', error);
      }
    }
  };

  const handleWhatIfToggle = () => {
    const newWhatIfMode = !isWhatIfMode;
    setIsWhatIfMode(newWhatIfMode);
    
    // Restore original prediction when turning off what-if mode
    if (!newWhatIfMode && originalPrediction) {
      setPrediction(originalPrediction);
    }
  };

  const handleWhatIfSubmit = async (newValues) => {
    if (!patientId) return;
    
    setIsProcessingWhatIf(true);
    try {
      const response = await fetch(`/api/what-if/${patientId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newValues),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get what-if prediction');
      }
      
      const result = await response.json();
      // Update the prediction state with the new result
      setPrediction({
        result: result.result,
        confidence: result.confidence
      });
      
    } catch (error) {
      console.error('Error in what-if analysis:', error);
    } finally {
      setIsProcessingWhatIf(false);
    }
  };

  const formatJSONResponse = (responseText) => {
    try {
      // First, try to extract JSON from the response using regex
      const jsonRegex = /{[\s\S]*"recommendations"[\s\S]*}/;
      const match = responseText.match(jsonRegex);
      
      if (!match) {
        console.warn('No JSON found in response');
        return responseText;
      }
  
      // Clean up the extracted JSON string
      const cleanedJson = match[0]
        .replace(/<\/?p>/g, '')  // Remove <p> tags
        .replace(/<\/?h[1-6]>/g, '')  // Remove header tags
        .replace(/\\n/g, '')  // Remove escaped newlines
        .replace(/\n/g, '')   // Remove newlines
        .trim();
  
      // Parse the cleaned JSON
      const jsonResponse = JSON.parse(cleanedJson);
      
      // Validate the expected structure
      if (!jsonResponse.recommendations || !Array.isArray(jsonResponse.recommendations)) {
        console.warn('Invalid JSON structure');
        return responseText;
      }
  
      // Format the recommendations with bold title and line breaks
      let formattedHtml = '<p><strong>Recommended actions:</strong></p><br>';
      
      // Add each recommendation with line break between items
      jsonResponse.recommendations.forEach((rec, index) => {
        formattedHtml += `<p>${index + 1}. ${rec.action} | ${rec.timeline}</p><br>`;
      });
      
      // Remove the last <br> tag to avoid extra space at the end
      formattedHtml = formattedHtml.replace(/<br>$/, '');
      
      return formattedHtml;
    } catch (error) {
      console.error('Error processing JSON response:', error);
      // Return original response if JSON processing fails
      return responseText;
    }
  };
  
  const handleRecommendationClick = async (displayPrompt, detailedPrompt) => {
    const questionId = Date.now() + '-user';
    const isRecommendationRequest = true;  // Add this flag
      
    // Add user message and scroll to it
    setMessages(prevMessages => {
      const newMessages = [...prevMessages, {
        id: questionId,
        side: 'right',
        text: `<p>${displayPrompt}</p>`,
        isRecommendationRequest  // Add the flag to the message
      }];
      
      setTimeout(() => {
        if (chatRef.current) {
          const questionElement = chatRef.current.querySelector(`[data-message-id="${questionId}"]`);
          if (questionElement) {
            questionElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }
      }, 100);
      
      return newMessages;
    });
  
    setIsLoading(true);
  
    try {
      const [responseText, responseId] = await sendBotResponse(detailedPrompt, currentUserId);
      const formattedResponse = formatJSONResponse(responseText);
  
      // Add bot's response to messages
      setMessages(prevMessages => [...prevMessages, {
        id: responseId,
        side: 'left',
        text: formattedResponse,
        isRecommendationRequest  // Add the flag to the response
      }]);
      
      setTimeout(() => {
        if (chatRef.current) {
          const responseElement = chatRef.current.querySelector(`[data-message-id="${responseId}"]`);
          if (responseElement) {
            responseElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }
      }, 100);
    } catch (error) {
      console.error('Error getting bot response:', error);
      setMessages(prevMessages => [...prevMessages, {
        id: Date.now() + '-error',
        side: 'left',
        text: '<p>Sorry, I encountered an error processing your request.</p>',
        isRecommendationRequest
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeatureClick = (feature, question, featureSection, visualizationType) => {
    // If it's a recommendation request from CounterfactualPlot
    if (feature === null && visualizationType === undefined) {
      // Route to the recommendation handler
      handleRecommendationClick(question, featureSection);
      return;
    }
    
    // Otherwise handle as normal feature click
    const questionId = Date.now() + '-user';
    const responseId = Date.now() + '-bot';
    
    setMessages(prevMessages => [...prevMessages,
      {
        id: questionId,
        side: 'right',
        text: `<p>${question}</p>`
      },
      {
        id: responseId,
        side: 'left',
        text: featureSection
      }
    ]);

    // Add scrolling behavior
    if (chatRef.current) {
      setTimeout(() => {
        const questionElement = chatRef.current.querySelector(`[data-message-id="${questionId}"]`);
        if (questionElement) {
          questionElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 100);
    }
  };

  const handleQuestionSelect = async (question) => {
    console.log('Selected question with userId:', currentUserId);
    trackChatInteraction('suggested_question_click', question);
    const questionId = Date.now() + '-user';
  
    // Get current visualization type's question pool
    const currentPool = additionalQuestionPool[selectedVisualization] || [];
    
    // Get current questions except the selected one
    const remainingQuestions = suggestedQuestions.filter(q => q !== question);
    
    // Special handling for feature importance visualization
    if (selectedVisualization === 'feature_importance') {
      // Keep the risk explanation question if it wasn't the selected question
      if (question === remainingQuestions[0]) {
        // If risk question was selected, use the second question as first
        remainingQuestions.push(currentPool[0]);
      }
    }
  
    // Get potential new questions (excluding both current and remaining questions)
    const potentialNewQuestions = currentPool.filter(q => 
      !suggestedQuestions.includes(q) && !remainingQuestions.includes(q)
    );
  
    // Select a random new question from potential questions
    let newQuestion = '';
    if (potentialNewQuestions.length > 0) {
      const randomIndex = Math.floor(Math.random() * potentialNewQuestions.length);
      newQuestion = potentialNewQuestions[randomIndex];
    } else {
      // If no new questions available, use a default question
      newQuestion = "Can you explain more about this aspect?";
    }
  
    // Update suggested questions with remaining questions plus new question
    const updatedQuestions = [...remainingQuestions, newQuestion];
    setSuggestedQuestions(updatedQuestions);
    
    // Rest of the function remains the same...
    setMessages(prevMessages => [...prevMessages, {
      id: questionId,
      side: 'right',
      text: `<p>${question}</p>`
    }]);
  
    setIsLoading(true);
  
    try {
      const [responseText, responseId] = await sendBotResponse(question, currentUserId);
      
      let textContent, visualizationData;
      if (responseText.includes('<json>')) {
        const [text, json] = responseText.split('<json>');
        textContent = await translateContent(text);
        try {
          visualizationData = JSON.parse(json);
        } catch (error) {
          console.error('Error parsing visualization data:', error);
          visualizationData = null;
        }
      } else {
        textContent = await translateContent(responseText);
        visualizationData = null;
      }
      
      const formattedResponse = textContent.startsWith('<p>') 
        ? textContent 
        : `<p>${textContent}</p>`;
      
      setMessages(prevMessages => [...prevMessages, {
        id: responseId,
        side: 'left',
        text: formattedResponse
      }]);
  
      // Only update visualization if type changes
      if (visualizationData && visualizationData.type && visualizationData.type !== selectedVisualization) {
        setSelectedVisualization(visualizationData.type);
        setVisualizationData(visualizationData);
        setSuggestedQuestions(getVisualizationQuestions(visualizationData.type, prediction));
      }
  
      if (chatRef.current) {
        setTimeout(() => {
          const questionElement = chatRef.current.querySelector(`[data-message-id="${questionId}"]`);
          if (questionElement) {
            questionElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }, 100);
      }
    } catch (error) {
      console.error('Error getting bot response:', error);
      const errorMessage = await translateContent('Sorry, I encountered an error processing your request.');
      setMessages(prevMessages => [...prevMessages, {
        id: Date.now() + '-error',
        side: 'left',
        text: `<p>${errorMessage}</p>`
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const processResponse = (responseText) => {
    // Check if response includes visualization data
    if (responseText.includes('<json>')) {
      // Split response into text and visualization parts
      const [textContent, jsonContent] = responseText.split('<json>');
      
      try {
        // Parse the visualization data
        const visualizationData = JSON.parse(jsonContent);
        return {
          textContent: textContent.trim(),
          visualizationData
        };
      } catch (error) {
        console.error('Error parsing visualization data:', error);
        return {
          textContent: textContent.trim(),
          visualizationData: null
        };
      }
    }
    
    // If no visualization data, return text only
    return {
      textContent: responseText,
      visualizationData: null
    };
  };
  
  // // Function to update visualization display
  // const updateVisualization = (visualizationData, setSelectedVisualization, setVisualizationData) => {
  //   if (visualizationData && visualizationData.type) {
  //     // Update visualization type
  //     setSelectedVisualization(visualizationData.type);
  //     // Update visualization data
  //     setVisualizationData(visualizationData);
  //   }
  // };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;
    
    setMessages(prevMessages => [...prevMessages, {
      id: Date.now() + '-user',
      side: 'right',
      text: `<p>${inputValue}</p>`
    }]);
  
    setIsLoading(true);
  
    try {
      const [responseText, responseId] = await sendBotResponse(inputValue, currentUserId);
      
      let textContent, visualizationData;
      if (responseText.includes('<json>')) {
        const [text, json] = responseText.split('<json>');
        textContent = await translateContent(text);
        try {
          visualizationData = JSON.parse(json);
        } catch (error) {
          console.error('Error parsing visualization data:', error);
          visualizationData = null;
        }
      } else {
        textContent = await translateContent(responseText);
        visualizationData = null;
      }
      
      const formattedResponse = textContent.startsWith('<p>') 
        ? textContent 
        : `<p>${textContent}</p>`;
      
      setMessages(prevMessages => [...prevMessages, {
        id: responseId,
        side: 'left',
        text: formattedResponse
      }]);
  
      if (visualizationData && visualizationData.type) {
        setSelectedVisualization(visualizationData.type);
        setVisualizationData(visualizationData);
        // Update suggested questions based on new visualization type
        setSuggestedQuestions(getVisualizationQuestions(visualizationData.type));
      }
  
    } catch (error) {
      console.error('Error getting bot response:', error);
      const errorMessage = await translateContent('Sorry, I encountered an error processing your request.');
      setMessages(prevMessages => [...prevMessages, {
        id: Date.now() + '-error',
        side: 'left',
        text: `<p>${errorMessage}</p>`
      }]);
    } finally {
      setIsLoading(false);
    }
  
    setInputValue('');
  };

  const generateContextualQuestions = (responseText) => {
    const textLower = responseText.toLowerCase();
    
    if (textLower.includes('risk factor') || textLower.includes('importance')) {
      return [
        "Can you explain these factors in more detail?",
        "How do these factors interact with each other?",
        "What changes would have the biggest impact?"
      ];
    } else if (textLower.includes('prediction') || textLower.includes('assessment')) {
      return [
        "What's the confidence level of this prediction?",
        "What are the key factors in this assessment?",
        "How can the risk be reduced?"
      ];
    } else if (textLower.includes('pattern') || textLower.includes('trend')) {
      return [
        "Can you show me specific examples?",
        "How reliable are these patterns?",
        "What causes these patterns?"
      ];
    }
    
    return [
      "Tell me more about this",
      "What are the key factors?",
      "How can this be improved?"
    ];
  };

  // Render main visualization section
  const renderMainContent = () => {
    if (!patientData) {
      return (
        <Card className="backdrop-blur-sm bg-white/95 shadow-md h-full">
          <WelcomeScreen />
        </Card>
      );
    }

    return (
      <div className="h-full flex flex-col">
        {/* Risk Assessment should never be disabled */}
        <div>
          <RiskAssessmentCard prediction={prediction} />
        </div>
        
        {/* Only visualizations are disabled in what-if mode */}
        <div className={`flex-1 ${isWhatIfMode ? 'opacity-50 pointer-events-none' : ''}`}>
          <Card className="backdrop-blur-sm bg-white/95 shadow-md h-full overflow-hidden">
            <CardHeader className="border-b border-gray-100 py-2 px-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <LineChart className="w-4 h-4 text-blue-500" />
                  <CardTitle data-translate="true" className="text-sm font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-purple-600">
                    Analysis Visualization
                  </CardTitle>
                </div>
                <select
                  value={selectedVisualization}
                  onChange={(e) => handleVisualizationChange(e.target.value)}
                  className="px-3 py-1.5 text-sm border rounded-lg bg-white/50 backdrop-blur-sm
                            border-gray-200 text-gray-700 focus:ring-2 focus:ring-blue-500/20 
                            focus:border-blue-500 transition-all"
                >
                  {visualizationOptions.map(option => (
                    <option key={option.value} value={option.value} data-translate="true">
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </CardHeader>
            <div className="relative flex-1 h-[calc(100%-3.5rem)]">
              <div className="absolute inset-0 overflow-y-auto overflow-x-hidden">
                <div className="p-2">
                  <VisualizationRenderer 
                    visualizationData={visualizationData}
                    onFeatureClick={handleFeatureClick}
                    userId={currentUserId}
                  />
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    );
  };

  const handleSignOut = async () => {
    try {
      // Get session ID from localStorage if exists
      const sessionId = localStorage.getItem('sessionId');
      const userId = localStorage.getItem('userId');
  
      if (!userId) {
        console.warn('No user ID found during sign out');
        // Still proceed with local cleanup
        localStorage.clear();
        onSignOut();
        return;
      }
  
      // Call backend to end session
      const response = await fetch('/api/auth/signout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId,
          userId
        }),
      });
  
      const data = await response.json();
  
      if (!response.ok) {
        throw new Error(data.error || 'Sign out failed');
      }
  
      // Clear local storage
      localStorage.clear();
  
      // Call the onSignOut prop passed from App.js
      onSignOut();
  
    } catch (error) {
      console.error('Error signing out:', error);
      
      // Show error to user
      setError(error.message || 'Failed to sign out. Please try again.');
      
      // Attempt local cleanup anyway after error
      setTimeout(() => {
        localStorage.clear();
        onSignOut();
      }, 2000);
    }
  };

  const renderUserInfo = () => (
    <div className="flex items-center gap-2 text-sm text-gray-600">
      <span>Logged in as: {username}</span>
      <button 
        onClick={onSignOut}
        className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
      >
        Sign Out
      </button>
    </div>
  );

  return (
    <DashboardLayout
      isWhatIfMode={isWhatIfMode}
      patientColumn={
        <PatientSection
          patientId={patientId}
          setPatientId={setPatientId}
          handlePatientSearch={handlePatientSearch}
          patientData={patientData}
          language={language}
          setLanguage={setLanguage}
          userInfo={renderUserInfo()} // Add user info to patient section
          onSignOut={handleSignOut}  
        />
      }
      chatColumn={
        <ChatSection
          messages={messages}
          isLoading={isLoading}
          suggestedQuestions={suggestedQuestions}
          handleQuestionSelect={handleQuestionSelect}
          inputValue={inputValue}
          setInputValue={setInputValue}
          handleSubmit={handleSubmit}
          chatRef={chatRef}
          userId={currentUserId} 
        />
      }
    >
      {renderMainContent()}
      {error && <ErrorPopup message={error} onClose={handleErrorClose} />}
    </DashboardLayout>
  );
};

export default Dashboard;

