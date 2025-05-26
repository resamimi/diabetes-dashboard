import { BrainCircuit, Bot, Users, Search, ClipboardCheck, Globe, LogOut } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card';
import PatientDataPlot from './components/PatientDataPlot';
import React, { useRef, useEffect, useState } from 'react';
import LoadingMessage from './components/LoadingMessage';
import SuggestedPrompts from './components/SuggestedPrompts';
import { useTranslation } from './context/TranslationProvider';
import useTimeTracking from './hooks/useTimeTracking';

export const WelcomeScreen = () => {
  const { language, translateContent } = useTranslation();
  const [translations, setTranslations] = useState({
    title: "Welcome to the Diabetes Risk Analysis System",
    description: "To begin your analysis, either enter a patient ID in the left panel or ask a question to our AI assistant in the right panel.",
    patientTitle: "Patient Analysis",
    patientDesc: "Enter a patient ID to view their detailed health metrics and risk assessment",
    assistantTitle: "AI Assistant",
    assistantDesc: "Ask questions about diabetes risk factors and get detailed explanations"
  });

  useEffect(() => {
    const translateTexts = async () => {
      if (language === 'en') {
        setTranslations({
          title: "Welcome to the Diabetes Risk Analysis System",
          description: "To begin your analysis, either enter a patient ID in the left panel or ask a question to our AI assistant in the right panel.",
          patientTitle: "Patient Analysis",
          patientDesc: "Enter a patient ID to view their detailed health metrics and risk assessment",
          assistantTitle: "AI Assistant",
          assistantDesc: "Ask questions about diabetes risk factors and get detailed explanations"
        });
        return;
      }

      const translatedTexts = {
        title: await translateContent("Welcome to the Diabetes Risk Analysis System"),
        description: await translateContent("To begin your analysis, either enter a patient ID in the left panel or ask a question to our AI assistant in the right panel."),
        patientTitle: await translateContent("Patient Analysis"),
        patientDesc: await translateContent("Enter a patient ID to view their detailed health metrics and risk assessment"),
        assistantTitle: await translateContent("AI Assistant"),
        assistantDesc: await translateContent("Ask questions about diabetes risk factors and get detailed explanations")
      };

      setTranslations(translatedTexts);
    };

    translateTexts();
  }, [language, translateContent]);

  return (
    <div className="h-full flex flex-col items-center justify-center text-center px-8 py-12">
      <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 shadow-lg">
        <BrainCircuit className="w-8 h-8 text-white" />
      </div>
      
      <h2 data-translate="true" className="text-base font-semibold text-gray-900 mb-3">
        {translations.title}
      </h2>
      
      <p data-translate="true" className="text-sm text-gray-600 mb-6 max-w-md">
        {translations.description}
      </p>
      
      <div className="grid grid-cols-2 gap-4 w-full max-w-md">
        <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
          <div className="flex items-center gap-2 mb-2">
            <Users className="w-5 h-5 text-blue-600" />
            <h3 className="text-sm font-medium text-gray-900">
              {translations.patientTitle}
            </h3>
          </div>
          <p className="text-sm text-gray-600">
            {translations.patientDesc}
          </p>
        </div>
        
        <div className="p-4 bg-purple-50 rounded-lg border border-purple-100">
          <div className="flex items-center gap-2 mb-2">
            <Bot className="w-5 h-5 text-purple-600" />
            <h3 className="text-sm font-medium text-gray-900">
              {translations.assistantTitle}
            </h3>
          </div>
          <p className="text-sm text-gray-600">
            {translations.assistantDesc}
          </p>
        </div>
      </div>
    </div>
  );
};

export const DashboardLayout = ({ 
  children, 
  patientColumn, 
  chatColumn,
  isWhatIfMode 
}) => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 p-6">
      <div className="grid grid-cols-3 gap-6 max-w-[1800px] mx-auto min-h-[calc(100vh-3rem)] max-h-[calc(100vh-3rem)]">
        {/* Patient Data Column - Added overflow handling */}
        <div className="space-y-6 overflow-auto">
          {patientColumn}
        </div>

        {/* Middle Column - Added overflow handling */}
        <div className="overflow-auto">
          {children}
        </div>

        {/* Chat Column - Added overflow handling */}
        <div className={`overflow-auto ${isWhatIfMode ? 'opacity-50 pointer-events-none' : ''}`}>
          {chatColumn}
        </div>
      </div>
    </div>
  );
};


export const PatientSection = ({
  patientId,
  setPatientId,
  handlePatientSearch,
  patientData,
  language,
  setLanguage,
  userInfo,
  onSignOut
}) => {
  const { translateContent } = useTranslation();
  const [translations, setTranslations] = useState({
    settings: "Settings",
    patientIdLabel: "Patient ID",
    patientIdPlaceholder: "Enter ID...",
    languageLabel: "Language",
    noDataMessage: "Enter a patient ID to view their data",
    signOut: "Sign Out" 
  });

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'sl', name: 'Slovenian' }
  ];

  useEffect(() => {
    const updateTranslations = async () => {
      if (language === 'en') {
        setTranslations({
          settings: "Settings",
          patientIdLabel: "Patient ID",
          patientIdPlaceholder: "Enter ID...",
          languageLabel: "Language",
          noDataMessage: "Enter a patient ID to view their data",
          signOut: "Sign Out"
        });
        return;
      }

      const translated = {
        settings: await translateContent("Settings"),
        patientIdLabel: await translateContent("Patient ID"),
        patientIdPlaceholder: await translateContent("Enter ID..."),
        languageLabel: await translateContent("Language"),
        noDataMessage: await translateContent("Enter a patient ID to view their data"),
        signOut: await translateContent("Sign Out")
      };

      setTranslations(translated);
    };

    updateTranslations();
  }, [language, translateContent]);

  return (
    <Card className="backdrop-blur-sm bg-white/95 shadow-md hover:shadow-lg transition-shadow duration-200 h-full">
      <CardHeader className="border-b border-gray-100 py-2 px-4">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <ClipboardCheck className="w-4 h-4 text-blue-500" />
            <CardTitle className="text-sm font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-purple-600">
              {translations.settings}
            </CardTitle>
          </div>
          <button
            onClick={onSignOut}
            className="text-sm text-gray-600 hover:text-gray-900 flex items-center gap-1 px-2 py-1 rounded-md hover:bg-gray-100 transition-colors"
          >
            <LogOut className="w-3 h-3" />
            {translations.signOut}
          </button>
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          {/* Patient ID Section */}
          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-600">
              {translations.patientIdLabel}
            </label>
            <form onSubmit={handlePatientSearch} className="relative">
              <input
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                placeholder={translations.patientIdPlaceholder}
                className="w-full px-2 py-1 border border-gray-200 rounded-lg pr-8 text-sm
                        focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
              />
              <button
                type="submit"
                className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded-lg
                        hover:bg-gray-100 active:bg-gray-200 transition-colors"
              >
                <Search className="w-3.5 h-3.5 text-gray-500" />
              </button>
            </form>
          </div>

          {/* Language Selection Section */}
          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-600">
              {translations.languageLabel}
            </label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full px-2 py-1 border border-gray-200 rounded-lg text-sm bg-white
                       focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
            >
              {languages.map(lang => (
                <option key={lang.code} value={lang.code}>
                  {lang.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-3">
        {patientData ? (
          <PatientDataPlot
            visualizationData={patientData}
            className="w-full"
          />
        ) : (
          <div className="h-64 flex items-center justify-center text-gray-500">
            <p className="text-sm">{translations.noDataMessage}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export const ChatSection = ({
  messages,
  isLoading,
  suggestedQuestions,
  handleQuestionSelect,
  inputValue,
  setInputValue,
  handleSubmit,
  chatRef,
  userId // Add userId prop for time tracking
}) => {
  const { language, translateContent } = useTranslation();
  const inputRef = useRef(null);
  const [translatedMessages, setTranslatedMessages] = useState(messages);
  const [translatedQuestions, setTranslatedQuestions] = useState(suggestedQuestions);

  // Initialize time tracking
  const timeTracking = useTimeTracking(userId, 'ChatSection', {
    inactivityThreshold: 300000, // 5 minutes for chat
    logInterval: 120000, // Log every 2 minutes
    minTimeToLog: 5000 // Minimum 5 seconds
  });

  // Log time when user starts typing
  const handleInputFocus = () => {
    timeTracking.logCurrentDuration('chat_input_focus');
  };

  // Log time when message is sent
  const handleMessageSubmit = (e) => {
    timeTracking.logCurrentDuration('message_sent');
    handleSubmit(e);
  };

  // Translate messages when they change or language changes
  useEffect(() => {
    const translateMessages = async () => {
      if (language === 'en') {
        setTranslatedMessages(messages);
        return;
      }

      const translated = await Promise.all(
        messages.map(async (message) => ({
          ...message,
          text: await translateContent(message.text)
        }))
      );

      setTranslatedMessages(translated);
    };

    translateMessages();
  }, [messages, language, translateContent]);

  // Translate suggested questions when they change or language changes
  useEffect(() => {
    const translateQuestions = async () => {
      if (language === 'en') {
        setTranslatedQuestions(suggestedQuestions);
        return;
      }

      const translated = await Promise.all(
        suggestedQuestions.map(q => translateContent(q))
      );

      setTranslatedQuestions(translated);
    };

    translateQuestions();
  }, [suggestedQuestions, language, translateContent]);

  // Translate placeholder text
  const [placeholder, setPlaceholder] = useState("Type your question...");

  useEffect(() => {
    const updatePlaceholder = async () => {
      const translatedPlaceholder = await translateContent("Type your question...");
      setPlaceholder(translatedPlaceholder);
    };

    if (language !== 'en') {
      updatePlaceholder();
    } else {
      setPlaceholder("Type your question...");
    }
  }, [language, translateContent]);

  // Log time when component unmounts
  useEffect(() => {
    return () => {
      timeTracking.logCurrentDuration('chat_section_closed');
    };
  }, []);

  return (
    <Card className="backdrop-blur-sm bg-white/95 shadow-md h-full flex flex-col overflow-hidden">
      <CardHeader className="border-b border-gray-100 py-3 px-6 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-purple-600" />
          <CardTitle className="text-sm font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-purple-600">
            AI Assistant
          </CardTitle>
        </div>
      </CardHeader>

      <div className="flex flex-col flex-1 overflow-hidden">
        <div 
          ref={chatRef}
          className="flex-1 overflow-y-auto p-4 space-y-4"
          style={{ maxHeight: 'calc(100vh - 10rem)' }}
        >
          {translatedMessages.map(message => (
            <div 
              key={message.id}
              data-message-id={message.id}
              className={`chat-message ${message.side} break-words`}
            >
              <div className="message-bubble break-words">
                <div 
                  className="break-words"
                  dangerouslySetInnerHTML={{ __html: message.text }} 
                />
              </div>
              {message.side === 'left' && message.id !== 'intro' && (
                <div className="message-actions">
                  <button 
                    title="Positive" 
                    onClick={() => timeTracking.logCurrentDuration('feedback_positive')}
                  >
                    👍
                  </button>
                  <button 
                    title="Negative"
                    onClick={() => timeTracking.logCurrentDuration('feedback_negative')}
                  >
                    👎
                  </button>
                  <button 
                    title="Insightful"
                    onClick={() => timeTracking.logCurrentDuration('feedback_insightful')}
                  >
                    💡
                  </button>
                  <button 
                    title="Confused"
                    onClick={() => timeTracking.logCurrentDuration('feedback_confused')}
                  >
                    😕
                  </button>
                  <button 
                    title="Wrong Answer"
                    onClick={() => timeTracking.logCurrentDuration('feedback_wrong')}
                  >
                    ❌
                  </button>
                </div>
              )}
            </div>
          ))}

          {isLoading && <LoadingMessage />}
          
          {!isLoading && translatedQuestions.length > 0 && (
            <SuggestedPrompts 
              questions={translatedQuestions}
              onQuestionClick={(question) => {
                timeTracking.logCurrentDuration('suggested_question_selected');
                handleQuestionSelect(question);
              }}
            />
          )}
        </div>

        <div className="flex-shrink-0 px-3 py-2 border-t border-gray-100 bg-white/50">
          <form onSubmit={handleMessageSubmit} className="relative">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onFocus={handleInputFocus}
              placeholder={placeholder}
              className="w-full px-3 py-1.5 pr-10 text-sm border border-gray-200 rounded-lg
                      focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
            />
            <button 
              type="submit"
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-lg
                      hover:bg-gray-100 active:bg-gray-200 transition-colors"
            >
              <svg className="w-4 h-4 text-gray-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
              </svg>
            </button>
          </form>
        </div>
      </div>
    </Card>
  );
};


