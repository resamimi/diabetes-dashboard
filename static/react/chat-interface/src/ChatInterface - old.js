import React, { useState, useRef, useEffect } from 'react';
import { marked } from 'marked';
// import './ChatInterface.css';
import './styles/layout.css';
import './styles/messages.css';
import './styles/features.css';
import './styles/utilities.css';
import WrongAnswerDialog from './components/WrongAnswerDialog';
import Sidebar from './components/Sidebar';
import ChatMessage from './components/ChatMessage';
import FavoriteMessages from './components/FavoriteMessages';
import { useChatMessages } from './hooks/useChatMessages';
import { sendBotResponse, logFeedback } from './services/api';
import LoadingMessage from './components/LoadingMessage';

const ChatInterface = ({ datasetObjective, currentUserId }) => {
  const [messages, addMessage] = useChatMessages(datasetObjective);
  const [favoriteMessages, setFavoriteMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [userInputs, setUserInputs] = useState([]);
  const [userInputIndex, setUserInputIndex] = useState(0);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [suggestedPrompt, setSuggestedPrompt] = useState('');
  const [isWrongAnswerDialogOpen, setIsWrongAnswerDialogOpen] = useState(false);
  const [selectedMessageId, setSelectedMessageId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const chatRef = useRef(null);
  const inputRef = useRef(null);
  const messageRefs = useRef({});

  // Update message refs when messages change
  useEffect(() => {
    messages.forEach(message => {
      if (!messageRefs.current[message.id]) {
        messageRefs.current[message.id] = React.createRef();
      }
    });
  }, [messages]);

  // Handle suggested prompts
  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.side === 'left' && lastMessage.text.toLowerCase().includes('more description')) {
        setSuggestedPrompt('more description');
      } else {
        setSuggestedPrompt('');
      }
    }
  }, [messages]);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    addMessage('right', inputValue, '');
    setUserInputs(prevInputs => [...prevInputs, inputValue]);
    setUserInputIndex(prevIndex => prevIndex + 1);
    setIsLoading(true);
    try {
      await botResponse(inputValue);
    } finally {
      setIsLoading(false);
    }
    setInputValue('');
    setSelectedCategory(null);
  };

  const botResponse = async (rawText) => {
    try {
      const [dataText, logText] = await sendBotResponse(rawText, currentUserId);
      let msgText = '';
      let figJson = '';
      let mdText = '';

      if (dataText.includes('<json>')) {
        [msgText, figJson] = dataText.split('<json>');
      } else {
        msgText = dataText;
      }

      if (msgText.includes('<markdown>')) {
        [, mdText] = msgText.split('<markdown>');
        msgText = marked(mdText);
      }

      if (figJson) {
        const figureData = { data: JSON.parse(figJson) };
        addMessage('left', msgText, logText, figureData);
      } else {
        addMessage('left', msgText, logText);
      }
    } catch (error) {
      console.error('Error in botResponse:', error);
      addMessage('left', 'Sorry, I encountered an error processing your request.', '');
    }
  };

  const handleFeedback = async (messageId, feedbackType) => {
    try {
      await logFeedback(messageId, feedbackType, currentUserId);
    } catch (error) {
      console.error('Error logging feedback:', error);
    }
  };

  const handleWrongAnswer = (messageId) => {
    setSelectedMessageId(messageId);
    setIsWrongAnswerDialogOpen(true);
  };

  const shortenMessage = (text) => {
    const plainText = text.replace(/<[^>]*>/g, '');
    return plainText.length > 50 ? plainText.substring(0, 50) + '...' : plainText;
  };

  const toggleFavorite = (message) => {
    setFavoriteMessages(prevFavorites => {
      const isAlreadyFavorite = prevFavorites.some(fav => fav.id === message.id);
      
      if (isAlreadyFavorite) {
        return prevFavorites;
      }

      const favoriteMessage = {
        id: message.id,
        shortText: shortenMessage(message.text),
        fullText: message.text,
        timestamp: new Date().toLocaleTimeString()
      };

      return [...prevFavorites, favoriteMessage];
    });
  };

  const scrollToMessage = (messageId) => {
    const messageRef = messageRefs.current[messageId];
    if (messageRef?.current) {
      messageRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
      messageRef.current.classList.add('highlight-message');
      setTimeout(() => {
        messageRef.current.classList.remove('highlight-message');
      }, 2000);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setUserInputIndex(prevIndex => Math.max(0, prevIndex - 1));
      setInputValue(userInputs[userInputIndex] || '');
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setUserInputIndex(prevIndex => Math.min(userInputs.length, prevIndex + 1));
      setInputValue(userInputs[userInputIndex] || '');
    }
  };

  const handleCategorySelect = (category) => {
    setSelectedCategory(selectedCategory === category ? null : category);
  };

  const handleQuestionSelect = (question) => {
    setInputValue(question);
    setSelectedCategory(null);
    setIsSidebarOpen(false);
  };

  const handleSuggestionClick = () => {
    setInputValue('more description');
    setSuggestedPrompt('');
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  return (
    <div className="chat-interface">
      <header className="chat-header">
        <div className="header-content">
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
            className="sidebar-toggle"
          >
            {isSidebarOpen ? '✖' : '☰'}
          </button>
          <h1>Conversational and Visual Interface for Explainable AI</h1>
        </div>
      </header>

      <div className="chat-container">
        <Sidebar
          isOpen={isSidebarOpen}
          selectedCategory={selectedCategory}
          onCategorySelect={handleCategorySelect}
          onQuestionSelect={handleQuestionSelect}
        />

        <div className="chat-main">
          <div className="chat-messages" ref={chatRef}>
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                ref={messageRefs.current[message.id]}
                message={message}
                onFeedback={handleFeedback}
                onWrongAnswer={handleWrongAnswer}
                onToggleFavorite={toggleFavorite}
              />
            ))}
            {isLoading && <LoadingMessage />}
          </div>

          {suggestedPrompt && (
            <div className="suggested-prompt" onClick={handleSuggestionClick}>
              Click to ask for more details →
            </div>
          )}

          <form className="chat-input" onSubmit={handleSubmit}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter your message or open the sidebar for guided questions..."
              ref={inputRef}
            />
            <button type="submit">
              <svg 
                viewBox="0 0 24 24" 
                width="24" 
                height="24" 
                stroke="currentColor" 
                strokeWidth="2" 
                fill="none" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              >
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </form>
        </div>

        <FavoriteMessages
          favorites={favoriteMessages}
          onMessageClick={scrollToMessage}
        />
      </div>

      <WrongAnswerDialog
        isOpen={isWrongAnswerDialogOpen}
        onClose={() => setIsWrongAnswerDialogOpen(false)}
        onSelect={handleFeedback}
        messageId={selectedMessageId}
      />
    </div>
  );
};

export default ChatInterface;