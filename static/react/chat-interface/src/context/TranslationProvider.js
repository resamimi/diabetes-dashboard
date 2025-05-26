import React, { createContext, useContext, useCallback, useEffect, useState, useRef } from 'react';

export const TranslationContext = createContext();
const BATCH_SIZE = 10; // Number of texts to translate in one batch
const BATCH_DELAY = 100; // Milliseconds to wait between batches

export const TranslationProvider = ({ children }) => {
  const [language, setLanguage] = useState('en');
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState(null);
  const translationCache = useRef(new Map());
  
  // Comprehensive storage for original content
  const originalContent = useRef(new Map());
  
  const pendingTranslations = useRef([]);
  const isProcessing = useRef(false);

  // Enhanced language change handler
  const handleLanguageChange = useCallback((newLanguage) => {
    const previousLanguage = language;
    setLanguage(newLanguage);

    if (newLanguage === 'en') {
      // Restore ALL original content when switching back to English
      originalContent.current.forEach((originalInfo, element) => {
        if (document.contains(element)) {
          // Restore different types of content
          if (originalInfo.type === 'text') {
            element.textContent = originalInfo.value;
          } else if (originalInfo.type === 'attribute') {
            element.setAttribute(originalInfo.attribute, originalInfo.value);
          } else if (originalInfo.type === 'innerHTML') {
            element.innerHTML = originalInfo.value;
          }
        }
      });

      // Clear translation caches
      translationCache.current.clear();
      originalContent.current.clear();
    }
  }, [language]);

  // Comprehensive content storage function
  const storeOriginalContent = (element, content, type, attribute = null) => {
    if (!originalContent.current.has(element)) {
      originalContent.current.set(element, {
        type,
        value: content,
        attribute: attribute || null
      });
    }
  };

  // Batch translate multiple texts
  const batchTranslate = async (texts) => {
    if (!texts.length) return {};
    
    try {
      // Process texts in parallel but with individual requests
      const translations = await Promise.all(
        texts.map(async (text) => {
          const response = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: text,
              to: language
            })
          });

          if (!response.ok) throw new Error('Translation failed');
          
          const data = await response.json();
          return [text, data.translation];
        })
      );

      // Convert array of translations to object
      return Object.fromEntries(translations);
    } catch (err) {
      console.error('Batch translation error:', err);
      return {};
    }
  };

  // Process translation queue in batches
  const processTranslationQueue = useCallback(async () => {
    if (isProcessing.current || !pendingTranslations.current.length) return;
    
    isProcessing.current = true;
    setIsTranslating(true);

    try {
      while (pendingTranslations.current.length > 0) {
        // Take a batch of elements to translate
        const batch = pendingTranslations.current.splice(0, BATCH_SIZE);
        const textsToTranslate = [];
        const elementMap = new Map();

        // Prepare batch of texts
        batch.forEach(({ element, textContent, type, attribute }) => {
          if (!translationCache.current.has(textContent)) {
            textsToTranslate.push(textContent);
            elementMap.set(textContent, [...(elementMap.get(textContent) || []), { element, type, attribute }]);
          } else {
            // Use cached translation
            const cachedTranslation = translationCache.current.get(textContent);
            
            if (type === 'text') {
              element.textContent = cachedTranslation;
            } else if (type === 'attribute') {
              element.setAttribute(attribute, cachedTranslation);
            } else if (type === 'innerHTML') {
              element.innerHTML = cachedTranslation;
            }
          }
        });

        if (textsToTranslate.length) {
          // Translate batch
          const translations = await batchTranslate(textsToTranslate);
          
          // Update elements and cache
          Object.entries(translations).forEach(([original, translated]) => {
            translationCache.current.set(original, translated);
            const elements = elementMap.get(original);
            if (elements) {
              elements.forEach(({ element, type, attribute }) => {
                if (document.contains(element)) {
                  if (type === 'text') {
                    storeOriginalContent(element, element.textContent, 'text');
                    element.textContent = translated;
                  } else if (type === 'attribute') {
                    storeOriginalContent(element, element.getAttribute(attribute), 'attribute', attribute);
                    element.setAttribute(attribute, translated);
                  } else if (type === 'innerHTML') {
                    storeOriginalContent(element, element.innerHTML, 'innerHTML');
                    element.innerHTML = translated;
                  }
                }
              });
            }
          });
        }

        // Wait a bit before processing next batch to prevent UI freezing
        await new Promise(resolve => setTimeout(resolve, BATCH_DELAY));
      }
    } finally {
      isProcessing.current = false;
      setIsTranslating(false);
    }
  }, [language]);

  // Handle dynamic content changes
  const handleContentChanges = useCallback((mutations) => {
    if (language === 'en') return;

    mutations.forEach(mutation => {
      if (mutation.type === 'childList' && mutation.addedNodes.length) {
        mutation.addedNodes.forEach(node => {
          if (node.nodeType !== Node.ELEMENT_NODE && node.nodeType !== Node.TEXT_NODE) return;

          const processTextNode = (textNode) => {
            const text = textNode.textContent.trim();
            if (text) {
              pendingTranslations.current.push({
                element: textNode,
                textContent: text,
                type: 'text'
              });
            }
          };

          // Function to process element attributes
          const processElementAttributes = (element) => {
            ['placeholder', 'title', 'aria-label', 'alt'].forEach(attr => {
              if (!element.hasAttribute(attr)) return;
              const value = element.getAttribute(attr)?.trim();
              if (!value) return;

              pendingTranslations.current.push({
                element: element,
                textContent: value,
                type: 'attribute',
                attribute: attr
              });
            });

            // Check for innerHTML of certain safe elements
            const safeInnerHTMLElements = ['div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
            if (safeInnerHTMLElements.includes(element.tagName.toLowerCase())) {
              const innerHTML = element.innerHTML.trim();
              if (innerHTML && !innerHTML.includes('<')) {
                pendingTranslations.current.push({
                  element: element,
                  textContent: innerHTML,
                  type: 'innerHTML'
                });
              }
            }
          };

          // If it's a text node, process it directly
          if (node.nodeType === Node.TEXT_NODE) {
            processTextNode(node);
            return;
          }

          // If it's an element, walk through its text nodes
          const walker = document.createTreeWalker(
            node,
            NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT,
            {
              acceptNode: (node) => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                  const tag = node.tagName.toLowerCase();
                  if (['script', 'style', 'code', 'pre'].includes(tag)) {
                    return NodeFilter.FILTER_REJECT;
                  }
                  return NodeFilter.FILTER_ACCEPT;
                }
                return node.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
              }
            }
          );

          let currentNode;
          while (currentNode = walker.nextNode()) {
            if (currentNode.nodeType === Node.TEXT_NODE) {
              processTextNode(currentNode);
            } else if (currentNode.nodeType === Node.ELEMENT_NODE) {
              processElementAttributes(currentNode);
            }
          }
        });
      }
    });

    // Start processing queue if not already processing
    if (!isProcessing.current) {
      processTranslationQueue();
    }
  }, [language, processTranslationQueue]);

  // Initialize mutation observer
  useEffect(() => {
    if (language === 'en') return;

    const observer = new MutationObserver(handleContentChanges);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true
    });

    // Initial translation of existing content
    handleContentChanges([{
      type: 'childList',
      addedNodes: [document.body],
      removedNodes: []
    }]);

    return () => observer.disconnect();
  }, [language, handleContentChanges]);

  // Function to translate content programmatically
  const translateContent = useCallback(async (text) => {
    if (language === 'en' || !text) return text;

    // Check cache first
    if (translationCache.current.has(text)) {
      return translationCache.current.get(text);
    }

    try {
      const translations = await batchTranslate([text]);
      const translated = translations[text] || text;
      translationCache.current.set(text, translated);
      return translated;
    } catch (err) {
      console.error('Translation error:', err);
      return text;
    }
  }, [language]);

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  return (
    <TranslationContext.Provider value={{ 
      language, 
      setLanguage: handleLanguageChange, 
      isTranslating,
      translateContent,
      error 
    }}>
      {children}
      {isTranslating && (
        <div className="fixed right-4 bottom-4 bg-white rounded-lg shadow-lg p-3 flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-gray-600">Translating...</span>
        </div>
      )}
      {error && (
        <div className="fixed bottom-4 right-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg shadow-lg">
          <p className="text-sm">{error}</p>
        </div>
      )}
    </TranslationContext.Provider>
  );
};

export const useTranslation = () => {
  const context = useContext(TranslationContext);
  if (!context) {
    throw new Error('useTranslation must be used within a TranslationProvider');
  }
  return context;
};