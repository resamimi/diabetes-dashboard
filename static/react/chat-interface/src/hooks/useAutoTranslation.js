import { useCallback } from 'react';

export const useAutoTranslation = () => {
  // Language code mapping for Azure Translator
  const languageMap = {
    'en': 'en',
    'sl': 'sl'  // Slovenian in Azure Translator
  };

  const translateText = useCallback(async (text, targetLang) => {
    if (!text || targetLang === 'en') return text;
    
    try {
      // Map the language code
      const apiLangCode = languageMap[targetLang];
      
      const response = await fetch('/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          to: apiLangCode
        })
      });

      if (!response.ok) {
        throw new Error(`Translation failed: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      return data.translation;
    } catch (error) {
      console.error('Translation error:', error);
      return text; // Fallback to original text
    }
  }, []);

  // Helper function to translate HTML content
  const translateHtml = useCallback(async (html, targetLang) => {
    if (!html || targetLang === 'en') return html;

    try {
      // Split HTML into text and tags
      const parts = html.split(/(<[^>]*>)/);
      
      // Translate only the text parts
      const translatedParts = await Promise.all(
        parts.map(async part => {
          if (part.startsWith('<') || !part.trim()) {
            return part;
          }
          return await translateText(part.trim(), targetLang);
        })
      );
      
      return translatedParts.join('');
    } catch (error) {
      console.error('HTML translation error:', error);
      return html;
    }
  }, [translateText]);

  return { translateText, translateHtml };
};