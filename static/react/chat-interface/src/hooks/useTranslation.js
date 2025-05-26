import { useCallback } from 'react';
import { translations } from '../translations';

export const useTranslation = (language) => {
  const t = useCallback((key) => {
    if (!translations[language] || !translations[language][key]) {
      console.warn(`Missing translation for key: ${key}`);
      return translations.en[key] || key;
    }
    return translations[language][key];
  }, [language]);

  return { t };
};