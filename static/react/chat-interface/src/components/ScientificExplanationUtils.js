import React, { useEffect, useState, useRef } from 'react';
import { HelpCircle } from 'lucide-react';

// Citation popup component that can be reused
export const CitationPopup = ({ citation, onMouseEnter, onMouseLeave, style }) => {
  return (
    <div 
      className="fixed bg-white rounded-md shadow-lg border border-gray-200 text-xs citation-popup z-50"
      style={{
        ...style,
        width: '300px',
        transform: 'translateY(8px)'
      }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div 
        className="absolute -top-2 left-[28px] w-4 h-4 bg-white border-t border-l border-gray-200 transform rotate-45"
      />
      
      <div className="relative p-2.5">
        <div className="text-gray-600 pr-4 leading-relaxed">{citation}</div>
      </div>
    </div>
  );
};

// Hook to manage citations and their interactions
export const useCitations = () => {
  const [activeCitationData, setActiveCitationData] = useState(null);
  const [references, setReferences] = useState({});
  const timeoutRef = useRef(null);

  const clearTimeout = () => {
    if (timeoutRef.current) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  };

  const startCloseTimeout = () => {
    clearTimeout();
    timeoutRef.current = window.setTimeout(() => {
      setActiveCitationData(null);
    }, 300);
  };

  const handleCitationMouseEnter = (event, citationNumber) => {
    clearTimeout();
    
    const citation = references[citationNumber];
    if (!citation) return;

    const rect = event.target.getBoundingClientRect();
    const citationCenterX = rect.left + (rect.width / 2);
    const popupLeftOffset = citationCenterX - 28;
    
    setActiveCitationData({
      citation,
      style: {
        position: 'fixed',
        top: rect.bottom,
        left: popupLeftOffset,
      }
    });
  };

  useEffect(() => {
    const handleMouseOver = (e) => {
      const citationElement = e.target.closest('.citation-link');
      const popupElement = e.target.closest('.citation-popup');
      
      if (citationElement) {
        const citationNumber = parseInt(citationElement.dataset.citation);
        handleCitationMouseEnter(e, citationNumber);
      } else if (popupElement) {
        clearTimeout();
      }
    };

    const handleMouseOut = (e) => {
      const citationElement = e.target.closest('.citation-link');
      const popupElement = e.target.closest('.citation-popup');
      const toElement = e.relatedTarget;

      if (citationElement && (toElement?.closest('.citation-popup'))) return;
      if (popupElement && (toElement?.closest('.citation-link'))) return;

      if ((citationElement || popupElement) && 
          !toElement?.closest('.citation-link') && 
          !toElement?.closest('.citation-popup')) {
        startCloseTimeout();
      }
    };

    document.addEventListener('mouseover', handleMouseOver);
    document.addEventListener('mouseout', handleMouseOut);
    
    return () => {
      document.removeEventListener('mouseover', handleMouseOver);
      document.removeEventListener('mouseout', handleMouseOut);
      clearTimeout();
    };
  }, [references]);

  return {
    activeCitationData,
    setReferences,
    handlePopupMouseEnter: clearTimeout,
    handlePopupMouseLeave: startCloseTimeout
  };
};

// Extract references from report text
export const extractReferences = (report) => {
  const refsMatch = report.match(/<ol class='references'>(.*?)<\/ol>/s);
  if (refsMatch) {
    const refsHTML = refsMatch[1];
    const refItems = refsHTML.match(/<li>(.*?)<\/li>/g);
    
    if (refItems) {
      const refsObj = {};
      refItems.forEach((item, index) => {
        const refText = item.replace(/<li>|<\/li>/g, '');
        refsObj[index + 1] = refText;
      });
      return refsObj;
    }
  }
  return {};
};

// Convert citation numbers to hoverable spans
export const modifyTextWithHoverableCitations = (text) => {
  return text.replace(/\[(\d+)\]/g, (match, number) => {
    return `<span class="citation-link inline-flex items-center text-blue-600 cursor-pointer hover:text-blue-800 hover:bg-blue-50 px-1 rounded transition-colors" data-citation="${number}">${match}</span>`;
  });
};

// Extract and format feature section from report
export const extractFeatureSection = (report, feature, visualizationType) => {
  let match;
  
  if (visualizationType === 'feature_range') {
    // For feature range plot, content is wrapped in div tags
    const rangeRegex = new RegExp(`<h4>${feature}</h4>\\s*<div>(.*?)</div>`, 's');
    match = report.match(rangeRegex);
  } else {
    // For feature importance plot, content is between h4 tags
    const importanceRegex = new RegExp(`<h4>${feature}</h4>(.*?)(?=<h4>|$)`, 's');
    match = report.match(importanceRegex);
  }
  
  if (!match) {
    console.log(`No match found for feature ${feature} in ${visualizationType} report`);
    return null;
  }
  
  const sectionText = match[1];
  
  // Try to find a direct quote first
  const directQuotePattern = /<b>Direct quote<\/b>[^']+'([^']+)'/;
  const directQuoteMatch = sectionText.match(directQuotePattern);
  
  if (directQuoteMatch) {
    const introSentence = sectionText.split('<br>')[0];
    const fullQuote = sectionText.match(/<b>Direct quote<\/b>.*?'[^']+'/)[0];
    const modifiedIntro = modifyTextWithHoverableCitations(introSentence);
    return `<p>${modifiedIntro}<br><br><i><b>Research evidence:</b> ${fullQuote}</i></p>`;
  }
  
  const interpretationPattern = /<b>Interpretation<\/b>[^<]+/;
  const interpretationMatch = sectionText.match(interpretationPattern);
  
  if (interpretationMatch) {
    const introSentence = sectionText.split('<br>')[0];
    const fullText = sectionText.split(/<br><i>/)[1]?.split('</i>')[0];
    const modifiedIntro = modifyTextWithHoverableCitations(introSentence);
    if (fullText) {
      return `<p>${modifiedIntro}<br><br><i><b>Research evidence:</b> ${fullText}</i></p>`;
    }
  }
  
  const firstSentence = sectionText.split('.')[0] + '.';
  return `<p>${modifyTextWithHoverableCitations(firstSentence)}</p>`;
};

// Helper component for feature help button
export const FeatureHelpButton = ({ feature, onClick }) => (
  <button
    onClick={() => onClick(feature)}
    className="p-0.5 rounded-full hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition-colors"
    title={`Show details about ${feature}`}
  >
    <HelpCircle className="w-4 h-4 text-blue-500 hover:text-blue-600" />
  </button>
);