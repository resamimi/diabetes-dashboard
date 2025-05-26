import { useEffect, useCallback } from 'react';

const useActivityTracking = (userId) => {
  const logActivity = useCallback(async (activityType, details) => {
    // Skip logging if no userId
    if (!userId) {
      console.warn('Attempted to log activity without user ID');
      return;
    }

    try {
      // Ensure userId is a number
      const numericUserId = parseInt(userId, 10);
      if (isNaN(numericUserId)) {
        console.warn('Invalid user ID format');
        return;
      }

      const response = await fetch('/api/activity/log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: numericUserId,
          type: activityType,
          details: {
            ...details,
            timestamp: new Date().toISOString(),
            url: window.location.pathname,
            userAgent: navigator.userAgent,
            screenResolution: {
              width: window.screen.width,
              height: window.screen.height
            }
          }
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to log activity');
      }
    } catch (error) {
      // Log error but don't throw - activity logging should not break the app
      console.warn('Error logging activity:', error);
    }
  }, [userId]);

  // Track page views
  useEffect(() => {
    if (userId) {
      logActivity('page_view', {
        title: document.title,
        referrer: document.referrer
      });
    }
  }, [userId, logActivity]);

  // Track chat interactions
  const trackChatInteraction = useCallback((messageType, content) => {
    logActivity('chat_interaction', {
      messageType,
      content,
      messageLength: content.length
    });
  }, [logActivity]);

  // Track visualization interactions
  const trackVisualizationInteraction = useCallback((visualizationType, action, details = {}) => {
    logActivity('visualization_interaction', {
      visualizationType,
      action,
      ...details
    });
  }, [logActivity]);

  // Track clicks
  useEffect(() => {
    const handleClick = (e) => {
      const target = e.target;
      const elementType = target.tagName.toLowerCase();
      const elementId = target.id;
      const elementClass = target.className;
      const elementText = target.textContent?.slice(0, 100);

      logActivity('click', {
        elementType,
        elementId,
        elementClass,
        elementText,
        x: e.pageX,
        y: e.pageY
      });
    };

    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [logActivity]);

  // Track hovers
  useEffect(() => {
    let hoverTimeout;
    const handleHoverStart = (e) => {
      hoverTimeout = setTimeout(() => {
        const target = e.target;
        logActivity('hover', {
          elementType: target.tagName.toLowerCase(),
          elementId: target.id,
          elementClass: target.className,
          elementText: target.textContent?.slice(0, 100),
          x: e.pageX,
          y: e.pageY,
          duration: 500
        });
      }, 500);
    };

    const handleHoverEnd = () => {
      clearTimeout(hoverTimeout);
    };

    window.addEventListener('mouseover', handleHoverStart);
    window.addEventListener('mouseout', handleHoverEnd);
    
    return () => {
      window.removeEventListener('mouseover', handleHoverStart);
      window.removeEventListener('mouseout', handleHoverEnd);
      clearTimeout(hoverTimeout);
    };
  }, [logActivity]);

  // Track scrolling behavior
  useEffect(() => {
    let scrollTimeout;
    const handleScroll = () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        const scrollDepth = (window.pageYOffset + window.innerHeight) / document.documentElement.scrollHeight * 100;
        logActivity('scroll', {
          scrollDepth: Math.round(scrollDepth),
          scrollY: window.pageYOffset
        });
      }, 150);
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
      clearTimeout(scrollTimeout);
    };
  }, [logActivity]);

  return {
    trackChatInteraction,
    trackVisualizationInteraction
  };
};

export default useActivityTracking;