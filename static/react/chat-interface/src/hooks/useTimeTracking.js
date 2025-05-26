import { useEffect, useRef, useCallback } from 'react';

const useTimeTracking = (userId, componentName, options = {}) => {
  const startTime = useRef(Date.now());
  const isActive = useRef(true);
  const lastInteractionTime = useRef(Date.now());
  const inactivityTimeout = useRef(null);
  const periodicLogTimeout = useRef(null);

  // Get options with defaults
  const {
    inactivityThreshold = 60000, // 1 minute of inactivity before pausing
    logInterval = 30000, // Log every 30 seconds while active
    minTimeToLog = 5000 // Minimum 5 seconds to log an interaction
  } = options;

  const logTimeSpent = useCallback(async (duration, reason = 'interval') => {
    if (duration < minTimeToLog) return;

    try {
      await fetch('/api/activity/log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId,
          type: 'time_spent',
          details: {
            componentName,
            duration,
            reason,
            timestamp: new Date().toISOString(),
            sessionData: {
              startTime: new Date(startTime.current).toISOString(),
              endTime: new Date().toISOString(),
              isActive: isActive.current
            }
          }
        })
      });
    } catch (error) {
      console.error('Error logging time spent:', error);
    }
  }, [userId, componentName, minTimeToLog]);

  // Handle user interactions
  const handleInteraction = useCallback(() => {
    const now = Date.now();
    
    // If component was inactive, log the previous inactive period
    if (!isActive.current) {
      const inactiveDuration = now - lastInteractionTime.current;
      logTimeSpent(inactiveDuration, 'inactive_period');
      isActive.current = true;
      startTime.current = now;
    }

    lastInteractionTime.current = now;

    // Clear and reset inactivity timeout
    if (inactivityTimeout.current) {
      clearTimeout(inactivityTimeout.current);
    }

    inactivityTimeout.current = setTimeout(() => {
      if (isActive.current) {
        const activeDuration = Date.now() - startTime.current;
        logTimeSpent(activeDuration, 'inactivity_detected');
        isActive.current = false;
        startTime.current = Date.now();
      }
    }, inactivityThreshold);
  }, [logTimeSpent, inactivityThreshold]);

  // Set up periodic logging
  useEffect(() => {
    // Initial setup
    startTime.current = Date.now();
    lastInteractionTime.current = Date.now();
    isActive.current = true;

    // Set up periodic logging
    const logPeriodically = () => {
      if (isActive.current) {
        const duration = Date.now() - startTime.current;
        logTimeSpent(duration, 'periodic');
      }
    };

    periodicLogTimeout.current = setInterval(logPeriodically, logInterval);

    // Add interaction listeners
    const interactionEvents = ['mousedown', 'keydown', 'mousemove', 'touchstart', 'scroll', 'wheel'];
    interactionEvents.forEach(event => {
      document.addEventListener(event, handleInteraction);
    });

    // Cleanup function
    return () => {
      // Log final duration before unmounting
      const finalDuration = Date.now() - startTime.current;
      logTimeSpent(finalDuration, 'component_unmount');

      // Clear timeouts and intervals
      clearInterval(periodicLogTimeout.current);
      if (inactivityTimeout.current) {
        clearTimeout(inactivityTimeout.current);
      }

      // Remove event listeners
      interactionEvents.forEach(event => {
        document.removeEventListener(event, handleInteraction);
      });
    };
  }, [logTimeSpent, handleInteraction, logInterval]);

  // Return methods to manually control tracking
  return {
    pauseTracking: () => {
      if (isActive.current) {
        const duration = Date.now() - startTime.current;
        logTimeSpent(duration, 'manual_pause');
        isActive.current = false;
        startTime.current = Date.now();
      }
    },
    resumeTracking: () => {
      if (!isActive.current) {
        startTime.current = Date.now();
        isActive.current = true;
        handleInteraction();
      }
    },
    logCurrentDuration: (reason = 'manual') => {
      const duration = Date.now() - startTime.current;
      logTimeSpent(duration, reason);
      startTime.current = Date.now();
    }
  };
};

export default useTimeTracking;