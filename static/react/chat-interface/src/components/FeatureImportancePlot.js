import React, { useCallback, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Alert, AlertDescription } from './ui/alert';
import { 
  useCitations, 
  CitationPopup, 
  extractReferences, 
  extractFeatureSection,
  FeatureHelpButton 
} from './ScientificExplanationUtils';
import useTimeTracking from '../hooks/useTimeTracking';

const getColorScale = (value, isUnlikelyDiabetes) => {
  // Increase intensity by using value squared for more dramatic scaling
  const normalizedValue = Math.pow(Math.abs(value) / 100, 0.7); // Use power of 0.7 for stronger contrast
  
  if (isUnlikelyDiabetes) {
    if (value >= 0) {
      // Make greens more vibrant for positive values
      return {
        background: `rgb(${Math.round(14 + (normalizedValue * 40))}, ${Math.round(197 - (normalizedValue * 100))}, ${Math.round(94 - (normalizedValue * 70))})`,
        hover: `rgb(${Math.round(14 + (normalizedValue * 30))}, ${Math.round(197 - (normalizedValue * 110))}, ${Math.round(94 - (normalizedValue * 100))})`
      };
    } else {
      // Make oranges more intense for negative values
      return {
        background: `rgb(${Math.round(251 - (normalizedValue * 20))}, ${Math.round(146 - (normalizedValue * 100))}, ${Math.round(40 + (normalizedValue * 20))})`,
        hover: `rgb(${Math.round(251 - (normalizedValue * 30))}, ${Math.round(146 - (normalizedValue * 110))}, ${Math.round(40 + (normalizedValue * 30))})`
      };
    }
  } else {
    if (value >= 0) {
      // Make oranges more intense for positive values
      return {
        background: `rgb(${Math.round(251 - (normalizedValue * 20))}, ${Math.round(146 - (normalizedValue * 100))}, ${Math.round(40 + (normalizedValue * 20))})`,
        hover: `rgb(${Math.round(251 - (normalizedValue * 30))}, ${Math.round(146 - (normalizedValue * 110))}, ${Math.round(40 + (normalizedValue * 30))})`
      };
    } else {
      // Make greens more vibrant for negative values
      return {
        background: `rgb(${Math.round(14 + (normalizedValue * 40))}, ${Math.round(197 - (normalizedValue * 100))}, ${Math.round(94 - (normalizedValue * 70))})`,
        hover: `rgb(${Math.round(14 + (normalizedValue * 30))}, ${Math.round(197 - (normalizedValue * 110))}, ${Math.round(94 - (normalizedValue * 100))})`
      };
    }
  }
};

const FeatureImportancePlot = ({ 
  visualizationData, 
  className,
  onFeatureClick,
  userId // Add userId prop
}) => {
  const {
    activeCitationData,
    setReferences,
    handlePopupMouseEnter,
    handlePopupMouseLeave
  } = useCitations();

  // Initialize time tracking with visualization-specific settings
  const timeTracking = useTimeTracking(userId, 'FeatureImportancePlot', {
    inactivityThreshold: 180000, // 3 minutes
    logInterval: 60000, // Log every minute
    minTimeToLog: 10000 // Minimum 10 seconds
  });
  
  // Track hover time for each feature
  const [hoveredFeature, setHoveredFeature] = useState(null);

  // Extract references when visualization data changes
  useEffect(() => {
    if (visualizationData?.data?.report) {
      const refs = extractReferences(visualizationData.data.report);
      setReferences(refs);
    }
  }, [visualizationData, setReferences]);

  // Handle feature click with time tracking
  const handleFeatureClick = useCallback((feature) => {
    if (!visualizationData?.data?.report) {
      console.log('Warning: No report data available for feature:', feature);
      return;
    }

    const featureSection = extractFeatureSection(visualizationData.data.report, feature, 'feature_importance');
    if (!featureSection) {
      console.log('Warning: No section found for feature:', feature);
      return;
    }

    const question = `What's the scientific evidence about the importance of ${feature} in diabetes assessment?`;

    // Log feature interaction time
    timeTracking.logCurrentDuration(`feature_click_${feature}`);

    if (onFeatureClick) {
      onFeatureClick(feature, question, featureSection, 'feature_importance');
    }
  }, [visualizationData, onFeatureClick, timeTracking]);

  // Handle feature hover start
  const handleFeatureHoverStart = useCallback((feature) => {
    setHoveredFeature(feature);
    timeTracking.logCurrentDuration(`feature_hover_start_${feature}`);
  }, [timeTracking]);

  // Handle feature hover end
  const handleFeatureHoverEnd = useCallback((feature) => {
    setHoveredFeature(null);
    timeTracking.logCurrentDuration(`feature_hover_end_${feature}`);
  }, [timeTracking]);

  if (!visualizationData?.data?.Feature || 
      !visualizationData?.data?.Importance ||
      !visualizationData?.data?.ClassName) {
    return <div>Missing required visualization data</div>;
  }

  const data = visualizationData.data.Feature.map((feature, index) => ({
    feature: feature,
    value: visualizationData.data.Importance[index]
  }));

  const maxValue = Math.max(...data.map(d => Math.abs(d.value)));
  const scale = 200 / maxValue;

  const isUnlikelyDiabetes = visualizationData.data.ClassName.toLowerCase().includes('unlikely');
  const visualizationOffset = isUnlikelyDiabetes ? "translate-x-10" : "-translate-x-10";

  const getBarColor = (value) => {
    if (isUnlikelyDiabetes) {
      return value >= 0 
        ? 'bg-green-600 hover:bg-green-700'
        : 'bg-orange-400 hover:bg-orange-500';
    } else {
      return value >= 0 
        ? 'bg-orange-400 hover:bg-orange-500'
        : 'bg-green-600 hover:bg-green-700';
    }
  };

  return (
    <div 
      className={className}
      onMouseEnter={() => timeTracking.resumeTracking()}
      onMouseLeave={() => timeTracking.pauseTracking()}
    >
      <Alert className="bg-blue-50 border-blue-200">
        <AlertDescription className="text-blue-800" data-translate="true">
          Shows how each health factor influences the AI's assessment: "{visualizationData.data.ClassName}".
           Factors with orange bars influence in increasing diabetes risk and the ones with green bars influence in decreasing diabetes risk.
           The length of each bar shows how strongly that factor affects the assessment.
        </AlertDescription>
      </Alert>

      <div className="w-full max-w-2xl mx-auto p-4 mt-4 bg-white rounded-lg shadow-lg">
        <div className="relative">
          <div className={`absolute inset-0 ${visualizationOffset}`}>
            <div className="absolute left-[60%] top-0 bottom-0 w-0.5 bg-gray-300 shadow-sm" />
            
            <div className="text-center text-sm font-medium">
              <span className="absolute left-[25%] top-0 text-gray-600 hover:text-gray-800 transition-colors">
                Against Assessment
              </span>
              <span className="absolute left-[62%] top-0 text-gray-600 hover:text-gray-800 transition-colors">
                Towards Assessment
              </span>
            </div>

            <div className="absolute inset-0 pt-8">
              {data.map((item, index) => {
                const barColors = getColorScale(item.value, isUnlikelyDiabetes);
                return (
                  <div 
                    key={index} 
                    className="h-7 flex items-center"
                    onMouseEnter={() => handleFeatureHoverStart(item.feature)}
                    onMouseLeave={() => handleFeatureHoverEnd(item.feature)}
                  >
                    <div
                      className={`absolute h-5 ${
                        item.value >= 0 ? 'left-[60%]' : 'right-[40%]'
                      } transition-all duration-200 shadow-sm ${
                        item.value >= 0 ? 'rounded-r-sm' : 'rounded-l-sm'
                      } ${
                        hoveredFeature === item.feature ? 'ring-2 ring-blue-300' : ''
                      }`}
                      style={{
                        width: `${Math.abs(item.value * scale)}px`,
                        backgroundColor: barColors.background,
                        '--hover-color': barColors.hover,
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = barColors.hover;
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = barColors.background;
                      }}
                    >
                      {Math.abs(item.value * scale) >= 40 ? (
                        <span
                          className={`absolute ${
                            item.value >= 0 ? 'right-2' : 'left-2'
                          } top-1/2 -translate-y-1/2 text-xs font-semibold text-white whitespace-nowrap`}
                        >
                          {item.value >= 0 ? `+${item.value}%` : `${item.value}%`}
                        </span>
                      ) : (
                        <span
                          className={`absolute ${
                            item.value >= 0 ? 'left-full pl-1' : 'right-full pr-1'
                          } top-1/2 -translate-y-1/2 text-xs font-semibold text-gray-700 whitespace-nowrap`}
                        >
                          {item.value >= 0 ? `+${item.value}%` : `${item.value}%`}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="relative pt-8">
            {data.map((item, index) => (
              <div 
                key={index} 
                className="h-7 flex items-center"
              >
                <div className="absolute left-1 flex items-center gap-0.5">
                  <span className="text-gray-700 text-sm font-medium group-hover:text-gray-900 transition-colors">
                    {item.feature}
                  </span>
                  <FeatureHelpButton 
                    feature={item.feature}
                    onClick={() => handleFeatureClick(item.feature)}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {activeCitationData && createPortal(
        <CitationPopup
          citation={activeCitationData.citation}
          style={activeCitationData.style}
          onMouseEnter={handlePopupMouseEnter}
          onMouseLeave={handlePopupMouseLeave}
        />,
        document.body
      )}
    </div>
  );
};

export default FeatureImportancePlot;

