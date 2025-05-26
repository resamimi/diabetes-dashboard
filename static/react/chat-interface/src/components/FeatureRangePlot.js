import React, { useCallback, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Alert, AlertDescription } from './ui/alert';
import { HelpCircle } from 'lucide-react';
import { 
  useCitations, 
  CitationPopup, 
  extractReferences, 
  extractFeatureSection,
  FeatureHelpButton 
} from './ScientificExplanationUtils';
import useTimeTracking from '../hooks/useTimeTracking';

const RangeDot = ({ value, maxValue, isInfinite }) => {
  if (isInfinite) return null; 
  
  const percentage = (value / maxValue) * 100;
  return (
    <div 
      className="absolute w-3 h-3 rounded-full shadow-sm"
      style={{ 
        backgroundColor: 'currentColor',
        left: `${percentage}%`,
        marginLeft: '-6px',
        top: '50%',
        transform: 'translateY(-50%)',
        zIndex: 2
      }}
    />
  );
};

const ValueMarker = ({ value, maxValue }) => {
  if (value === undefined) return null;
  
  const percentage = Math.min((value / maxValue) * 100, 100);
  
  return (
    <div 
      className="absolute"
      style={{ 
        left: `${percentage}%`,
        zIndex: 3,
      }}
    >
      {/* Vertical line */}
      <div className="absolute w-0.5 h-16 bg-purple-500" style={{ bottom: '0', transform: 'translateX(-50%)' }} />
      
      {/* Dot */}
      <div className="absolute w-3 h-3 bg-purple-500 rounded-full" style={{ bottom: '0', transform: 'translate(-50%, 50%)' }} />
      
      {/* Value label */}
      <div 
        className="absolute text-xs font-semibold text-purple-600 whitespace-nowrap"
        style={{ 
          bottom: '-24px',
          transform: 'translateX(-50%)',
        }}
      >
        {value.toFixed(1)}
      </div>
    </div>
  );
};

const FeatureAxis = ({ 
  feature, 
  datacentricRange, 
  scientificRange,
  featureValue,
  onFeatureClick,
  timeTracking,
  isActive,
  onHover 
}) => {
  const isInf = value => value === 1000;
  
  const values = [...datacentricRange, ...scientificRange, featureValue]
    .filter(v => v !== undefined && !isInf(v));
  const maxValue = Math.max(...values) * 1.2;
  
  const getTickValues = (max) => {
    const step = Math.ceil(max / 5);
    return Array.from({ length: 5 }, (_, i) => Math.round(i * step));
  };

  const ticks = getTickValues(maxValue);
  const [dcLower, dcUpper] = datacentricRange;
  const [sciLower, sciUpper] = scientificRange;
  
  const dcLowerPct = (dcLower / maxValue) * 100;
  const dcUpperPct = isInf(dcUpper) ? 100 : (dcUpper / maxValue) * 100;
  const sciLowerPct = (sciLower / maxValue) * 100;
  const sciUpperPct = isInf(sciUpper) ? 100 : (sciUpper / maxValue) * 100;

  const handleRangeHover = (rangeType) => {
    timeTracking.logCurrentDuration(`range_hover_${rangeType}_${feature}`);
  };

  return (
    <div 
      className={`group mb-4 last:mb-0 bg-white px-4 pt-2 pb-6 rounded-lg border 
                  ${isActive ? 'border-blue-200 shadow-md' : 'border-gray-100 shadow-sm'} 
                  hover:shadow-md transition-all duration-200`}
      onMouseEnter={() => {
        onHover(true);
        timeTracking.logCurrentDuration(`feature_hover_start_${feature}`);
      }}
      onMouseLeave={() => {
        onHover(false);
        timeTracking.logCurrentDuration(`feature_hover_end_${feature}`);
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span className="text-sm font-medium text-gray-800">{feature}</span>
        <FeatureHelpButton 
          feature={feature} 
          onClick={() => {
            timeTracking.logCurrentDuration(`feature_help_click_${feature}`);
            onFeatureClick(feature);
          }} 
        />
      </div>

      <div className="relative">
        <div className="relative h-16">
          {ticks.map((value) => (
            <React.Fragment key={value}>
              <div
                className="absolute h-[calc(100%+4px)] w-px bg-gray-200/70"
                style={{
                  left: `${(value / maxValue) * 100}%`,
                  top: '-2px',
                }}
              />
              <div
                className="absolute text-xs font-medium text-gray-500"
                style={{
                  left: `${(value / maxValue) * 100}%`,
                  top: '100%',
                  transform: 'translateX(-50%)',
                  marginTop: '4px'
                }}
              >
                {value}
              </div>
            </React.Fragment>
          ))}

          <div
            className="absolute h-[calc(100%+4px)] w-px bg-gray-200/70"
            style={{
              right: 0,
              top: '-2px',
            }}
          />
          <div
            className="absolute text-xs font-medium text-gray-500"
            style={{
              right: 0,
              top: '100%',
              transform: 'translateX(50%)',
              marginTop: '4px'
            }}
          >
            inf.
          </div>

          {/* Data-centric range */}
          <div 
            className="relative h-6 mb-4"
            onMouseEnter={() => handleRangeHover('datacentric')}
            onMouseLeave={() => timeTracking.logCurrentDuration(`range_leave_datacentric_${feature}`)}
          >
            <div className="absolute w-full h-px bg-gray-200" style={{ top: '50%' }} />
            <div 
              className={`absolute h-2 bg-indigo-400/80 hover:bg-indigo-400 transition-colors
                         ${isInf(dcUpper) ? 'rounded-r-none' : 'rounded-full'}`}
              style={{
                left: `${dcLowerPct}%`,
                width: `${dcUpperPct - dcLowerPct}%`,
                top: '50%',
                transform: 'translateY(-50%)'
              }}
            />
            <div className="absolute inset-0 text-indigo-400">
              <RangeDot value={dcLower} maxValue={maxValue} isInfinite={isInf(dcLower)} />
              <RangeDot value={dcUpper} maxValue={maxValue} isInfinite={isInf(dcUpper)} />
            </div>
            
            {/* Data-centric range labels */}
            <div 
              className="absolute text-xs font-medium text-indigo-600 transition-opacity opacity-80 group-hover:opacity-100"
              style={{ 
                left: `${dcLowerPct}%`, 
                top: '-16px',
                transform: 'translateX(-50%)'
              }}
            >
              {dcLower}
            </div>
            <div 
              className="absolute text-xs font-medium text-indigo-600 transition-opacity opacity-80 group-hover:opacity-100"
              style={{ 
                left: !isInf(dcUpper) ? `${dcUpperPct}%` : '100%', 
                top: '-16px',
                transform: `translateX(${!isInf(dcUpper) ? '-50%' : '0'})`,
                right: isInf(dcUpper) ? 0 : 'auto'
              }}
            >
              {isInf(dcUpper) ? 'inf.' : dcUpper}
            </div>
          </div>

          {/* Scientific range */}
          <div 
            className="relative h-6"
            onMouseEnter={() => handleRangeHover('scientific')}
            onMouseLeave={() => timeTracking.logCurrentDuration(`range_leave_scientific_${feature}`)}
          >
            <div className="absolute w-full h-px bg-gray-200" style={{ top: '50%' }} />
            <div 
              className={`absolute h-2 bg-amber-400/80 hover:bg-amber-400 transition-colors
                         ${isInf(sciUpper) ? 'rounded-r-none' : 'rounded-full'}`}
              style={{
                left: `${sciLowerPct}%`,
                width: `${sciUpperPct - sciLowerPct}%`,
                top: '50%',
                transform: 'translateY(-50%)'
              }}
            />
            <div className="absolute inset-0 text-amber-400">
              <RangeDot value={sciLower} maxValue={maxValue} isInfinite={isInf(sciLower)} />
              <RangeDot value={sciUpper} maxValue={maxValue} isInfinite={isInf(sciUpper)} />
            </div>
            
            {/* Scientific range labels */}
            <div 
              className="absolute text-xs font-medium text-amber-600 transition-opacity opacity-80 group-hover:opacity-100"
              style={{ 
                left: `${sciLowerPct}%`, 
                top: '-16px',
                transform: 'translateX(-50%)'
              }}
            >
              {sciLower}
            </div>
            <div 
              className="absolute text-xs font-medium text-amber-600 transition-opacity opacity-80 group-hover:opacity-100"
              style={{ 
                left: !isInf(sciUpper) ? `${sciUpperPct}%` : '100%', 
                top: '-16px',
                transform: `translateX(${!isInf(sciUpper) ? '-50%' : '0'})`,
                right: isInf(sciUpper) ? 0 : 'auto'
              }}
            >
              {isInf(sciUpper) ? 'inf.' : sciUpper}
            </div>
          </div>

          {/* Current value marker */}
          <ValueMarker value={featureValue} maxValue={maxValue} />
        </div>
      </div>
    </div>
  );
};

const FeatureRangePlot = ({ 
  visualizationData, 
  className,
  onFeatureClick,
  userId 
}) => {
  const {
    activeCitationData,
    setReferences,
    handlePopupMouseEnter,
    handlePopupMouseLeave
  } = useCitations();

  // Initialize time tracking
  const timeTracking = useTimeTracking(userId, 'FeatureRangePlot', {
    inactivityThreshold: 180000, // 3 minutes
    logInterval: 60000, // Log every minute
    minTimeToLog: 10000 // Minimum 10 seconds
  });

  // Active feature state management
  const [activeFeature, setActiveFeature] = useState(null);

  // Handle feature interactions
  const handleFeatureClick = useCallback((feature) => {
    if (!visualizationData?.data?.report) {
      console.log('Warning: No report data available for feature:', feature);
      return;
    }

    const featureSection = extractFeatureSection(visualizationData.data.report, feature, 'feature_range');
    if (!featureSection) {
      console.log('Warning: No section found for feature:', feature);
      return;
    }

    timeTracking.logCurrentDuration(`feature_click_${feature}`);

    const question = `What are the scientific guidelines for ${feature} in diabetes risk assessment?`;
    onFeatureClick(feature, question, featureSection, 'feature_range');
  }, [visualizationData, onFeatureClick, timeTracking]);

  // Handle help button clicks
  const handleHelpClick = useCallback((helpType) => {
    timeTracking.logCurrentDuration(`help_click_${helpType}`);
    
    const helpContent = {
      'data-centric-help': {
        question: "What is a AI-observed range and how is it determined?",
        answer: `<p>An AI-observed range represents the typical value range for a feature among patients who received the same prediction ("${visualizationData.data.class_label}") from the AI model.</p>
                 <p>This range helps understand what values are commonly associated with this specific prediction.</p>`
      },
      'scientific-help': {
        question: "What is a scientific range and what does it tell us?",
        answer: `<p>A scientific range represents evidence-based thresholds established through medical research.</p>
                 <p>These ranges are derived from scientific papers and clinical guidelines.</p>`
      }
    };

    const content = helpContent[helpType];
    onFeatureClick(helpType, content.question, content.answer, 'feature_range');
  }, [timeTracking, visualizationData, onFeatureClick]);

  // Extract references when visualization data changes
  useEffect(() => {
    if (visualizationData?.data?.report) {
      const refs = extractReferences(visualizationData.data.report);
      setReferences(refs);
    }
  }, [visualizationData, setReferences]);

  if (!visualizationData?.data) {
    return <div>Missing visualization data</div>;
  }

  const { 
    feature_names: features,
    datacentric_ranges: datacentricRanges,
    scientific_ranges: scientificRanges,
    class_label: classLabel,
    feature_values: featureValues
  } = visualizationData.data;

  return (
    <div 
      className={className}
      onMouseEnter={() => timeTracking.resumeTracking()}
      onMouseLeave={() => timeTracking.pauseTracking()}
    >
      <Alert className="bg-blue-50 border-blue-200">
        <AlertDescription data-translate="true" className="text-blue-800">
          For the most important health measures, compares AI-observed ranges (from AI assessments) with scientific ranges (from medical research). For more details about the ranges click on the help buttons next the labels. The comparison shows how much AI findings align with established medical guidelines.
        </AlertDescription>
      </Alert>

      <div className="w-full max-w-2xl mx-auto p-2 bg-white rounded-lg shadow-lg">
        <div className="flex flex-wrap items-center gap-4 mb-3">
        <div data-translate="true" className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-indigo-400" />
            <span className="text-sm text-gray-600">AI-observed Range</span>
              <button
                onClick={() => {
                  timeTracking.logCurrentDuration('data_centric_help_click');
                  handleHelpClick('data-centric-help');
                }}
                className="p-0.5 rounded-full hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition-colors"
                title="Learn about AI-observed ranges"
              >
              <HelpCircle className="w-4 h-4 text-blue-500 hover:text-blue-600" />
            </button>
          </div>
          <div data-translate="true" className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-amber-400" />
            <span className="text-sm text-gray-600">Scientific Range</span>
            <button
              onClick={() => {
                timeTracking.logCurrentDuration('scientific_help_click');
                handleHelpClick('scientific-help');
              }}
              className="p-0.5 rounded-full hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition-colors"
              title="Learn about scientific ranges"
            >
              <HelpCircle className="w-4 h-4 text-blue-500 hover:text-blue-600" />
            </button>
          </div>
          {featureValues && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-purple-500" />
              <span className="text-xs text-gray-600">Patient Factor Value</span>
            </div>
          )}
        </div>

        <div 
          className="space-y-2"
          onScroll={() => timeTracking.logCurrentDuration('feature_list_scroll')}
        >
          {features.map((feature, index) => (
            <FeatureAxis
              key={feature}
              feature={feature}
              datacentricRange={datacentricRanges[index]}
              scientificRange={scientificRanges[index]}
              featureValue={featureValues?.[index]}
              onFeatureClick={() => handleFeatureClick(feature)}
              timeTracking={timeTracking}
              isActive={activeFeature === feature}
              onHover={(isHovered) => setActiveFeature(isHovered ? feature : null)}
            />
          ))}
        </div>
      </div>

      {activeCitationData && createPortal(
        <CitationPopup
          citation={activeCitationData.citation}
          style={activeCitationData.style}
          onMouseEnter={() => {
            handlePopupMouseEnter();
            timeTracking.logCurrentDuration('citation_popup_hover_start');
          }}
          onMouseLeave={() => {
            handlePopupMouseLeave();
            timeTracking.logCurrentDuration('citation_popup_hover_end');
          }}
        />,
        document.body
      )}
    </div>
  );
};

export default FeatureRangePlot;
