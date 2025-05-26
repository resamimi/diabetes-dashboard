import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Card, CardContent } from './ui/card';
import { ChevronRight, Brain, TrendingDown, ArrowRight, AlertCircle } from 'lucide-react';
import useTimeTracking from '../hooks/useTimeTracking';

const FeasibilityBadge = ({ level }) => {
  const colors = {
    'Very Easy': 'bg-green-50 text-green-700 border-green-200',
    'Easy': 'bg-green-100 text-green-700 border-green-300',
    'Moderate': 'bg-yellow-50 text-yellow-700 border-yellow-200',
    'Challenging': 'bg-orange-50 text-orange-700 border-orange-200',
    'Very Challenging': 'bg-red-50 text-red-700 border-red-200',
    'Difficult': 'bg-red-100 text-red-800 border-red-300'
  };

  return (
    <div className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${colors[level]} border shadow-sm`}>
      {level}
    </div>
  );
};

const TimelineStep = ({
  step,
  isLast,
  onGetRecommendations,
  timeTracking,
  isActive,
  onHover,
  stepIndex
}) => {
  const riskReduction = step.risk_reduction.toFixed(1);
  const hasMinimalProgress = parseFloat(riskReduction) < 0.1;

  // Track feature hover states
  const [hoveredFeature, setHoveredFeature] = useState(null);
  
  const handleFeatureHover = useCallback((feature, isEntering) => {
    setHoveredFeature(isEntering ? feature : null);
    timeTracking.logCurrentDuration(
      isEntering ? 
      `feature_hover_start_${feature}_step${stepIndex}` : 
      `feature_hover_end_${feature}_step${stepIndex}`
    );
  }, [timeTracking, stepIndex]);

  // Track value changes focus
  const handleValueFocus = useCallback((feature, isStart) => {
    timeTracking.logCurrentDuration(
      `value_comparison_${isStart ? 'start' : 'end'}_${feature}_step${stepIndex}`
    );
  }, [timeTracking, stepIndex]);

  return (
    <div 
      className={`relative pb-1 ${isActive ? 'z-10' : 'z-0'}`}
      onMouseEnter={() => {
        onHover(true);
        timeTracking.logCurrentDuration(`step_hover_start_${stepIndex}`);
      }}
      onMouseLeave={() => {
        onHover(false);
        timeTracking.logCurrentDuration(`step_hover_end_${stepIndex}`);
      }}
    >
      {!isLast && (
        <div className="absolute left-4 top-4 h-full w-px bg-blue-100" />
      )}
      
      <div className={`relative flex items-start gap-4 transition-all duration-200
                    ${isActive ? 'transform scale-[1.02]' : ''}`}>
        <div className="h-8 w-8 rounded-full bg-white border border-blue-200 flex items-center justify-center z-10">
          <span className="text-sm font-medium text-blue-500">{stepIndex + 1}</span>
        </div>

        <div className={`flex-1 bg-white rounded-lg border transition-all duration-200
                        ${isActive ? 'border-blue-200 shadow-md' : 'border-gray-100 shadow-sm'}`}>
          <div className="p-2">
            <div className="flex flex-col gap-2">
              {/* Changes and progress/risk reduction */}
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-1">
                  {Object.entries(step.changes).map(([feature, change]) => (
                    <div 
                      key={feature} 
                      className={`flex items-center text-sm transition-colors duration-200
                                ${hoveredFeature === feature ? 'bg-blue-50 rounded-md px-2' : ''}`}
                      onMouseEnter={() => handleFeatureHover(feature, true)}
                      onMouseLeave={() => handleFeatureHover(feature, false)}
                    >
                      <span className="font-medium text-gray-700">{feature}:</span>
                      <span 
                        className="ml-1 text-gray-600"
                        onMouseEnter={() => handleValueFocus(feature, true)}
                        onMouseLeave={() => handleValueFocus(feature, false)}
                      >
                        {change.from}
                      </span>
                      <ArrowRight className="w-3 h-3 mx-1 text-blue-500" />
                      <span 
                        className="text-blue-600 font-medium"
                        onMouseEnter={() => handleValueFocus(feature, true)}
                        onMouseLeave={() => handleValueFocus(feature, false)}
                      >
                        {change.to}
                      </span>
                    </div>
                  ))}
                </div>
                
                <div className="flex flex-col items-end gap-1">
                  {hasMinimalProgress ? (
                    <div className="flex items-center gap-1 text-sm">
                      <TrendingDown className="w-3 h-3 text-blue-500" />
                      <span className="font-medium text-blue-600">
                        Keep going! Building momentum...
                      </span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1 text-sm">
                      <TrendingDown className="w-3 h-3 text-green-600" />
                      <span className="font-medium text-green-600">
                        {riskReduction}% risk reduction
                      </span>
                    </div>
                  )}
                  {step.prediction_flipped && (
                    <span className="text-xs px-1.5 py-0.5 bg-green-50 text-green-700 rounded-full border border-green-200">
                      Prediction Changes
                    </span>
                  )}
                </div>
              </div>

              {/* Feasibility and recommendation */}
              <div className="flex items-center justify-between mt-1">
                <div 
                  onMouseEnter={() => timeTracking.logCurrentDuration(`feasibility_hover_start_${stepIndex}`)}
                  onMouseLeave={() => timeTracking.logCurrentDuration(`feasibility_hover_end_${stepIndex}`)}
                >
                  <FeasibilityBadge level={step.feasibility} />
                </div>
                
                <button
                  onClick={() => {
                    timeTracking.logCurrentDuration(`recommendations_requested_step${stepIndex}`);
                    onGetRecommendations(step, stepIndex);
                  }}
                  className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-blue-700
                         hover:text-blue-800 hover:bg-blue-50 rounded-md transition-colors"
                  title="Get key recommendations for this step"
                  onMouseEnter={() => timeTracking.logCurrentDuration(`recommendation_button_hover_start_${stepIndex}`)}
                  onMouseLeave={() => timeTracking.logCurrentDuration(`recommendation_button_hover_end_${stepIndex}`)}
                >
                  <Brain className="w-3 h-3" />
                  Get Recommendations
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const CounterfactualTimeline = ({
  visualizationData,
  onAskRecommendation,
  userId
}) => {
  // Initialize time tracking
  const timeTracking = useTimeTracking(userId, 'CounterfactualTimeline', {
    inactivityThreshold: 180000, // 3 minutes
    logInterval: 60000, // Log every minute
    minTimeToLog: 10000 // Minimum 10 seconds
  });

  // Track active step
  const [activeStep, setActiveStep] = useState(null);

  // Track scroll position
  const timelineRef = useRef(null);
  const [visibleSteps, setVisibleSteps] = useState(new Set());

  // Observe timeline steps visibility
  useEffect(() => {
    if (!timelineRef.current) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          const stepIndex = entry.target.getAttribute('data-step-index');
          if (entry.isIntersecting) {
            setVisibleSteps(prev => new Set(prev).add(stepIndex));
            timeTracking.logCurrentDuration(`step_visible_${stepIndex}`);
          } else {
            setVisibleSteps(prev => {
              const newSet = new Set(prev);
              newSet.delete(stepIndex);
              return newSet;
            });
          }
        });
      },
      { threshold: 0.5 }
    );

    const steps = timelineRef.current.querySelectorAll('[data-step-index]');
    steps.forEach(step => observer.observe(step));

    return () => observer.disconnect();
  }, [timeTracking]);

  const handleGetRecommendations = useCallback((step, stepIndex) => {  // Add stepIndex parameter
    let changes = [];
    Object.entries(step.changes).forEach(([feature, change]) => {
      const direction = change.to > change.from ? 'increase' : 'decrease';
      const amount = Math.abs(change.to - change.from).toFixed(1);
      changes.push(`${feature}: ${direction} by ${amount} units (${change.from} → ${change.to})`);
    });
  
    const detailedPrompt = `You MUST respond with ONLY a JSON object. No introductions, explanations, or other text.
      The JSON object should contain actionable recommendations to achieve these changes:
      ${changes.map(c => `- ${c}`).join('\n')}

      Your response must follow this EXACT format:
      {
        "recommendations": [
          {
            "action": "specific actionable step with frequency/intensity",
            "timeline": "clear timeline"
          }
        ]
      }

      Rules:
      1. Each action should be a single, specific activity (e.g., "30-min brisk walking 5 times/week")
      2. Actions must be 16 words or less
      3. Each timeline should be clear and concise (e.g., "Start now, maintain 3 months")
      4. Timelines must be 8 words or less
      5. Provide 2-3 recommendations maximum
      6. Response must be valid JSON
      7. DO NOT include any text outside the JSON object`;
  
    const displayPrompt = `What specific steps should I take to make these changes in Step ${stepIndex + 1}?`;
  
    timeTracking.logCurrentDuration('recommendation_request_generated');
    onAskRecommendation(displayPrompt, detailedPrompt);
  }, [timeTracking, onAskRecommendation]);

  if (!visualizationData?.data) {
    return <div>Missing visualization data</div>;
  }

  // Handle special no recommendations case
  if (visualizationData.data.noRecommendationsNeeded) {
    timeTracking.logCurrentDuration('no_recommendations_case_viewed');
    return (
      <div className="w-full space-y-2">
        <Alert className="bg-blue-50 border-blue-200">
          <AlertTitle className="flex items-center gap-2 text-blue-800">
            <AlertCircle className="w-4 h-4" />
            No Changes Needed
          </AlertTitle>
          <AlertDescription data-translate="true" className="text-blue-800 mt-2">
            Your assessment shows you are unlikely to have diabetes. Keep maintaining your current healthy lifestyle habits - no changes are recommended at this time.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const {
    timeline_steps,
    current_prediction,
    target_prediction
  } = visualizationData.data;

  return (
    <div 
      className="w-full space-y-2" 
      ref={timelineRef}
      onMouseEnter={() => timeTracking.resumeTracking()}
      onMouseLeave={() => timeTracking.pauseTracking()}
    >
      <Alert className="bg-blue-50 border-blue-200">
        <AlertDescription className="text-blue-800">
          Review a personalized health improvement plan showing realistic steps to reduce diabetes risk. 
          Each step builds on previous changes, with feasibility ratings and estimated risk reduction. 
          Click "Get Recommendations" for specific clinical guidance.
        </AlertDescription>
      </Alert>

      <Card>
        <CardContent className="pt-4 pb-2">
          <div className="flow-root pl-4">
            <div className="space-y-2">
              {timeline_steps.map((step, idx) => (
                <TimelineStep
                  key={idx}
                  step={step}
                  isLast={idx === timeline_steps.length - 1}
                  onGetRecommendations={handleGetRecommendations}
                  timeTracking={timeTracking}
                  isActive={activeStep === idx}
                  onHover={(isHovered) => setActiveStep(isHovered ? idx : null)}
                  stepIndex={idx}
                />
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CounterfactualTimeline;
