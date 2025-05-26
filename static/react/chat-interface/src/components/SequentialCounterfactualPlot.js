import React, { useState } from 'react';
import { Card, CardContent } from './ui/card';
import { Alert, AlertDescription } from './ui/alert';
import { ArrowUpRight, Brain, Clock, Target } from 'lucide-react';

const FeasibilityBadge = ({ level }) => {
  const colors = {
    'Very Easy to Achieve': {
      bg: 'bg-green-50',
      text: 'text-green-800',
      border: 'border-green-200'
    },
    'Easy to Achieve': {
      bg: 'bg-green-100',
      text: 'text-green-700',
      border: 'border-green-300'
    },
    'Moderately Easy': {
      bg: 'bg-lime-50',
      text: 'text-lime-700',
      border: 'border-lime-200'
    },
    'Moderately Challenging': {
      bg: 'bg-yellow-50',
      text: 'text-yellow-700',
      border: 'border-yellow-200'
    },
    'Challenging': {
      bg: 'bg-orange-50',
      text: 'text-orange-700',
      border: 'border-orange-200'
    },
    'Very Challenging': {
      bg: 'bg-red-50',
      text: 'text-red-700',
      border: 'border-red-200'
    },
    'Difficult to Achieve': {
      bg: 'bg-red-100',
      text: 'text-red-800',
      border: 'border-red-300'
    }
  };

  const style = colors[level] || colors['Moderately Challenging'];

  return (
    <div className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${style.bg} ${style.text} ${style.border} border shadow-sm`}>
      {level}
    </div>
  );
};

const TimelineMarker = ({ 
  month, 
  isFirst, 
  isLast, 
  isSelected, 
  probability,
  targetPrediction,
  onClick 
}) => {
  const isClose = probability <= 0.4;  // Threshold for being close to target
  
  return (
    <div className="flex flex-col items-center">
      {/* Connecting line */}
      {!isFirst && (
        <div className="w-24 h-0.5 -translate-y-[21px] -translate-x-12 bg-gray-200" />
      )}
      
      {/* Marker */}
      <button
        onClick={onClick}
        className={`relative w-10 h-10 rounded-full flex items-center justify-center 
                   border-2 transition-all duration-200 ${
          isSelected
            ? 'bg-blue-50 border-blue-500 shadow-md scale-110'
            : isClose
              ? 'bg-green-50 border-green-200 hover:border-green-300'
              : 'bg-white border-gray-200 hover:border-gray-300'
        }`}
      >
        <Clock className={`w-4 h-4 ${
          isSelected 
            ? 'text-blue-600'
            : isClose
              ? 'text-green-600'
              : 'text-gray-600'
        }`} />
        
        {/* Month label */}
        <div className="absolute -bottom-6 text-xs font-medium text-gray-600">
          {month}m
        </div>
        
        {/* Probability */}
        <div className={`absolute -top-6 text-xs font-medium ${
          isClose ? 'text-green-600' : 'text-gray-600'
        }`}>
          {Math.round(probability * 100)}%
        </div>
      </button>
    </div>
  );
};

const FeatureBox = ({ 
  feature,
  currentValue,
  targetValue,
  isHighlighted
}) => {
  const change = targetValue - currentValue;
  const changePercent = (change / currentValue) * 100;
  
  return (
    <div className={`p-3 rounded-lg border transition-all duration-200 ${
      isHighlighted
        ? 'border-blue-200 bg-blue-50/50 shadow-sm'
        : 'border-gray-100 bg-white'
    }`}>
      <div className="text-sm font-medium text-gray-600 mb-2">
        {feature}
      </div>
      
      <div className="flex items-center gap-2">
        <div className="text-lg font-semibold text-gray-700">
          {currentValue.toFixed(1)}
        </div>
        
        <div className="flex items-center">
          <div className={`text-xs font-medium ${
            change < 0 ? 'text-green-600' : 'text-orange-600'
          }`}>
            {change < 0 ? '↓' : '↑'} {Math.abs(changePercent).toFixed(1)}%
          </div>
        </div>
        
        <div className={`text-sm ${
          isHighlighted ? 'text-blue-600 font-medium' : 'text-gray-500'
        }`}>
          → {targetValue.toFixed(1)}
        </div>
      </div>
    </div>
  );
};

const SequentialCounterfactualPlot = ({
  visualizationData,
  onAskRecommendation
}) => {
  const [selectedStep, setSelectedStep] = useState(0);
  
  if (!visualizationData?.data) {
    return <div>Missing visualization data</div>;
  }
  
  const {
    data_sample: currentValues,
    counterfactuals,
    rankings,
    current_prediction,
    target_prediction
  } = visualizationData.data;
  
  // Get initial values
  const instance_id = Object.values(currentValues)[0] 
    ? Object.keys(Object.values(currentValues)[0])[0]
    : null;
    
  const initialState = {};
  Object.keys(currentValues).forEach(feature => {
    initialState[feature] = currentValues[feature][instance_id];
  });
  
  // Build cumulative state for each step
  const steps = [{
    changes: initialState,
    probability: 1.0,  // Initial probability
    ...rankings.ranked_options[0]
  }];
  
  counterfactuals.forEach((cf, index) => {
    const prevState = steps[steps.length - 1].changes;
    const newState = { ...prevState };
    
    // Apply changes
    Object.entries(cf).forEach(([feature, value]) => {
      if (value !== null) {
        newState[feature] = value;
      }
    });
    
    steps.push({
      changes: newState,
      ...rankings.ranked_options[index]
    });
  });
  
  // Generate detailed prompt for recommendations
  const generateDetailedPrompt = () => {
    const currentStep = steps[selectedStep];
    const prevStep = steps[selectedStep - 1];
    
    let prompt = `Respond with a JSON object containing specific actions and timelines for these health changes:\n`;
    
    Object.entries(currentStep.changes).forEach(([feature, targetValue]) => {
      const prevValue = prevStep.changes[feature];
      if (targetValue !== prevValue) {
        const change = targetValue - prevValue;
        const direction = change < 0 ? 'decrease' : 'increase';
        prompt += `- ${feature}: ${direction} by ${Math.abs(change).toFixed(1)} units (${prevValue.toFixed(1)} → ${targetValue.toFixed(1)})\n`;
      }
    });
    
    prompt += `\nProvide response in this exact JSON format:
    {
      "recommendations": [
        {
          "action": "brief and clear action (max 15 words)",
          "timeline": "detailed timeline (max 5 words)"
        }
      ]
    }
    
    Include 2-3 key recommendations. Each action should be specific and actionable.
    Include measurable goals and clear timelines. Provide response ONLY in the specified JSON format.`;
    
    return prompt;
  };
  
  const generateUserDisplayPrompt = () => {
    const currentStep = steps[selectedStep];
    const prevStep = steps[selectedStep - 1];
    
    const changes = Object.entries(currentStep.changes)
      .filter(([feature, value]) => value !== prevStep.changes[feature])
      .map(([feature, value]) => {
        const change = value - prevStep.changes[feature];
        const direction = change < 0 ? 'decrease' : 'increase';
        return `${feature} ${direction}`;
      })
      .join(' and ');
    
    return `What specific steps should I take to ${changes}?`;
  };

  return (
    <div className="w-full space-y-4">
      <Alert className="bg-blue-50 border-blue-200">
        <AlertDescription className="text-blue-800">
          Shows a step-by-step path to improve health metrics over time, with each step building on previous improvements.
        </AlertDescription>
      </Alert>
      
      <Card>
        <CardContent className="pt-6">
          {/* Timeline visualization */}
          <div className="relative mb-12">
            <div className="flex justify-center items-center gap-12">
              {steps.map((step, index) => (
                <TimelineMarker
                  key={index}
                  month={step.timeline_months || 0}
                  isFirst={index === 0}
                  isLast={index === steps.length - 1}
                  isSelected={index === selectedStep}
                  probability={step.probability}
                  targetPrediction={target_prediction}
                  onClick={() => setSelectedStep(index)}
                />
              ))}
            </div>
          </div>
          
          {/* Selected step details */}
          <div className="space-y-6">
            {/* Header with prediction and difficulty */}
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <div className="text-sm font-medium text-gray-500">
                  {selectedStep === 0 ? 'Current State' : 'Step Changes'}
                </div>
                <div className="text-lg font-semibold text-gray-800">
                  {selectedStep === 0 
                    ? 'Starting Point'
                    : `${steps[selectedStep].timeline_months}-Month Progress`
                  }
                </div>
              </div>
              
              {selectedStep > 0 && (
                <FeasibilityBadge level={steps[selectedStep].feasibility} />
              )}
            </div>
            
            {/* Feature changes grid */}
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(steps[selectedStep].changes).map(([feature, value]) => {
                const prevValue = selectedStep > 0 
                  ? steps[selectedStep - 1].changes[feature]
                  : value;
                  
                return (
                  <FeatureBox
                    key={feature}
                    feature={feature}
                    currentValue={prevValue}
                    targetValue={value}
                    isHighlighted={prevValue !== value}
                  />
                );
              })}
            </div>
            
            {/* Recommendation button */}
            {selectedStep > 0 && (
              <div className="flex justify-center pt-2">
                <button
                  onClick={() => {
                    const detailedPrompt = generateDetailedPrompt();
                    const displayPrompt = generateUserDisplayPrompt();
                    onAskRecommendation(displayPrompt, detailedPrompt);
                  }}
                  className="flex items-center gap-2 px-4 py-2 
                           bg-gradient-to-r from-blue-500 to-blue-600 
                           hover:from-blue-600 hover:to-blue-700 
                           text-white text-sm font-medium rounded-lg 
                           shadow-sm hover:shadow-md transition-all duration-200"
                >
                  <Brain className="w-4 h-4" />
                  Get Detailed Plan for This Step
                  <ArrowUpRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SequentialCounterfactualPlot;