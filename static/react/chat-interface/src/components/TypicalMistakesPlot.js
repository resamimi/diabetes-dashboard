import React from 'react';
import { Alert, AlertDescription } from './ui/alert';
import { Card, CardContent } from './ui/card';

const TypicalMistakesPlot = ({ visualizationData }) => {
  if (!visualizationData?.data?.rules) {
    return <div>Missing visualization data</div>;
  }

  const { rules } = visualizationData.data;

  // Helper function to parse condition string and extract feature name and value
  const parseCondition = (condition) => {
    const matches = condition.match(/\((.*?)\s*([<>]=?)\s*([\d.]+)\)/);
    if (!matches) return null;
    return {
      feature: matches[1],
      operator: matches[2],
      value: parseFloat(matches[3])
    };
  };

  // Find the maximum number of samples across all rules for scaling
  const maxSamples = Math.max(...rules.map(rule => rule.samplesNumber));
  
  // Calculate the height of a bar based on number of samples
  const getBarHeight = (samplesNumber) => {
    return (samplesNumber / maxSamples) * 120;
  };

  return (
    <div className="w-full space-y-2">
      <Alert className="bg-blue-50 border-blue-200">
        <AlertDescription className="text-blue-800">
          Shows patterns where the model frequently makes mistakes, helping to increase users' appropriate trust by informing them about when the model can not be trusted.
        </AlertDescription>
      </Alert>

      <Card>
        <CardContent className="p-2">
          <div className="space-y-2">
            {rules.map((rule, ruleIndex) => {
              const barHeight = getBarHeight(rule.samplesNumber);
              const errorHeight = (barHeight * rule.percentage) / 100;
              const conditionCount = rule.conditions.length;
                                            const lastAxisY = (conditionCount - 1) * 35;
              
              return (
                <div 
                  key={ruleIndex} 
                  className="group bg-white px-2 pt-2 pb-2 rounded-lg border border-gray-100 
                           shadow-sm hover:shadow-md transition-all duration-200"
                >
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    viewBox={`0 0 420 ${lastAxisY + 50}`}
                    className="w-full h-auto"
                    preserveAspectRatio="xMidYMid meet"
                  >
                    <defs>
                      <linearGradient id={`featureGradient${ruleIndex}`} x1="0%" y1="0%" x2="100%" y1="0%">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.2"/>
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.4"/>
                      </linearGradient>
                      <linearGradient id={`errorGradient${ruleIndex}`} x1="0%" y1="0%" x2="100%" y1="0%">
                        <stop offset="0%" stopColor="#f97316" stopOpacity="0.7"/>
                        <stop offset="100%" stopColor="#f97316" stopOpacity="0.9"/>
                      </linearGradient>
                    </defs>

                    {/* Feature axes */}
                    <g transform="translate(100,10)">
                      {rule.conditions.map((condition, condIndex) => {
                        const parsed = parseCondition(condition);
                        if (!parsed) return null;
                        
                        return (
                          <g key={condIndex} transform={`translate(0,${condIndex * 35})`}>
                            {/* Feature name */}
                            <text 
                              x="0" 
                              y="0" 
                              fontSize="13" 
                              fill="#1e293b" 
                              fontFamily="Arial" 
                              fontWeight="600"
                              dominantBaseline="middle"
                            >
                              {parsed.feature}
                            </text>

                            {/* Axis group */}
                            <g transform="translate(80,0)">
                              {/* Axis line */}
                              <line x1="0" y1="0" x2="200" y2="0" stroke="#e2e8f0" strokeWidth="1.5"/>
                              
                              {/* Feature range indicator */}
                              <rect 
                                x={parsed.operator.includes('>') ? "100" : "0"} 
                                y="-4" 
                                width="100" 
                                height="8" 
                                fill={`url(#featureGradient${ruleIndex})`}
                                rx="4"
                                className="transition-opacity duration-200"
                                style={{ opacity: 0.8 }}
                              />

                              {/* Axis labels */}
                              <text 
                                x="0" 
                                y="16" 
                                fontSize="10" 
                                fill="#64748b" 
                                fontFamily="Arial"
                                textAnchor="middle"
                              >
                                0
                              </text>

                              {/* Condition value */}
                              <text 
                                x="100" 
                                y="16" 
                                fontSize="11" 
                                fill="#475569" 
                                fontFamily="Arial"
                                textAnchor="middle"
                                fontWeight="500"
                              >
                                {parsed.value.toFixed(2)}
                              </text>

                              {/* Infinity label */}
                              <text 
                                x="200" 
                                y="16" 
                                fontSize="10" 
                                fill="#64748b" 
                                fontFamily="Arial"
                                textAnchor="middle"
                              >
                                inf
                              </text>
                            </g>
                          </g>
                        );
                      })}

                      {/* Vertical error rate bar */}
                      <g transform="translate(-85,20)">
                        {/* Background bar */}
                        <rect 
                          x="0" 
                          y={lastAxisY - barHeight} 
                          width="30" 
                          height={barHeight} 
                          fill="#f8fafc" 
                          stroke="#e2e8f0" 
                          strokeWidth="1.5"
                          rx="2"
                        />
                        {/* Error portion */}
                        <rect 
                          x="0" 
                          y={lastAxisY - errorHeight} 
                          width="30" 
                          height={errorHeight} 
                          fill={`url(#errorGradient${ruleIndex})`}
                          rx="2"
                        />
                        {/* Description */}
                        <text 
                          x="0" 
                          y={lastAxisY + 15}
                          fontSize="12" 
                          fill="#475569" 
                          fontFamily="Arial" 
                          fontWeight="600"
                        >
                          {rule.samplesNumber} samples, {rule.percentage}% incorrect
                        </text>
                      </g>
                    </g>
                  </svg>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TypicalMistakesPlot;