import React from 'react';
import { Card, CardHeader } from './ui/card';
import { AlertCircle, TrendingUp } from 'lucide-react';

const RiskAssessmentCard = ({ prediction }) => {
  // Add a check for null prediction
  if (!prediction) return null;

  // Make sure you're using the correct property names that match the backend response
  const isHighRisk = prediction.result === 'High Risk';
  
  return (
    <Card className="backdrop-blur-sm bg-white/95 shadow-md mb-3">
      <CardHeader className="border-b border-gray-100 p-3 pb-5">
        <div className="space-y-2">
          <div>
            <div className="text-xs font-semibold text-gray-500 mb-1" data-translate="true">Risk Assessment</div>
            <div className="flex items-center justify-between">
              <div data-translate="true" className="flex items-center gap-2">
                <AlertCircle className={`w-5 h-5 ${isHighRisk ? 'text-orange-500' : 'text-green-500'}`} />
                <div className={`text-lg font-semibold ${
                  isHighRisk ? 'text-orange-600' : 'text-green-600'
                }`}>
                  {prediction.result}
                </div>
              </div>

              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-blue-500" />
                <div className="text-lg font-bold text-blue-600">
                  {prediction.confidence}%
                </div>
              </div>
            </div>
          </div>

          {/* Progress bar */}
          <div className="relative">
            <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden shadow-inner border border-gray-200">
              <div 
                className={`h-full rounded-full transition-all duration-500 shadow-sm ${
                  isHighRisk 
                    ? 'bg-gradient-to-r from-orange-500 to-orange-400' 
                    : 'bg-gradient-to-r from-green-500 to-green-400'
                }`}
                style={{ 
                  width: `${prediction.confidence}%`,
                  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)'
                }}
              />
            </div>
            
            {/* Progress bar scale marks */}
            <div className="absolute -bottom-4 left-0 w-full flex justify-between px-0.5">
              <div className="text-[10px] text-gray-400">0%</div>
              <div className="text-[10px] text-gray-400">50%</div>
              <div className="text-[10px] text-gray-400">100%</div>
            </div>
          </div>
        </div>
      </CardHeader>
    </Card>
  );
};

export default RiskAssessmentCard;