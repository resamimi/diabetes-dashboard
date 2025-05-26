import React from 'react';

const FeatureChangeBox = ({ 
  feature, 
  fromValue, 
  toValue,
  isLast = false
}) => {
  // Use a fixed width for consistent spacing
  const boxWidth = 180;
  const boxHeight = 36;
  const arrowBoxWidth = 60;
  
  // Determine color scheme based on value change
  const isIncrease = toValue > fromValue;
  const colors = isIncrease ? {
    arrow: '#f97316',
    arrowHover: '#ea580c',
    text: 'text-orange-600',
    bg: 'bg-orange-50',
    border: 'border-orange-200'
  } : {
    arrow: '#22c55e',
    arrowHover: '#16a34a',
    text: 'text-green-600',
    bg: 'bg-green-50',
    border: 'border-green-200'
  };

  return (
    <div className={`flex items-center gap-2 ${!isLast ? 'mb-1.5' : ''}`}>
      {/* Feature name */}
      <div className="w-16 text-sm font-medium text-gray-700">
        {feature}:
      </div>
      
      {/* Change visualization */}
      <div className="flex items-center bg-white rounded-lg border border-gray-100 shadow-sm">
        {/* Starting value */}
        <div className="px-2 py-1 min-w-[48px] text-center">
          <span className="text-sm text-gray-600">
            {fromValue}
          </span>
        </div>

        {/* Arrow section */}
        <div className={`px-3 py-1 ${colors.bg} ${colors.border} border-x flex items-center justify-center min-w-[60px]`}>
          <div className="relative w-full flex items-center justify-center">
            {/* Arrow line */}
            <div className="absolute inset-0 flex items-center">
              <div className="w-full h-px bg-current opacity-20" />
            </div>
            
            {/* Arrow */}
            <svg
              className={`w-5 h-5 ${colors.text}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              {isIncrease ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 7l5 5m0 0l-5 5m5-5H6"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M11 17l-5-5m0 0l5-5m-5 5h12"
                />
              )}
            </svg>
          </div>
        </div>

        {/* Target value */}
        <div className={`px-2 py-1 min-w-[48px] text-center ${colors.text} font-medium`}>
          {toValue}
        </div>
      </div>
    </div>
  );
};

// Export named for use in CounterfactualTimeline
export const FeatureChanges = ({ changes }) => {
  return (
    <div className="flex flex-col">
      {Object.entries(changes).map(([feature, change], index, array) => (
        <FeatureChangeBox
          key={feature}
          feature={feature}
          fromValue={change.from}
          toValue={change.to}
          isLast={index === array.length - 1}
        />
      ))}
    </div>
  );
};

export default FeatureChanges;