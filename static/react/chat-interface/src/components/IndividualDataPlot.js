import React from 'react';
import { AlertDescription, Alert } from './ui/alert';
import { Activity } from 'lucide-react';


const FeatureAxis = ({ 
  feature, 
  value, 
  minValue, 
  maxValue 
}) => {
  const percentage = ((value - minValue) / (maxValue - minValue)) * 100;
  
  const getTickValues = (min, max) => {
    const range = max - min;
    const step = range / 4;
    return Array.from({ length: 5 }, (_, i) => +(min + i * step).toFixed(2));
  };

  const ticks = getTickValues(minValue, maxValue);

  return (
    <div className="flex items-center mb-4 group"> {/* Removed last:mb-4 and made mb-8 consistent */}
      {/* Feature label */}
      <div className="w-48 pr-6">
        <span className="text-sm font-semibold text-gray-700 group-hover:text-gray-900 transition-colors">
          {feature}
        </span>
      </div>

      {/* Axis container */}
      <div className="flex-1">
        <div className="relative h-12">
          {/* Background for the axis */}
          <div 
            className="absolute inset-0 bg-gradient-to-b from-gray-50 to-white rounded-lg border border-gray-100"
            style={{ top: '8px', bottom: '8px' }}
          />

          {/* Base axis line */}
          <div 
            className="absolute w-full h-px bg-gray-300" 
            style={{ top: '50%' }}
          />
          
          {/* Tick marks and labels */}
          {ticks.map((tickValue, index) => (
            <React.Fragment key={index}>
              <div
                className="absolute h-3 w-px bg-gray-300"
                style={{
                  left: `${(index / 4) * 100}%`,
                  top: 'calc(50% - 6px)'
                }}
              />
              <div
                className="absolute text-xs font-medium text-gray-500"
                style={{
                  left: `${(index / 4) * 100}%`,
                  bottom: '-4px',
                  transform: 'translateX(-50%)'
                }}
              >
                {tickValue}
              </div>
            </React.Fragment>
          ))}

          {/* Value marker with pulse animation */}
          <div className="absolute transform -translate-x-1/2" style={{ left: `${percentage}%`, top: '50%' }}>
            {/* Pulse effect */}
            <div className="absolute w-6 h-6 bg-blue-100 rounded-full animate-pulse opacity-0 group-hover:opacity-100 -translate-x-1/2 -translate-y-1/2" />
            
            {/* Marker dot */}
            <div className="absolute w-4 h-4 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full shadow-md transform -translate-x-1/2 -translate-y-1/2 group-hover:scale-110 transition-transform duration-200" />
            
            {/* Value label */}
            <div className="absolute w-full text-center -top-6">
              <div className="inline-block bg-gray-800 text-white text-xs font-medium px-2 py-1 rounded-md shadow-sm opacity-90 group-hover:opacity-100 transition-opacity">
                {value.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const IndividualDataPlot = ({ 
  visualizationData, 
  className
}) => {
  if (!visualizationData?.data) {
    return <div>Missing visualization data</div>;
  }

  const { sample_data, max_values, min_values } = visualizationData.data;
  const features = Object.keys(sample_data);
  const sampleIndex = Object.keys(sample_data[features[0]])[0];

  return (
    <div className={className}>
      <div className="flex items-center gap-2 pb-4 mb-3 border-b border-gray-100">
        <Activity className="w-4 h-4 text-blue-500" />
        <h2 className="text-sm font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-purple-600">
          Patient Data
        </h2>
      </div>
      
      <div>
        {features.map((feature) => (
          <FeatureAxis
            key={feature}
            feature={feature}
            value={Number(sample_data[feature][sampleIndex])}
            minValue={Number(min_values[feature])}
            maxValue={Number(max_values[feature])}
          />
        ))}
      </div>
    </div>
  );
};

export default IndividualDataPlot;