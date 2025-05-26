import React from 'react';

const featureInfo = {
  "Pregnancies": {
    display: "Number of Children"
  },
  "Glucose": {
    display: "Glucose",
    unit: "mg/dL",
    hasRanges: true,
    warningStart: 100,
    criticalStart: 126,
    labelTransform: 'translateX(100%)'
  },
  "BloodPressure": {
    display: "Blood Pressure",
    unit: "mmHg",
    hasRanges: true,
    warningStart: 120,
    criticalStart: 130,
    labelTransform: 'translateX(-50%)', // Adjusted to center the labels
    extendedMax: 150 // Add explicit extended max value
  },
  "SkinThickness": {
    display: "Skin Thickness",
    unit: "mm"
  },
  "Insulin": {
    display: "Insulin",
    unit: "μU/mL",
    hasRanges: true,
    warningStart: 166,
    criticalStart: 200,
    labelTransform: 'translateX(200%)'
  },
  "BMI": {
    display: "BMI",
    unit: "kg/m²",
    hasRanges: true,
    warningStart: 25,
    criticalStart: 30,
    labelTransform: 'translateX(250%)'
  },
  "DiabetesPedigreeFunction": {
    display: "DiabetesPedigree\nFunction",
    multiline: true
  },
  "Age": {
    display: "Age",
    unit: "years"
  }
};

const FeatureNameDisplay = ({ feature }) => {
  const info = featureInfo[feature] || { display: feature };

  if (info.multiline) {
    return (
      <div className="leading-tight">
        {info.display.split('\n').map((line, i) => (
          <React.Fragment key={i}>
            <span className="text-sm font-medium text-gray-700">
              {line}
            </span>
            {i < info.display.split('\n').length - 1 && <br />}
          </React.Fragment>
        ))}
      </div>
    );
  }
  
  return (
    <div className="leading-tight">
      <span className="text-sm font-medium text-gray-700">
        {info.display}
      </span>
      {info.unit && (
        <div className="text-xs text-gray-500">
          {info.unit}
        </div>
      )}
    </div>
  );
};

const Legend = () => (
  <div className="mb-3 flex items-center gap-4 text-xs text-gray-600">
    <div className="flex items-center gap-2">
      <div className="w-4 border-l-2 border-yellow-400 border-dashed h-4"></div>
      <span>Warning threshold</span>
    </div>
    <div className="flex items-center gap-2">
      <div className="w-4 border-l-2 border-red-400 border-dashed h-4"></div>
      <span>Critical threshold</span>
    </div>
  </div>
);

const ThresholdLabel = ({ feature, value, position, color }) => {
  const info = featureInfo[feature] || {};
  const transform = info.labelTransform || 'translateX(-50%)';

  return (
    <div 
      className={`absolute text-[10px] font-medium ${color} whitespace-nowrap`}
      style={{ 
        left: `${position}%`,
        top: '100%',
        transform,
        marginTop: '2px'
      }}
    >
      {value}
    </div>
  );
};

const RangeBar = ({ feature, minValue, maxValue }) => {
  const info = featureInfo[feature];
  if (!info?.hasRanges) return null;

  const warningPosition = ((info.warningStart - minValue) / (maxValue - minValue)) * 100;
  const criticalPosition = ((info.criticalStart - minValue) / (maxValue - minValue)) * 100;

  return (
    <div className="absolute bottom-0 left-0 right-0 h-1 flex">
      {/* Normal range (green) */}
      <div 
        className="h-full bg-green-300"
        style={{ width: `${warningPosition}%` }}
      />
      {/* Warning range (yellow) */}
      <div 
        className="h-full bg-amber-400"
        style={{ width: `${criticalPosition - warningPosition}%` }}
      />
      {/* Critical range (orange) */}
      <div 
        className="h-full bg-orange-600"
        style={{ width: `${100 - criticalPosition}%` }}
      />
    </div>
  );
};

const FactorDisplay = ({ feature, value, minValue, maxValue, distribution }) => {
  const maxDist = Math.max(...distribution.counts);
  const normalizedDist = distribution.counts.map(count => (count / maxDist) * 100);
  const markerPosition = ((value - minValue) / (maxValue - minValue)) * 100;
  
  const info = featureInfo[feature];
  const hasRanges = info?.hasRanges ?? false;
  const warningStart = info?.warningStart;
  const criticalStart = info?.criticalStart;

  const warningPosition = hasRanges 
    ? ((warningStart - minValue) / (maxValue - minValue)) * 100
    : null;
  const criticalPosition = hasRanges 
    ? ((criticalStart - minValue) / (maxValue - minValue)) * 100
    : null;

  const getValueSquareColor = () => {
    if (!hasRanges) return 'bg-blue-400';
    if (value >= criticalStart) return 'bg-orange-600';
    if (value >= warningStart) return 'bg-amber-400';
    return 'bg-blue-400';
  };

  return (
    <div className={`flex items-center gap-4 ${hasRanges ? 'mb-5' : 'mb-2'}`}>
      <div className="w-32">
        <FeatureNameDisplay feature={feature} />
      </div>

      <div className="relative flex-1 max-w-2xl">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="flex h-12">
            <div className={`w-14 ${getValueSquareColor()} text-white font-bold flex items-center justify-center`}>
              <span className="text-lg">{value.toFixed(1)}</span>
            </div>
            
            <div className="flex-1 relative bg-gray-50">
              {/* Range bar at the bottom for features with ranges */}
              {info?.hasRanges && (
                <div className="absolute bottom-0 left-0 right-0">
                  <RangeBar 
                    feature={feature}
                    minValue={minValue}
                    maxValue={maxValue}
                  />
                </div>
              )}

              {/* Distribution and other elements - adjusted spacing based on feature type */}
              <div className={`absolute inset-0 ${info?.hasRanges ? 'bottom-1' : 'bottom-0'} flex items-end`}>
                {normalizedDist.map((height, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-gray-200"
                    style={{ height: `${height}%` }}
                  />
                ))}
              </div>

              {hasRanges && (
                <>
                  <div 
                    className="absolute top-0 bottom-1 border-l-2 border-yellow-400 border-dashed"
                    style={{ left: `${warningPosition}%` }}
                  />
                  <div 
                    className="absolute top-0 bottom-1 border-l-2 border-red-400 border-dashed"
                    style={{ left: `${criticalPosition}%` }}
                  />
                </>
              )}
              
              <div 
                className={`absolute top-0 ${info?.hasRanges ? 'bottom-1' : 'bottom-0'} w-0.5 bg-gray-800`}
                style={{ left: `${markerPosition}%` }}
              >
                <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-0 h-0 
                              border-l-[4px] border-r-[4px] border-b-[4px]
                              border-transparent border-b-gray-800" />
              </div>

              {/* Min/max numbers positioning */}
              <div className={`absolute ${info?.hasRanges ? 'bottom-1' : 'bottom-0'} left-0 right-0 flex justify-between px-2`}>
                <span className="text-xs text-gray-500">{minValue.toFixed(1)}</span>
                <span className="text-xs text-gray-500">{maxValue.toFixed(1)}</span>
              </div>
            </div>
          </div>
        </div>

        {hasRanges && (
          <>
            <ThresholdLabel 
              feature={feature}
              value={warningStart} 
              position={warningPosition} 
              color="text-yellow-600"
            />
            <ThresholdLabel 
              feature={feature}
              value={criticalStart} 
              position={criticalPosition} 
              color="text-red-600"
            />
          </>
        )}
      </div>
    </div>
  );
};

const PatientDataPlot = ({ visualizationData, className }) => {
  if (!visualizationData?.data) {
    return <div>Missing visualization data</div>;
  }

  const { 
    sample_data, 
    max_values: rawMaxValues, 
    min_values,
    distributions 
  } = visualizationData.data;

  // Adjust max values for features that need extended ranges
  const max_values = {...rawMaxValues};
  if (max_values.BloodPressure && max_values.BloodPressure < 150) {
    max_values.BloodPressure = 150; // Force Blood Pressure max to 150
  }

  const features = Object.keys(sample_data);
  const sampleIndex = Object.keys(sample_data[features[0]])[0];

  return (
    <div className={className}>
      <Legend />
      {features.map((feature) => (
        <FactorDisplay
          key={feature}
          feature={feature}
          value={Number(sample_data[feature][sampleIndex])}
          minValue={Number(min_values[feature])}
          maxValue={Number(max_values[feature])}
          distribution={distributions[feature]}
        />
      ))}
    </div>
  );
};

export default PatientDataPlot;