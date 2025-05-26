import React, { useState, useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Alert, AlertTitle, AlertDescription } from './ui/alert';

// Custom tooltip component
const CustomTooltip = ({ active, payload, coordinate }) => {
  if (!active || !payload || !payload[0]) return null;

  const sample = payload[0].payload;
  const features = Object.entries(sample)
    .filter(([key]) => 
      !['x', 'y', 'id', 'True_Label', 'Predicted_Label'].includes(key)
    );

  const tooltipWidth = 200;
  const tooltipHeight = 200;
  const chartWidth = 800;
  const chartHeight = 500;
  
  let tooltipX = coordinate.x + 15;
  let tooltipY = coordinate.y - tooltipHeight / 2;
  
  if (tooltipX + tooltipWidth > chartWidth) {
    tooltipX = coordinate.x - tooltipWidth - 15;
  }
  
  if (tooltipY < 0) {
    tooltipY = 5;
  } else if (tooltipY + tooltipHeight > chartHeight) {
    tooltipY = chartHeight - tooltipHeight - 5;
  }

  return (
    <div 
      className="absolute bg-white rounded-lg shadow-lg border border-gray-200 z-10"
      style={{
        left: tooltipX,
        top: tooltipY,
        width: '200px'
      }}
    >
      <div className="p-3 space-y-2 text-xs">
        <div className="font-medium text-gray-500">
          ID: <span className="text-gray-900">{sample.id}</span>
        </div>

        <div className="space-y-1">
          <div>
            <span className="font-medium text-gray-500">True Label: </span>
            <span className={`font-medium ${
              sample.True_Label === 1 ? 'text-green-600' : 'text-blue-600'
            }`}>
              {sample.True_Label}
            </span>
          </div>
          <div>
            <span className="font-medium text-gray-500">Predicted: </span>
            <span className={`font-medium ${
              sample.Predicted_Label === 1 ? 'text-green-600' : 'text-blue-600'
            }`}>
              {sample.Predicted_Label}
            </span>
          </div>
        </div>

        <div className="pt-1 border-t border-gray-100">
          <div className="font-medium text-gray-500 mb-1">Features:</div>
          <div className="space-y-1">
            {features.map(([feature, value]) => (
              <div key={feature}>
                <span className="font-medium text-gray-500">{feature}: </span>
                <span className="text-gray-900">{value.toFixed(1)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const ScatterClassificationPlot = ({ visualizationData, className }) => {
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [activePoint, setActivePoint] = useState(null);
  const [isTooltipActive, setIsTooltipActive] = useState(true);
  
  const colorMap = {
    'True: 0, Pred: 0, Correct': '#4dabf7',    // Bright blue
    'True: 0, Pred: 1, Incorrect': '#9775fa',  // Purple
    'True: 1, Pred: 0, Incorrect': '#ffd43b',  // Yellow
    'True: 1, Pred: 1, Correct': '#69db7c'     // Bright green
  };
  
  const transformedData = useMemo(() => {
    if (!visualizationData?.data?.metadata?.categories || !visualizationData?.data?.samples) {
      return [];
    }

    const jitter = 0.35;
    return visualizationData.data.metadata.categories.map(category => ({
      name: category.name,
      color: colorMap[category.name] || category.color,
      data: visualizationData.data.samples
        .filter(d => 
          d.True_Label === category.true_label && 
          d.Predicted_Label === category.pred_label
        )
        .map(d => ({
          ...d,
          x: d.True_Label + (Math.random() * 2 - 1) * jitter,
          y: d.Predicted_Label + (Math.random() * 2 - 1) * jitter,
        }))
    }));
  }, [visualizationData]);

  if (!visualizationData?.data?.metadata) {
    return <div>No visualization data available</div>;
  }

  return (
    <div className={className}>
      <Alert className="bg-blue-50 border-blue-200">
        <AlertTitle className="text-blue-800">
          Model Performance
        </AlertTitle>
        <AlertDescription className="text-blue-800">
          Classification Accuracy: {(visualizationData.data.metadata.accuracy * 100).toFixed(1)}%
        </AlertDescription>
      </Alert>
      
      <div 
        className="w-full h-[500px]"
        onClick={() => {
          setActivePoint(null);
          setIsTooltipActive(false);
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart 
            margin={{ 
              top: 20, 
              right: 30, 
              bottom: 20, 
              left: 25 
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" opacity={0.4} />
            <XAxis
              type="number"
              dataKey="x"
              name="True Label"
              domain={[-0.5, 1.5]}
              ticks={[0, 1]}
              label={{ 
                value: 'Ground Truth', 
                position: 'bottom',
                offset: 0
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Predicted Label"
              domain={[-0.5, 1.5]}
              ticks={[0, 1]}
              label={{ 
                value: 'Prediction', 
                angle: -90, 
                position: 'left',
                offset: 0
              }}
            />
            <Tooltip 
              active={isTooltipActive}
              content={isTooltipActive ? <CustomTooltip /> : null}
              cursor={false}
              trigger="hover"
              isAnimationActive={false}
            />
            <Legend 
              onClick={(e) => setSelectedCategory(curr => curr === e.value ? null : e.value)}
              wrapperStyle={{ 
                paddingTop: "20px"
              }}
            />
            {transformedData.map((category) => (
              <Scatter
                key={category.name}
                name={category.name}
                data={selectedCategory ? (selectedCategory === category.name ? category.data : []) : category.data}
                fill={category.color}
                opacity={0.8}
                r={5}
                onMouseEnter={(props) => {
                  setActivePoint({
                    x: props.cx,
                    y: props.cy,
                    color: category.color
                  });
                  setIsTooltipActive(true);
                }}
                onMouseLeave={() => setActivePoint(null)}
                onClick={(e) => {
                  e.stopPropagation();
                }}
              />
            ))}
            {activePoint && (
              <circle
                cx={activePoint.x}
                cy={activePoint.y}
                r="8"
                stroke={activePoint.color}
                strokeWidth="2"
                fill="none"
                style={{ pointerEvents: 'none' }}
              />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ScatterClassificationPlot;