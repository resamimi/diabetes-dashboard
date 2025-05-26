import React from 'react';
import ScatterClassificationPlot from './ScatterClassificationPlot';
import FeatureImportancePlot from './FeatureImportancePlot';
import FeatureRangePlot from './FeatureRangePlot';
import TypicalMistakesPlot from './TypicalMistakesPlot';
import CounterfactualTimeline from './CounterfactualTimeline';
import PatientDataPlot from './PatientDataPlot';

const VisualizationRenderer = ({ 
  visualizationData, 
  className,
  onFeatureClick, 
  userId  
}) => {
  if (!visualizationData || !visualizationData.type) {
    return <div>No visualization data available</div>;
  }

  const handleCounterfactualRecommendation = (displayPrompt, detailedPrompt) => {
    console.log('Recommendation requested:', { displayPrompt, detailedPrompt }); 
    onFeatureClick(null, displayPrompt, detailedPrompt);
  };

  switch (visualizationData.type) {
    case "classification_scatter":
      return (
        <ScatterClassificationPlot 
          visualizationData={visualizationData} 
          className={className} 
        />
      );
    case "feature_importance":
      return (
        <FeatureImportancePlot 
          visualizationData={visualizationData} 
          className={className} 
          onFeatureClick={onFeatureClick}
          userId={userId}
        />
      );
    case "feature_range":
      return (
        <FeatureRangePlot 
          visualizationData={visualizationData} 
          className={className} 
          onFeatureClick={onFeatureClick}
          userId={userId}
        />
      );
    case "typical_mistakes":
      return (
        <TypicalMistakesPlot 
          visualizationData={visualizationData} 
          className={className}
        />
      );
    case "individual_data":
      return (
        <PatientDataPlot 
          visualizationData={visualizationData} 
          className={className}
        />
      );
    case "counterfactual_explanation":
      return (
        <CounterfactualTimeline
          visualizationData={visualizationData}
          className={className}
          onAskRecommendation={handleCounterfactualRecommendation}
          userId={userId}
        />
      );
    default:
      return <div>Unsupported visualization type: {visualizationData.type}</div>;
  }
};

export default VisualizationRenderer;

