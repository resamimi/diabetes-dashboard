
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from json import dumps
from plotly import utils
import copy


class FeatureRangeExplainer:
    
    def __init__(self, predictions, data, classNamesDict):
        self.predictions = predictions
        self.data = data
        self.classNamesDict = classNamesDict


    def analyze_feature_ranges(self, orderedFeatureNames, percentile_range=(25, 75)):
        """
        Analyze the range of each feature where most instances fall into for each prediction class.
        
        Parameters:
        orderedFeatureNames
        percentile_range (tuple): Tuple of (lower, upper) percentiles to define the range
        
        Returns:
        dict: Dictionary containing feature ranges for each class
        """
        # Convert self.predictions to numpy array if it isn't already
        self.predictions = np.array(self.predictions)
        classNumbers = list(set(self.predictions))
        columnNames = self.data.columns
        # Initialize results dictionary
        feature_ranges = {}
        
        # Analyze each feature
        for classNum in classNumbers:

            feature_ranges[self.classNamesDict[classNum]] = {}
            for feature in orderedFeatureNames:
                
                # Get ranges for instances
                if sum(self.predictions == classNum) > 0:
                    lowerBound = np.percentile(self.data[feature][self.predictions == classNum], percentile_range[0])
                    upperBound = np.percentile(self.data[feature][self.predictions == classNum], percentile_range[1])
                    feature_ranges[self.classNamesDict[classNum]][feature] = (lowerBound, upperBound)
            
        return feature_ranges


    def write_feature_ranges(self, feature_ranges, input_str):
        """
        Print the feature ranges in a readable format.
        
        Parameters:
        feature_ranges (dict): Output from analyze_feature_ranges function
        output_str: string to add results

        Returns:
        string: string containing the ranges
        """
        classNames = list(self.classNamesDict.values())
        classNames.sort(reverse=False)
        
        output_str = copy.deepcopy(input_str)
        output_str += f"<br>When the computer program decides to categorize someone as \"{classNames[0]}\", \
        these key factors have the most influence on its decision with values typically falling in these ranges:<br><br>"

        for clsNm in classNames:  

            for featureName, featureRange in feature_ranges[clsNm].items():

                if featureName == "glucose":
                    output_str += f"• Glucose: {featureRange[0]:.1f} to {featureRange[1]:.1f} mg/dL<br>"
                
                elif featureName == "bmi":
                    output_str += f"• BMI: {featureRange[0]:.1f} to {featureRange[1]:.1f} kg/m²<br>"

                else: output_str += f"• {featureName}: {featureRange[0]:.1f} to {featureRange[1]:.1f}<br>"

            output_str += f"<br>As for the cases that are identified as \"{clsNm}\", \
            the values of the key factors typically falling in these ranges:<br><br>"

        output_str += "The computer program learns to make predictions based on patterns in patient data. \
        Understanding these ranges, especially for the most important factors listed above, \
        helps us verify if the program is making decisions based on medically reasonable values. \
        This is particularly valuable for healthcare professionals, as you can compare these ranges \
        with established clinical values from medical research.<br><br>"

        return output_str