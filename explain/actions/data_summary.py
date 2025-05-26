"""Data summary operation."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import json
from plotly import utils


import numpy as np
import json

def create_classification_visualization(data, ground_truth, predictions):
    """
    Creates a structured data format for the React frontend visualization.
    
    Args:
        data: DataFrame containing the feature data
        ground_truth: Array of true labels
        predictions: Array of predicted labels
    
    Returns:
        JSON string containing structured data for visualization
    """
    # Create list of samples with their features and labels
    samples = []
    feature_names = data.columns.tolist()
    
    for idx, (features, true_label, pred_label) in enumerate(
        zip(data.values, ground_truth, predictions)
    ):
        sample = {
            "id": str(idx),
            "True_Label": int(true_label),
            "Predicted_Label": int(pred_label)
        }
        
        # Add features
        for feature_name, feature_value in zip(feature_names, features):
            sample[feature_name] = float(feature_value)
            
        samples.append(sample)
    
    # Calculate summary statistics
    total_samples = len(samples)
    correct_predictions = sum(1 for s in samples 
                            if s["True_Label"] == s["Predicted_Label"])
    
    # Create visualization data package
    visualization_data = {
        "type": "classification_scatter",
        "data": {
            "samples": samples,
            "metadata": {
                "feature_names": feature_names,
                "total_samples": total_samples,
                "accuracy": correct_predictions / total_samples,
                "categories": [
                    {
                        "name": "True: 0, Pred: 0, Correct",
                        "true_label": 0,
                        "pred_label": 0,
                        "color": "#3b82f6"  # blue
                    },
                    {
                        "name": "True: 0, Pred: 1, Incorrect",
                        "true_label": 0,
                        "pred_label": 1,
                        "color": "#ef4444"  # red
                    },
                    {
                        "name": "True: 1, Pred: 0, Incorrect",
                        "true_label": 1,
                        "pred_label": 0,
                        "color": "#f97316"  # orange
                    },
                    {
                        "name": "True: 1, Pred: 1, Correct",
                        "true_label": 1,
                        "pred_label": 1,
                        "color": "#22c55e"  # green
                    }
                ]
            }
        }
    }
    
    return json.dumps(visualization_data)

# Note, these are hardcode for compas!
def data_operation(conversation, parse_text, i, **kwargs):
    """Data summary operation."""
    description = conversation.describe.get_dataset_description()
    text = f"The data contains information related to <b>{description}</b>.<br><br>"

    # List out the feature names
    f_names = list(conversation.temp_dataset.contents['X'].columns)
    f_string = "<ul>"
    for fn in f_names:
        f_string += f"<li>{fn}</li>"
    f_string += "</ul>"
    df = conversation.temp_dataset.contents['X']
    text += f"The exact feature names in the data are listed as follows:{f_string}<br>"

    # Summarize performance
    model = conversation.get_var('model').contents
    score = conversation.describe.get_eval_performance(model, conversation.default_metric)

    # Note, if no eval data is specified this will return an empty string and nothing will happen.
    if score != "":
        text += score
        text += "<br><br>"

    groundTruth = conversation.temp_dataset.contents['y']
    predictions = model.predict(df)
    scatter_fig_json = create_classification_visualization(df, groundTruth, predictions)
    
    text += "The scatter plot below displays samples of the dataset by their sample ID, true label, predicted label, \
             whether the prediction was correct, and feature values for each point. \
             You can see this information by hovering on the data points. "
    text += f"Here <b>1</b> respresents prediction of the samples as <b>{conversation.class_names[1]}</b> \
              and <b>0</b> respresents <b>{conversation.class_names[0]}</b>.<br>"
    text += "You can zoom in to get near to the data points. \
            Also, holding the right button of your mouse and move it enables you view the samples from different angles. "            
    text += "The drop-down menu allows filtering to show specific subsets of the data \
            (e.g., only correct predictions, only incorrect predictions for a specific class, etc.).<br><br>"  
    text += "<b>Follow up:</b> I can provide an in-depth description of the dataset statistics. \
    Just ask for more description!"             
    text += f"<json>{scatter_fig_json}"

    # rest_of_text = "This pie chart shows the proportion of samples \
    # falling into four categories:<br>\
    # 1. True label 0, Predicted 0 (Correct)<br>\
    # 2. True label 0, Predicted 1 (Incorrect)<br>\
    # 3. True label 1, Predicted 0 (Incorrect)<br>\
    # 4. True label 1, Predicted 1 (Correct)<br>"
    # rest_of_text += f"Where 1 represents prediction of the samples as {conversation.class_names[1]} \
    #                 and 0 respresents {conversation.class_names[0]}.<br>"
    # rest_of_text += f"<json>{pie_fig_json}"

    # Create more in depth description of the data, summarizing a few statistics
    rest_of_text = ""
    rest_of_text += "Here's a more in depth summary of the data.<br><br>"

    for i, f in enumerate(f_names):
        mean = round(df[f].mean(), conversation.rounding_precision)
        std = round(df[f].std(), conversation.rounding_precision)
        min_v = round(df[f].min(), conversation.rounding_precision)
        max_v = round(df[f].max(), conversation.rounding_precision)
        new_feature = (f"{f}: The mean is {mean}, one standard deviation is {std},"
                       f" the minimum value is {min_v}, and the maximum value is {max_v}")
        new_feature += "<br><br>"

        rest_of_text += new_feature

    # text += "Let me know if you want to see an in depth description of the dataset statistics.<br><br>"
    conversation.store_followup_desc(rest_of_text)

    return text, 1
