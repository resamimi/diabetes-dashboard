
"""USAGE From Previous Version
elif follow_up_info["topic"] == "detailed_feature_importance_1":

            ids = follow_up_info["info"]
            wantSamePrediction = True
            data = conversation.temp_dataset.contents['X']
            groundTruth = conversation.temp_dataset.contents['y']
            model = conversation.get_var('model').contents
            predictions = model.predict(data)

            explainer = SimilarPredictionExplainer(predictions, data, groundTruth, 
                                                   data.columns, conversation.class_names, wantSamePrediction)

            # Explain a prediction
            # sample_to_explain = ids[0]
            prediction, similar_samples = explainer.explain(ids)
            predictionStr = conversation.class_names[prediction]

            print(f"Prediction for the sample: {predictionStr}")
            print(f"Number of samples with similar predictions: {len(similar_samples)}")

            # Create and show the visualization
            fig_json = explainer.visualize(ids, similar_samples, predictionStr)
            
            similar_explanation_summary = f"The plot below shows samples within the dataset that are predicted as \"{predictionStr}\" same as the instance(s) you wanted an explabation for.\
                                            These samples are denoted as the similar samples to the target sample(s). "
            similar_explanation_summary += f"The feature values of the samples (in the drop-down menu) are displayed \
                                            to give insight about samples predicted as \"{predictionStr}\" have which range of values for their features.<br><br>"
            similar_explanation_summary += "Machine learning models learn to predcit based on the data. The range of feature values, \
                                            especially for the important features mentioned before, increases \
                                            your understanding of whether the origin of decision-making of the model is reasonable or not. \
                                            This is particularly useful for domain experts as they are able to to match this range of values with the corresponding scientic numbers. "                                
            similar_explanation_summary += "The median and the first (Q1) and third (Q3) quartiles are calaulated for the similar samples' feature values. \
                                            The area between the two quartile lines represents the interquartile range (IQR), \
                                            where 50% of the similar samples' feature values fall.<br><br>"
            similar_explanation_summary += f"<b>Follow up:</b> For the sake of comparision, I can provide this visualization for the samples that are <b>not</b> predcited as {predictionStr}.\
                                            This will allow you to compare the ranges of feature values for samples of the two classes. Just ask for more description!"                                
            similar_explanation_summary += f"<json>{fig_json}"

            notSimExpInfo = [ids, data, groundTruth, predictions]
            notSimExpFollowUp = {"topic": "detailed_feature_importance_2", "info": notSimExpInfo}
            conversation.store_followup_desc(notSimExpFollowUp)

            return similar_explanation_summary, 1

        elif follow_up_info["topic"] == "detailed_feature_importance_2":

            [ids, data, groundTruth, predictions] = follow_up_info["info"]
            wantSamePrediction = False

            explainer = SimilarPredictionExplainer(predictions, data, groundTruth, 
                                                   data.columns, conversation.class_names, wantSamePrediction)

            prediction, similar_samples = explainer.explain(ids)
            predictionStr = conversation.class_names[prediction]

            print(f"Prediction for the sample: {predictionStr}")
            print(f"Number of samples with different predictions: {len(similar_samples)}")

            # Create and show the visualization
            fig_json = explainer.visualize(ids, similar_samples, predictionStr)

            different_explanation_summary = f"This plot displays samples within the dataset that are <b>not</b> predicted as {predictionStr} by each input feature. \
                                            Comparing the ranges of feature values, particularly the important features introdcued bedore, for samples precited as {predictionStr} with for ones predicted different, \
                                            gives you more insight about the origins of the ML model decisions within the data."                                
            different_explanation_summary += f"<json>{fig_json}"
            
            return different_explanation_summary, 1
"""


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from json import dumps
from plotly import utils


class SimilarPredictionExplainer:
    
    def __init__(self, predictions, data, groundTruth, columns, classNames, wantSamePrediction):
        self.predictions = predictions
        self.data = data
        self.groundTruth = groundTruth
        self.feature_names = columns
        self.class_names = classNames
        self.wantSamePrediction = wantSamePrediction
        # self.feature_names = [f"Feature {i+1}" for i in range(data.shape[1])]

    def explain(self, ids):
        sampleID = ids[0]
        sampleIdx = self.data.index.get_loc(sampleID)
        sample_pred = self.predictions[sampleIdx]
        if self.wantSamePrediction:
            similar_indices = np.where(self.predictions == sample_pred)[0]
        else:
            similar_indices = np.where(self.predictions != sample_pred)[0]

        similar_samples = []
        for idx in similar_indices:

            sampleDict = {'id': self.data.index[idx],
                          'features': self.data.iloc[idx],
                          'target': self.groundTruth.iloc[idx]}
            similar_samples.append(sampleDict)

        return sample_pred, similar_samples

    def visualize(self, ids, similar_samples, predictionStr):
        
        sampleID = ids[0]
        fig = make_subplots(rows=1, cols=1)
        sampleFeatures = self.data.loc[sampleID]
        annotations = []
        if self.wantSamePrediction:
            figTitle = f"Samples Predicted as {predictionStr}"
        else:
            figTitle = f"Samples Not Predicted as {predictionStr}"

        for i, feature_name in enumerate(self.feature_names):
            target_value = sampleFeatures[i]
            similar_values = [s['features'][i] for s in similar_samples]
            similar_ids = [s['id'] for s in similar_samples]

            # Calculate statistics
            median = np.median(similar_values)
            q1, q3 = np.percentile(similar_values, [25, 75])

            if self.wantSamePrediction and len(ids) == 1:
                trace_target = go.Scatter(
                    x=[sampleID],
                    y=[target_value],
                    mode='markers',
                    name='Target Sample',
                    marker=dict(size=12, color='red', line=dict(width=2, color='darkred')),
                    visible=(i == 0)
                )
                fig.add_trace(trace_target)

            trace_similar = go.Scatter(
                x=similar_ids,
                y=similar_values,
                mode='markers',
                name='Similar Samples',
                marker=dict(size=6, color='blue', opacity=0.2),
                visible=(i == 0)
            )

            trace_median = go.Scatter(
                x=[0, max(similar_ids) + 1],
                y=[median, median],
                mode='lines',
                name='Median',
                line=dict(color='green', width=2),
                visible=(i == 0),
                showlegend=False
            )

            trace_q1 = go.Scatter(
                x=[0, max(similar_ids) + 1],
                y=[q1, q1],
                mode='lines',
                name='Lower Quartile (25%)',
                line=dict(color='orange', width=2),
                visible=(i == 0),
                showlegend=False
            )

            trace_q3 = go.Scatter(
                x=[0, max(similar_ids) + 1],
                y=[q3, q3],
                mode='lines',
                name='Upper Quartile (75%)',
                line=dict(color='orange', width=2),
                visible=(i == 0),
                showlegend=False
            )

            fig.add_trace(trace_similar)
            fig.add_trace(trace_median)
            fig.add_trace(trace_q1)
            fig.add_trace(trace_q3)

            # Create annotations for each feature
            feature_annotations = [
                dict(
                    x=1.14,
                    y=median,
                    xref='paper',
                    yref='y',
                    text=f'Med: {median:.2f}',
                    showarrow=False,
                    font=dict(color='green', size=14, family='Arial Black'),
                    align='left',
                    visible=(i == 0)
                ),
                dict(
                    x=1.12,
                    y=q1,
                    xref='paper',
                    yref='y',
                    text=f'Q1: {q1:.2f}',
                    showarrow=False,
                    font=dict(color='orange', size=14, family='Arial Black'),
                    align='left',
                    visible=(i == 0)
                ),
                dict(
                    x=1.12,
                    y=q3,
                    xref='paper',
                    yref='y',
                    text=f'Q3: {q3:.2f}',
                    showarrow=False,
                    font=dict(color='orange', size=14, family='Arial Black'),
                    align='left',
                    visible=(i == 0)
                )
            ]
            annotations.extend(feature_annotations)

        buttons = []
        for i, feature_name in enumerate(self.feature_names):
            if self.wantSamePrediction and len(ids) == 1:
                traces_per_feature = 5
            else:
                traces_per_feature = 4
            visibility = [False] * len(self.feature_names) * traces_per_feature
            visibility[i*traces_per_feature:(i+1)*traces_per_feature] = [True] * traces_per_feature

            annotation_visibility = [False] * len(annotations)
            annotation_visibility[i*3:(i+1)*3] = [True] * 3

            visible_annotations = []
            for ann, vis in zip(annotations, annotation_visibility):
                new_ann = ann.copy()
                new_ann['visible'] = vis
                visible_annotations.append(new_ann)

            buttons.append(dict(
                label=feature_name,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Comparison for {feature_name}",
                       "annotations": visible_annotations}]
            ))

        # Calculate tick values and labels
        endNum = max(similar_ids)
        tick_step = max(1, endNum // 10)  # Show at most 10 ticks
        tick_values = list(range(0, endNum, tick_step))
        tick_labels = [str(val) for val in tick_values]

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.08,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )],
            annotations=annotations,
            title=figTitle,
            xaxis_title="Sample Index",
            yaxis_title="Feature Value",
            width=680,
            height=460,
            margin=dict(r=65, l=20, t=100, b=20),
            showlegend=True,
            xaxis=dict(
                tickmode='array',
                tickvals=tick_values,
                ticktext=tick_labels,
                title_standoff=12
            ),
            yaxis=dict(
                title_standoff=0
            ),
            legend=dict(
                yanchor="bottom",
                y=1.,
                xanchor="right",
                x=1.15,
            ),
            paper_bgcolor= "rgba(240,244,248,255)",
            plot_bgcolor= "rgba(240,244,248,255)",
        )

        fig_json = dumps(fig, cls=utils.PlotlyJSONEncoder)

        return fig_json