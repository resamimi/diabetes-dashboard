"""Function to show data instances.

For single instances, this function prints out the feature values. For many instances,
it returns the mean.
"""
import gin

from explain.actions.utils import gen_parse_op_text

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
from json import dumps
from plotly import utils

def createDropdownMenu(df, tracesSetNum, fig):

    buttonShowMap = [list(b) for b in [e==1 for e in np.eye(len(df.columns))]]
    buttons = []

    for traceIdx, traceBool in enumerate(buttonShowMap[0] * tracesSetNum):
        if not traceBool:
            fig.update_traces(visible=False, selector=traceIdx)

    for colIdx, col in enumerate(df.columns):
        buttons.append(dict(method='update', label=col, args=[{
                                                                # 'y': [df[col].values],
                                                                'visible': buttonShowMap[colIdx] * tracesSetNum
                                                                }]
        ))

    menuList = [
        dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                xanchor="left",
                y=1.18,
                yanchor="top"
            ),
        ]

    return menuList, fig


def createIndividualDataVis(df, Pred, fullData, conversation):

    colMin = fullData['X'].min(axis=0)
    colMax = fullData['X'].max(axis=0)

    fig = go.Figure()

    color = "royalblue"
    for idx, col in enumerate(df.columns):
        fig.add_trace(go.Bar(
                            x=["Min of Dataset"], 
                            y=[colMin[col]],
                            width=[0.4]*len(df[col].values),
                            marker=dict(color=color),
                            name="Min of Dataset"))

    for idx, col in enumerate(df.columns):
        fig.add_trace(go.Bar(
                            x=["Instance Value"], 
                            y=df[col].values,
                            width=[0.4]*len(df[col].values),
                            marker=dict(color=color),
                            name="Instance Value"))

    for idx, col in enumerate(df.columns):
        fig.add_trace(go.Bar(
                            x=["Max of Dataset"], 
                            y=[colMax[col]],
                            width=[0.4]*len(df[col].values),
                            marker=dict(color=color),
                            name="Max of Dataset"))

    tracesSetNum = 3
    menuList, fig = createDropdownMenu(df, tracesSetNum, fig)

    predStr = conversation.class_names[Pred.to_list()[0]]
    title = f"Prediction: {predStr}"

    # if Pred.to_list()[0] == 0:
    #     title = "Diabetes Prediction: Negative"
    # elif Pred.to_list()[0] == 1:
    #     title = "Diabetes Prediction: Positive"

    fig.update_layout(
        updatemenus=menuList,
        # annotations=[
        #     dict(text="Feature Name:", showarrow=False,
        #     x=-1, y=1.1, yref="paper", align="left")
        # ],
        autosize=False,
        width=460,
        height=460,
        title_text=title,
        showlegend=False
    )

    fig.update_layout(
        {
            "paper_bgcolor": "rgba(240,244,248,255)",
            "plot_bgcolor": "rgba(240,244,248,255)",
        }
    )

    # fig.show()
    fig_json = dumps(fig, cls=utils.PlotlyJSONEncoder)

    return fig_json


def createGroupedDataVis(df, Pred, fullData):

    NegDiabetesIdx = Pred.index[Pred == 0].tolist()
    PosDiabetesIdx = Pred.index[Pred == 1].tolist()
    # print("NegDiabetes\n", NegDiabetes)
    # print("PosDiabetes\n", PosDiabetes)
    # print("df.index\n", df.index)

    fig = go.Figure()
    
    for col in df.columns:
        NegDiabetes_df = df[df.index.isin(NegDiabetesIdx)]
        # NegDiabetesSeries = df[col].iloc[NegDiabetes]
        fig.add_trace(go.Scatter(
            x=NegDiabetes_df.index,
            y=NegDiabetes_df[col],
            mode='markers',
            name="Negative"
        ))
    
    for col in df.columns:
        PosDiabetes_df = df[df.index.isin(PosDiabetesIdx)]
        # PosDiabetesSeries = df[col].iloc[PosDiabetes]
        fig.add_trace(go.Scatter(
            x=PosDiabetes_df.index,
            y=PosDiabetes_df[col],
            mode='markers',
            name="Positive"
        ))

    tracesSetNum = 2
    menuList, fig = createDropdownMenu(df, tracesSetNum, fig)

    fig.update_layout(
        updatemenus=menuList,
        # annotations=[
        #     dict(text="Feature Name:", showarrow=False,
        #     x=-1, y=1.1, yref="paper", align="left")
        # ],
        autosize=False,
        width=660,
        height=460,
        margin=dict(r=65, l=20, t=100, b=20),
        # title_text=title,
        legend_title="Diabetes Prediction"
        # showlegend=False
    )

    fig.update_layout(
        xaxis_title="Instance ID",
        yaxis_title="Feature Value",
        # title="Scatter Plot of features by instance IDs",
    )

    fig.update_layout(
        {
            "paper_bgcolor": "rgba(240,244,248,255)",
            "plot_bgcolor": "rgba(240,244,248,255)",
        }
    )

    # fig.show()
    fig_json = dumps(fig, cls=utils.PlotlyJSONEncoder)

    return fig_json


@gin.configurable
def show_operation(conversation, parse_text, i, n_features_to_show=float("+inf"), **kwargs):
    """Generates text that shows an instance."""

    # fullData = conversation.build_temp_dataset(save=False).contents

    data = conversation.temp_dataset.contents['X']
    Pred = conversation.temp_dataset.contents['y']

    parse_op = gen_parse_op_text(conversation)
    if len(parse_op) > 0:
        intro_text = f"For the data with <b>{parse_op}</b>,"
    else:
        intro_text = "For all the instances in the data,"
    rest_of_info_string = "The rest of the features are<br><br>"
    init_len = len(rest_of_info_string)
    if len(data) == 0:
        return "There are no instances in the data that meet this description.", 0
    if len(data) == 1:
        return_string = f"{intro_text} the features are<br><br>"

        for i, feature_name in enumerate(data.columns):
            feature_value = data[feature_name].values[0]
            text = f"{feature_name}: {feature_value}<br>"
            if i < n_features_to_show:
                return_string += text
            else:
                rest_of_info_string += text

        featureNames = list(conversation.temp_dataset.contents['X'].columns)
        conversation.build_temp_dataset()
        fullData = conversation.temp_dataset.contents['X']
        
        # df = conversation.temp_dataset.contents['X']
        maxInfo = {}    
        minInfo = {}    
        for i, fname in enumerate(featureNames):
            min_v = round(fullData[fname].min())
            max_v = round(fullData[fname].max())    
            maxInfo[fname] = int(max_v)
            minInfo[fname] = int(min_v)

        dataPointDict = data.to_dict()
        
        visualization_data = {
            "type": "individual_data",
            "data": {
                "sample_data": dataPointDict,
                "max_values": maxInfo,
                "min_values": minInfo,
            }
        }

        fig_json = dumps(visualization_data)
        return_string += "Below you can see the plot of the instance's feature values alongside the minimum and maximum values across dataset:<br><br>"
        # fig_json = createIndividualDataVis(data, Pred, fullData, conversation)
        
        return_string += f"<json>{fig_json}"
        
    else:
        """
        return_string = f"{intro_text} the feature values are on average:<br><br>"
        for i, feature_name in enumerate(data.columns):
            feature_value = round(data[feature_name].mean(), conversation.rounding_precision)
            text = f"{feature_name}: {feature_value}<br>"
            if i < n_features_to_show:
                return_string += text
            else:
                rest_of_info_string += text
        """
        instance_ids = str(list(data.index))
        return_string = f"{intro_text} the instance id's are:<br><br>"
        return_string += instance_ids
        return_string += "<br><br>Which one do you want to see?<br><br>"
        return_string += "<br><br>Below you can see the scatter plot of the instances' feature values and their predictons: <br><br>"

        fig_json = createGroupedDataVis(data, Pred, fullData)
        return_string += f"<json>{fig_json}"

    # If we've written additional info to this string
    if len(rest_of_info_string) > init_len:
        return_string += "<br><br>I've truncated this instance to be concise. Let me know if you"
        return_string += " want to see the rest of it.<br><br>"
        conversation.store_followup_desc(rest_of_info_string)
    
    return return_string, 1
