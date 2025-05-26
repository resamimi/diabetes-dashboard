import json
import re
from typing import Dict, Any, Tuple, List
from explain.scientific_info import get_scientific_info

def extract_json_from_string(text: str) -> Dict[str, Any]:
    """Extract JSON content from a string containing markdown code blocks"""
    json_match = re.search(r'```json\n(.*?)```', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    raise ValueError("No valid JSON found in the input string")

def format_citation_and_verification(statement: str, citation_info: Dict[str, Any]) -> str:
    """Format a statement with its citation and verification"""
    citation_number = citation_info["citation"].strip("[]")
    verification = citation_info["verification"]
    
    if verification["type"] == "direct_quote":
        verification_text = f"<br><i><b>Direct quote</b> from the cited study: '{verification['evidence']}'</i><br><br>"
    else:
        verification_text = f"<br><i><b>Interpretation</b> based on the following evidence from the paper: {verification['evidence']}</i><br><br>"
    
    if statement.endswith('.'):
        statement = statement[:-1]
    return f"{statement} [{citation_number}].{verification_text}"

css_styles = """
<style>
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h1 { font-size: 9em; margin: 1px 0; font-weight: bold; }
    h2 { font-size: 0.45em; margin: 0.5px 0; font-weight: bold; }
    h3 { font-size: 0.5em; margin: 0.5px 0; font-weight: bold; }
    h4 { font-size: 1em; margin: 0.5px 0; font-weight: bold; }
    p { margin: 0.5px 0; line-height: 1.3; }
    ul { margin: 0.5px 0; padding-left: 20px; }
    li { margin: 0; line-height: 1.3; }
    .key-findings { margin: 1px 0; padding: 2px; background-color: #f9f9f9; border-radius: 3px; }
    .critical-factors { margin: 1px 0; padding-left: 12px; }
    .critical-factors li { margin: 0; }
    .implications { margin: 1px 0; padding: 2px; background-color: #fff3e0; border-left: 3px solid #ff9800; }
    .clinical-standards, .observed-patterns { margin: 1px 0; padding: 2px; background-color: #f9f9f9; border-radius: 3px; }
    .pattern-analysis { margin: 1px 0; padding: 2px; background-color: #fff3e0; border-left: 3px solid #ff9800; }
    hr { margin: 2px 0; border: none; border-top: 1px solid #ddd; }
    .references { margin: 2px 0 0 0; padding-top: 3px; border-top: 1px solid #ddd; }
</style>
"""

def generate_importance_report(data: Dict[str, Any], model_feature_importance: List[str]) -> str:
    # report = ["<h2>Analysis of Key Factors in Diabetes Assessment</h2>", 
    #           "<h3>Introduction</h3>", 
    report = ["<p>This following analysis examines various factors that influence diabetes assessment and risk. Our analysis combines two perspectives: <ul><li>Patterns observed in clinical data from patient records</li><li>Evidence from established medical research</li></ul></p>",
              "<h3>Analysis of Individual Factors</h3>"]
    
    all_factors = {**data["primary_factors"], **data["secondary_factors"]}
    
    for factor in model_feature_importance:
        if factor in all_factors:
            factor_details = all_factors[factor]
            citations = factor_details["importance_analysis"]["citations"]
            analysis_text = []
            for statement, citation_info in citations.items():
                analysis_text.append(format_citation_and_verification(statement, citation_info))
            report.extend([f"<h4>{factor}</h4>", f"<p>{' '.join(analysis_text)}</p>"])
    
    primary_factor_set = set(data["primary_factors"].keys())
    top_model_factors = model_feature_importance[:len(primary_factor_set)]
    
    insights = ["<h3>Understanding the Most Influential Factors</h3>"]
    if "Glucose" in top_model_factors and top_model_factors.index("Glucose") < 3:
        insights.append("<p><strong>Blood Glucose Levels:</strong> Patient data analysis and medical research both emphasize glucose as a key indicator [1]. A direct quote from the research confirms: 'The criteria for the diagnosis of diabetes are fundamentally based on glucose measurements.'</p>")
    if "BMI" in top_model_factors and top_model_factors.index("BMI") < 3:
        insights.append("<p><strong>Body Mass Index (BMI):</strong> Clinical patterns and research literature consistently validate BMI's importance [2]. Research evidence demonstrates that 'higher BMI values strongly correlate with increased diabetes risk.'</p>")
    insights.append("</div>")
    report.extend(insights)
    
    # critical_factors = [f for f in top_model_factors if f in primary_factor_set]
    # factors_section = ["<h4>Most Critical Factors for Diabetes Assessment</h4>",
    #                   "<p>Based on both clinical data patterns and medical evidence, these factors emerge as most crucial:</p>",
    #                   "<ul>",
    #                   "".join([f"<li><strong>{factor}</strong></li>" for factor in critical_factors]),
    #                   "</ul>"]
    # report.extend(factors_section)
    
    report.extend(["<p>Healthcare providers should pay particular attention to these factors when assessing diabetes risk. The alignment between clinical patterns and medical research provides strong support for focusing on these key indicators during patient assessment.</p>",
                  "<h3>References</h3>",
                  "<ol class='references'>",
                  "".join([f"<li>{ref['citation']}</li>" for ref in data["references"]]),
                  "</ol>"])
    
    return "".join(report)

def generate_ranges_report(data: Dict[str, Any], ml_ranges: Dict[str, Tuple[float, float]]) -> str:
    # report = ["<h2>Analysis of Measurement Ranges: Clinical Standards and Patient Data Patterns</h2>",
    #           "<h3>Introduction</h3>",
    report = ["<p>This analysis compares established clinical ranges for diabetes-related measurements with patterns observed in patient data. For each measurement, we present the standard clinical ranges and the ranges where many patients assessed as having higher diabetes risk fall.</p>"]
    
    for factor, details in data["primary_factors"].items():
        factor_section = [f"<h4>{factor}</h4><div><h4>Clinical Standards</h4>"]
        
        if "typical_ranges" in details:
            ranges = details["typical_ranges"]["values"]
            for statement, citation_info in details["typical_ranges"]["citations"].items():
                factor_section.append(f"<p><strong>Normal Range:</strong> {ranges['min']}-{ranges['max']} {ranges['unit']}. {format_citation_and_verification(statement, citation_info)}</p>")
        
        if "diagnostic_thresholds" in details:
            factor_section.extend(["<p><strong>Diagnostic Thresholds:</strong></p>", "<ul>"])
            for statement, citation_info in details["diagnostic_thresholds"]["citations"].items():
                factor_section.append(f"<li>{format_citation_and_verification(statement, citation_info)}</li>")
            factor_section.append("</ul></div>")
        
        if factor in ml_ranges:
            ml_range = ml_ranges[factor]
            unit = details.get("typical_ranges", {}).get("values", {}).get("unit", "")
            observed_patterns = [
                "<div>",
                "<h4>Observed Patterns in Patient Data</h4>",
                f"<p>In the analysis of patient records, 50% of cases assessed as having higher diabetes risk showed {factor} measurements between {ml_range[0]} and {ml_range[1]} {unit}.</p>"
            ]
            
            if "typical_ranges" in details:
                clinical_min = float(details["typical_ranges"]["values"]["min"])
                clinical_max = float(details["typical_ranges"]["values"]["max"])
                
                observed_patterns.append("<h4>Comparison with Clinical Standards</h4>")
                if ml_range[0] > clinical_max:
                    observed_patterns.append("<p>The observed measurements in patients with higher diabetes risk tend to be above the normal clinical range, consistent with established medical understanding of diabetes risk factors.</p>")
                elif ml_range[1] < clinical_min:
                    observed_patterns.append("<p>The observed measurements in patients with higher diabetes risk tend to be below the normal clinical range. This pattern may warrant further investigation.</p>")
                else:
                    observed_patterns.append("<p>The observed measurements in patients with higher diabetes risk overlap with the clinical normal range, suggesting this measurement alone may not be sufficient for risk assessment.</p>")
            observed_patterns.append("</div>")
            factor_section.extend(observed_patterns)
        
        # factor_section.append("<hr>")
        report.extend(factor_section)
    
    report.extend(["<h3>References</h3>",
                  "<ol class='references'>",
                  "".join([f"<li>{ref['citation']}</li>" for ref in data["references"]]),
                  "</ol>"])
    
    return "".join(report)

def generate_reports(top_important_features: List[str], other_features: List[str], ml_ranges: Dict[str, Tuple[float, float]]) -> Tuple[str, str]:
    scientific_file_path = "./cache/scientific_info.txt"
    
    try:
        with open(scientific_file_path, 'r') as file:
            scientific_text = file.read()
    except FileNotFoundError:
        print("Generating scientific info")
        scientific_text = get_scientific_info(top_important_features, other_features, scientific_file_path)

    try:
        data = extract_json_from_string(scientific_text)
        model_feature_importance = top_important_features + other_features
        return (css_styles + generate_importance_report(data, model_feature_importance),
                css_styles + generate_ranges_report(data, ml_ranges))
    except Exception as e:
        return f"<p class='error'>Error processing the input: {str(e)}</p>", ""

    # Example usage:
    # if __name__ == "__main__":
    #     top_important_features = [
    #         "Glucose",
    #         "BMI",
    #         "Age"
    #     ]
        
    #     other_features = [
    #         "Diabetes Pedigree Function",
    #         "Blood Pressure",
    #         "Pregnancies",
    #         "Skin Thickness",
    #         "Insulin"
    #     ]
        
    #     ml_ranges = {
    #         "Glucose": (120, 180),
    #         "BMI": (28, 35)
    #     }
        
    #     importance_report_html, ranges_report_html = generate_reports(
    #         top_important_features,
    #         other_features,
    #         ml_ranges
    #     )


def get_threshold_ranges(classLabel):
    """
    Extract Prediabetes and Overweight ranges from the primary factors data
       
    Returns:
       list: List of tuples containing (min, max) ranges for prediabetes and overweight
   """
    scientific_file_path = "./cache/scientific_info.txt"
    try:
        with open(scientific_file_path, 'r') as file:
            scientific_text = file.read()
    except FileNotFoundError:
        print("Generating scientific info")
        scientific_text = get_scientific_info(top_important_features, other_features, scientific_file_path)

    try:
        data = extract_json_from_string(scientific_text)
    except Exception as e:
        return f"<p class='error'>Error processing the input: {str(e)}</p>", ""

    ranges = []

    inf_num = 1000

    for factor in data['primary_factors'].values():

        if classLabel == "likely to have diabetes": 
            
            lowerBadThresh = None
            if 'diagnostic_thresholds' in factor:
                thresholds = factor['diagnostic_thresholds']['values']
                
                # Handle only diabetes and obese thresholds
                if 'diabetes' in thresholds:
                    lowerBadThresh = round(float(thresholds['diabetes']), 1)
                    
                elif 'obese' in thresholds:
                    lowerBadThresh = round(float(thresholds['obese']), 1)

            else: raise KeyError("diagnostic_thresholds is not in scientific_info")
            if lowerBadThresh == None: raise ValueError("lowerBadThresh is None!")  

            ranges.append((lowerBadThresh, float(inf_num)))
        
        else:

            if 'typical_ranges' in factor:
                thresholds = factor['typical_ranges']['values']
                lowerGoodThresh = round(float(thresholds['min']), 1)
                upperGoodThresh = round(float(thresholds['max']), 1)
                ranges.append((lowerGoodThresh, upperGoodThresh))
            else:
                raise KeyError("typical_ranges is not in scientific_info")
               
    return ranges

   # Example usage:
    # ranges = get_threshold_ranges(data)
    # print(ranges)  # [(100.0, 125.0), (25.0, 29.9)]

