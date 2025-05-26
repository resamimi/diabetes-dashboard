"""This module generates progressive counterfactual recommendations based on achievable incremental changes."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class FeatureChangeGuidelines:
    """Guidelines for reasonable feature changes based on medical research."""
    min_healthy: float  # Minimum healthy value
    max_healthy: float  # Maximum healthy value
    max_monthly_change: float  # Maximum reasonable change per month
    change_difficulty: float  # 0-1 score of how difficult changes are to achieve
    increase_risk: bool  # True if higher values increase risk
    
    @classmethod
    def get_feature_guidelines(cls) -> Dict[str, 'FeatureChangeGuidelines']:
        """Returns evidence-based guidelines for reasonable feature changes."""
        return {
            'BMI': cls(
                min_healthy=18.5,
                max_healthy=24.9,  # Standard healthy BMI range
                max_monthly_change=1.0,  # 1 BMI point per month is reasonable
                change_difficulty=0.7,
                increase_risk=True  # Higher BMI increases risk
            ),
            'Glucose': cls(
                min_healthy=70,
                max_healthy=100,  # Normal fasting glucose
                max_monthly_change=5.0,  # More gradual change per month
                change_difficulty=0.6,
                increase_risk=True  # Higher glucose increases risk
            ),
            # 'BloodPressure': cls(
            #     min_healthy=90,  # Optimal systolic range
            #     max_healthy=120,  
            #     max_monthly_change=5.0,  # Gradual change per month
            #     change_difficulty=0.5,
            #     increase_risk=False  # Higher BP in normal range can be protective
            # ),
            'Insulin': cls(
                min_healthy=16,
                max_healthy=166,  # Normal fasting insulin range
                max_monthly_change=10.0,  # More gradual monthly change
                change_difficulty=0.8,
                increase_risk=True  # Higher insulin increases risk
            )
        }

def get_next_target_value(
    current: float,
    guideline: FeatureChangeGuidelines,
    step_months: int
) -> Tuple[float, bool]:
    """
    Calculate next target value based on monthly change limits.
    Returns (target_value, was_changed).
    """
    max_change_per_step = guideline.max_monthly_change * step_months
    was_changed = False
    target = current

    if guideline.increase_risk:
        # For risk-increasing factors (like BMI, Glucose), decrease if too high
        if current > guideline.max_healthy:
            step_change = min(max_change_per_step, current - guideline.max_healthy)
            target = current - step_change
            was_changed = True
    else:
        # For protective factors (like normal-range BP), increase if too low
        if current < guideline.min_healthy:
            step_change = min(max_change_per_step, guideline.min_healthy - current)
            target = current + step_change
            was_changed = True
            
    return round(target, 1), was_changed

def generate_progressive_improvements(
    model: Any,
    initial_values: Dict[str, float],
    timeline_months: int,
    step_months: int = 3,
    debug: bool = True
) -> List[Dict]:
    """Generate progressive improvements over time."""
    guidelines = FeatureChangeGuidelines.get_feature_guidelines()
    improvements = []
    
    # Make a working copy of values that we'll update
    working_values = initial_values.copy()
    
    # Get initial probability
    initial_proba = model.predict_proba(pd.DataFrame([working_values]))[0][1]
    current_proba = initial_proba
    
    if debug:
        print(f"\nInitial probability: {initial_proba:.3f}")
    
    # For each time period
    for month in range(step_months, timeline_months + step_months, step_months):
        if debug:
            print(f"\nGenerating {month}-month changes:")
            print(f"Starting from:")
            for feat, val in working_values.items():
                if feat in guidelines:
                    guideline = guidelines[feat]
                    if val > guideline.max_healthy:
                        status = "high"
                    elif val < guideline.min_healthy:
                        status = "low"
                    else:
                        status = "healthy"
                    print(f"  {feat}: {val} ({status})")
        
        # Store values before changes for comparison
        old_values = working_values.copy()
        any_changes = False
        
        # Calculate changes for this period
        for feature, guideline in guidelines.items():
            if feature not in working_values:
                continue
                
            target, was_changed = get_next_target_value(
                working_values[feature],
                guideline,
                step_months
            )
            
            if was_changed:
                working_values[feature] = target
                any_changes = True
                if debug:
                    change = target - old_values[feature]
                    print(f"  {feature}: {old_values[feature]} -> {target} "
                          f"(change: {change:+.1f})")
        
        if not any_changes:
            if debug:
                print("No more changes needed")
            break
            
        # Calculate new probability
        new_proba = model.predict_proba(pd.DataFrame([working_values]))[0][1]
        
        # Store improvement if meaningful
        improvement = initial_proba - new_proba
        if debug:
            print(f"New probability: {new_proba:.3f} "
                  f"(total improvement: {improvement:.3f})")
        
        if new_proba < current_proba:
            improvements.append({
                'changes': working_values.copy(),
                'timeline_months': month,
                'new_probability': float(new_proba),
                'probability_improvement': float(improvement)
            })
            current_proba = new_proba
    
    return improvements

def format_counterfactuals_for_visualization(
    counterfactuals: List[Dict],
    instance_data: Dict[str, Dict[int, float]],
    min_values: Dict[str, float],
    max_values: Dict[str, float],
    current_prediction: str,
    initial_probability: float,
    target_prediction: str
) -> Dict:
    """Formats counterfactuals as sequential steps including all changes."""
    guidelines = FeatureChangeGuidelines.get_feature_guidelines()
    
    # Get initial values and probability
    instance_id = list(next(iter(instance_data.values())).keys())[0]
    initial_values = {
        feature: values[instance_id] 
        for feature, values in instance_data.items()
    }
    
    # Track features that change in any step
    changed_features = set()
    
    # Format timeline steps
    timeline_steps = []
    prev_values = initial_values.copy()
    initial_prob = initial_probability
    
    for cf in counterfactuals:
        # Calculate changes from previous step
        step_changes = {}
        difficulty_scores = []
        
        for feature, value in cf['changes'].items():
            if feature in guidelines:  # Only include modifiable features
                prev_value = prev_values[feature]
                change_magnitude = abs(value - prev_value)
                
                # Include any change from previous step
                if abs(value - prev_value) > 0.01:  # Small threshold to handle floating point
                    step_changes[feature] = {
                        'from': round(prev_value, 1),
                        'to': round(value, 1)
                    }
                    changed_features.add(feature)
                    
                    # Calculate difficulty
                    monthly_magnitude = change_magnitude / cf['timeline_months']
                    difficulty = (monthly_magnitude / guidelines[feature].max_monthly_change) * \
                               guidelines[feature].change_difficulty
                    difficulty_scores.append(difficulty)
                    
                    # Update prev_values for next step
                    prev_values[feature] = value
        
        if step_changes:
            # Calculate average difficulty for the step
            avg_difficulty = float(np.mean(difficulty_scores))  # Convert numpy.float64 to Python float
            
            # Determine feasibility category
            if avg_difficulty < 0.2:
                feasibility = 'Easy'
            # elif avg_difficulty < 0.4:
            #     feasibility = 'Easy'
            elif avg_difficulty < 0.5:
                feasibility = 'Moderate'
            # elif avg_difficulty < 0.6:
            #     feasibility = 'Challenging'
            elif avg_difficulty < 0.7:
                feasibility = 'Challenging'
            else:
                feasibility = 'Difficult'
            
            # Calculate risk reduction
            current_prob = float(cf['new_probability'])  # Convert numpy.float64 to Python float
            risk_reduction = float(((initial_prob - current_prob) / initial_prob) * 100)  # Convert to Python float
            
            # Convert numpy bool to Python bool for prediction_flipped
            prediction_flipped = bool((current_prob < 0.5 and initial_prob >= 0.5) or 
                                   (current_prob >= 0.5 and initial_prob < 0.5))
            
            # Add step to timeline
            timeline_steps.append({
                'changes': step_changes,
                'current_probability': current_prob,
                'risk_reduction': risk_reduction,
                'feasibility': feasibility,
                'prediction_flipped': prediction_flipped
            })
    
    # Filter features in initial values to only include those that change
    filtered_initial_values = {
        feature: float(initial_values[feature])  # Convert numpy.float64 to Python float
        for feature in changed_features
    }
    
    return {
        'initial_values': filtered_initial_values,
        'initial_probability': float(initial_prob),  # Convert numpy.float64 to Python float
        'timeline_steps': timeline_steps,
        'current_prediction': current_prediction,
        'target_prediction': target_prediction
    }

def generate_reasonable_counterfactuals(
    model: Any,
    instance_data: Dict[str, Dict[int, float]],
    max_months: int = 24,  # Up to 24 months
    debug: bool = True
) -> List[Dict]:
    """
    Main function to generate progressive counterfactual recommendations.
    """
    instance_id = list(next(iter(instance_data.values())).keys())[0]
    initial_values = {
        feature: values[instance_id] 
        for feature, values in instance_data.items()
    }
    
    return generate_progressive_improvements(
        model=model,
        initial_values=initial_values,
        timeline_months=max_months,
        step_months=3,  # 3-month intervals
        debug=debug
    )