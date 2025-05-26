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
                max_monthly_change=15,  # ~15 points/month with diet/exercise
                change_difficulty=0.6,
                increase_risk=True  # Higher glucose increases risk
            ),
            'BloodPressure': cls(
                min_healthy=60,
                max_healthy=120,  # Normal systolic range
                max_monthly_change=10,  # ~10 points/month through lifestyle
                change_difficulty=0.5,
                increase_risk=True  # Higher BP increases risk
            ),
            'Insulin': cls(
                min_healthy=16,
                max_healthy=166,  # Normal fasting insulin range
                max_monthly_change=20,  # Monthly improvement with lifestyle changes
                change_difficulty=0.8,
                increase_risk=True  # Higher insulin increases risk
            )
        }

def get_next_target_value(
    current: float,
    guideline: FeatureChangeGuidelines,
    step_months: int
) -> float:
    """Calculate next target value based on monthly change limits."""
    max_change_per_step = guideline.max_monthly_change * step_months
    
    if guideline.increase_risk:
        if current > guideline.max_healthy:
            # Calculate total change needed
            total_change_needed = current - guideline.max_healthy
            # Take a step towards healthy range
            step_change = min(max_change_per_step, total_change_needed)
            return round(current - step_change, 1)
    else:
        if current < guideline.min_healthy:
            total_change_needed = guideline.min_healthy - current
            step_change = min(max_change_per_step, total_change_needed)
            return round(current + step_change, 1)
    
    return current  # No change needed if in healthy range

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
    
    # Start with initial values
    current_values = initial_values.copy()
    initial_proba = model.predict_proba(pd.DataFrame([current_values]))[0][1]
    
    if debug:
        print(f"\nInitial probability: {initial_proba:.3f}")
    
    for month in range(step_months, timeline_months + step_months, step_months):
        if debug:
            print(f"\nGenerating {month}-month changes:")
            print(f"Starting from:")
            for feat, val in current_values.items():
                if feat in guidelines:
                    print(f"  {feat}: {val}")
        
        # Calculate target values for this period
        new_values = current_values.copy()
        changes_made = False
        
        for feature, guideline in guidelines.items():
            if feature not in current_values:
                continue
                
            # Get target value for this step
            target = get_next_target_value(
                current_values[feature],
                guideline,
                step_months
            )
            
            # Check if change is meaningful
            if abs(target - current_values[feature]) >= guideline.max_monthly_change * 0.25:
                new_values[feature] = target
                changes_made = True
                if debug:
                    print(f"  {feature}: {current_values[feature]} -> {target} "
                          f"(change: {target - current_values[feature]:.1f})")
        
        if not changes_made:
            continue
            
        # Calculate new probability
        new_proba = model.predict_proba(pd.DataFrame([new_values]))[0][1]
        improvement = initial_proba - new_proba
        
        if debug:
            print(f"New probability: {new_proba:.3f} "
                  f"(total improvement: {improvement:.3f})")
        
        # Store improvement if meaningful
        if improvement > 0.01:  # 1% total improvement
            improvements.append({
                'changes': new_values.copy(),
                'timeline_months': month,
                'new_probability': float(new_proba),
                'probability_improvement': float(improvement)
            })
            
            # Update current values for next iteration
            current_values = new_values.copy()
        
    return improvements

def format_counterfactuals_for_visualization(
    counterfactuals: List[Dict],
    instance_data: Dict[str, Dict[int, float]],
    min_values: Dict[str, float],
    max_values: Dict[str, float],
    current_prediction: str,
    target_prediction: str
) -> Dict:
    """Formats counterfactuals for visualization."""
    guidelines = FeatureChangeGuidelines.get_feature_guidelines()
    
    # Extract changes for each timepoint
    cf_changes = []
    rankings = {
        'ranked_options': []
    }
    
    for cf in counterfactuals:
        # Convert the complete state into changes from initial
        instance_id = list(next(iter(instance_data.values())).keys())[0]
        initial_values = {
            feature: values[instance_id] 
            for feature, values in instance_data.items()
        }
        
        changes = {}
        for feature, value in cf['changes'].items():
            if feature in guidelines:  # Only include modifiable features
                initial = initial_values[feature]
                if abs(value - initial) >= guidelines[feature].max_monthly_change * 0.25:
                    changes[feature] = round(value, 1)
        
        if changes:  # Only include if there are meaningful changes
            cf_changes.append(changes)
            
            # Calculate feasibility
            difficulty_scores = []
            for feature, value in changes.items():
                if feature in guidelines:
                    guideline = guidelines[feature]
                    initial = initial_values[feature]
                    change_magnitude = abs(value - initial)
                    monthly_magnitude = change_magnitude / cf['timeline_months']
                    
                    # Score based on how close to max monthly change
                    difficulty = (monthly_magnitude / guideline.max_monthly_change) * \
                               guideline.change_difficulty
                    difficulty_scores.append(difficulty)
            
            avg_difficulty = np.mean(difficulty_scores) if difficulty_scores else 0.5
            
            # Determine feasibility category
            if avg_difficulty < 0.3:
                feasibility = 'Very Easy to Achieve'
            elif avg_difficulty < 0.4:
                feasibility = 'Easy to Achieve'
            elif avg_difficulty < 0.5:
                feasibility = 'Moderately Easy'
            elif avg_difficulty < 0.6:
                feasibility = 'Moderately Challenging'
            elif avg_difficulty < 0.7:
                feasibility = 'Challenging'
            elif avg_difficulty < 0.8:
                feasibility = 'Very Challenging'
            else:
                feasibility = 'Difficult to Achieve'
                
            rankings['ranked_options'].append({
                'feasibility': feasibility,
                'timeline_months': cf['timeline_months'],
                'probability_improvement': cf['probability_improvement']
            })
    
    return {
        'data_sample': instance_data,
        'max_values': max_values,
        'min_values': min_values,
        'counterfactuals': cf_changes,
        'rankings': rankings,
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