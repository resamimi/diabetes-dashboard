import numpy as np
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass

@dataclass
class FeatureMetadata:
    """Metadata about features to determine how easy they are to change."""
    name: str
    changeability: float  # 0-1 score of how changeable the feature is
    time_to_change: int  # Estimated days to achieve meaningful change
    effort_level: float  # 0-1 score of effort required
    risk_level: float  # 0-1 score of risk involved in change
    min_healthy: float  # Minimum healthy value
    max_healthy: float  # Maximum healthy value
    
    @classmethod
    def get_default_metadata(cls) -> Dict[str, 'FeatureMetadata']:
        """Returns default metadata for diabetes features based on medical knowledge."""
        return {
            'Glucose': cls(
                name='Glucose',
                changeability=0.8,  # Fairly changeable through diet and medication
                time_to_change=30,  # ~1 month to see significant changes
                effort_level=0.6,   # Requires consistent diet management
                risk_level=0.3,     # Moderate risk if changed too quickly
                min_healthy=70,     # Minimum healthy fasting glucose
                max_healthy=100     # Maximum healthy fasting glucose
            ),
            'BMI': cls(
                name='BMI',
                changeability=0.7,  # Changeable but takes time
                time_to_change=90,  # ~3 months for significant change
                effort_level=0.8,   # High effort - diet and exercise needed
                risk_level=0.2,     # Low risk if done properly
                min_healthy=18.5,   # Minimum healthy BMI
                max_healthy=24.9    # Maximum healthy BMI
            ),
            'BloodPressure': cls(
                name='BloodPressure',
                changeability=0.6,  # Moderately changeable
                time_to_change=45,  # ~1.5 months for lifestyle changes to help
                effort_level=0.5,   # Moderate effort through diet and exercise
                risk_level=0.4,     # Moderate-high risk if changed too quickly
                min_healthy=60,     # Minimum healthy systolic
                max_healthy=120     # Maximum healthy systolic
            ),
            'Insulin': cls(
                name='Insulin',
                changeability=0.5,  # More difficult to change naturally
                time_to_change=60,  # ~2 months to improve insulin sensitivity
                effort_level=0.7,   # High effort through diet and exercise
                risk_level=0.6,     # Higher risk due to complexity
                min_healthy=16,     # Minimum healthy fasting insulin
                max_healthy=166     # Maximum healthy fasting insulin
            )
        }

class CounterfactualRanker:
    def __init__(self, feature_metadata: Dict[str, FeatureMetadata] = None):
        """Initialize with feature metadata or use defaults."""
        self.feature_metadata = feature_metadata or FeatureMetadata.get_default_metadata()
        
    def _extract_value(self, nested_dict: Dict[str, Dict[int, float]]) -> float:
        """Extract the single value from a nested dictionary."""
        if not nested_dict or not isinstance(next(iter(nested_dict.values())), (int, float)):
            raise ValueError(f"Invalid nested dictionary format: {nested_dict}")
        return float(next(iter(nested_dict.values())))

    def is_medically_valid(self, 
                          feature_name: str, 
                          value: float) -> bool:
        """Check if a proposed value is medically valid."""
        if feature_name not in self.feature_metadata:
            return True  # Skip validation for unknown features
            
        metadata = self.feature_metadata[feature_name]
        
        # Allow some flexibility beyond strictly healthy ranges
        min_allowed = metadata.min_healthy * 0.9  # 10% below minimum healthy
        max_allowed = metadata.max_healthy * 1.5  # 50% above maximum healthy
        
        return min_allowed <= value <= max_allowed

    def calculate_change_magnitude(self, 
                                 current_value: float, 
                                 target_value: float, 
                                 feature_name: str) -> float:
        """
        Calculate how significant a proposed change is relative to feature's healthy range.
        Add extra penalty for changes that go beyond the healthy range.
        
        Args:
            current_value: Current feature value
            target_value: Proposed new value
            feature_name: Name of the feature
        
        Returns:
            float: Score from 0-1 where higher means bigger relative change
        """
        if feature_name not in self.feature_metadata:
            return 0.5  # Default to moderate change for unknown features
            
        metadata = self.feature_metadata[feature_name]
        healthy_range = metadata.max_healthy - metadata.min_healthy
        
        # Calculate absolute change size
        change_size = abs(target_value - current_value)
        
        # Base difficulty - how many times larger than the healthy range is the change
        base_difficulty = change_size / healthy_range
        
        # Additional penalty if target is outside healthy range
        if target_value < metadata.min_healthy or target_value > metadata.max_healthy:
            distance_from_healthy = min(
                abs(target_value - metadata.min_healthy),
                abs(target_value - metadata.max_healthy)
            )
            # Extra penalty based on how far outside healthy range
            outside_penalty = distance_from_healthy / healthy_range
            base_difficulty += outside_penalty * 0.5  # 50% extra penalty for being outside range
            
        # Ensure the final score is at least proportional to the original change
        return max(min(base_difficulty, 2.0), change_size / (healthy_range * 2))

    def calculate_feature_difficulty(self,
                                   feature_name: str,
                                   change_magnitude: float) -> Dict[str, float]:
        """Calculate difficulty scores for changing a feature."""
        if feature_name not in self.feature_metadata:
            return {
                'difficulty_score': 0.5,
                'effort_required': 0.5,
                'risk_level': 0.5,
                'estimated_days': 45
            }
            
        metadata = self.feature_metadata[feature_name]
        
        # Scale difficulty factors by change magnitude
        scaled_effort = metadata.effort_level * change_magnitude
        scaled_risk = metadata.risk_level * change_magnitude
        scaled_time = metadata.time_to_change * change_magnitude
        
        # Combine scores (weighted sum)
        difficulty_score = (
            0.3 * (1 - metadata.changeability) +  # Lower changeability = higher difficulty
            0.3 * scaled_effort +
            0.2 * scaled_risk +
            0.2 * (scaled_time / 90)  # Normalize time by 90 days
        )
        
        return {
            'difficulty_score': difficulty_score,
            'effort_required': scaled_effort,
            'risk_level': scaled_risk,
            'estimated_days': scaled_time
        }
    
    def rank_counterfactuals(self,
                            current_values: Dict[str, Dict[int, Union[int, float]]],
                            counterfactuals: List[Dict[str, float]]) -> List[Dict]:
        """Rank counterfactual options by their feasibility."""
        ranked_options = []
        
        for cf_option in counterfactuals:
            option_scores = {
                'changes': cf_option.copy(),
                'feature_difficulties': {},
                'overall_scores': {},
                'invalid_changes': []
            }
            
            total_difficulty = 0
            max_time = 0
            max_risk = 0
            valid_changes = 0
            
            for feature, target_value in cf_option.items():
                if feature not in current_values:
                    option_scores['invalid_changes'].append(f"{feature} not found in current values")
                    continue
                
                if not self.is_medically_valid(feature, target_value):
                    option_scores['invalid_changes'].append(
                        f"Target value {target_value} for {feature} is outside medically valid range"
                    )
                    continue
                
                current_value = self._extract_value(current_values[feature])
                
                # Calculate change magnitude with proper capping at 1.0
                change_magnitude = self.calculate_change_magnitude(
                    current_value,
                    target_value,
                    feature
                )
                
                difficulty_scores = self.calculate_feature_difficulty(
                    feature,
                    change_magnitude
                )
                
                option_scores['feature_difficulties'][feature] = {
                    **difficulty_scores,
                    'current_value': current_value,
                    'target_value': target_value,
                    'change_magnitude': change_magnitude
                }
                
                total_difficulty += difficulty_scores['difficulty_score']
                max_time = max(max_time, difficulty_scores['estimated_days'])
                max_risk = max(max_risk, difficulty_scores['risk_level'])
                valid_changes += 1
            
            if valid_changes > 0:
                # Calculate overall scores with adjustment for multiple changes
                base_difficulty = total_difficulty / valid_changes
                
                # Add penalty for multiple changes (each additional change increases difficulty by 20%)
                multiple_change_factor = 1.0 + (0.2 * (valid_changes - 1))
                
                # Calculate adjusted difficulty
                adjusted_difficulty = base_difficulty * multiple_change_factor
                
                option_scores['overall_scores'] = {
                    'average_difficulty': adjusted_difficulty,
                    'total_difficulty': total_difficulty,
                    'max_time_days': max_time,
                    'max_risk_level': max_risk,
                    'num_changes': valid_changes
                }
                
                ranked_options.append(option_scores)
        
        # Sort by average difficulty (lower is better)
        ranked_options.sort(key=lambda x: x['overall_scores']['average_difficulty'])
        
        # Add feasibility categories
        for option in ranked_options:
            avg_diff = option['overall_scores']['average_difficulty']
            
            # Adjusted thresholds accounting for both multiple changes and magnitude
            if avg_diff < 0.4:
                option['feasibility'] = 'Very Easy to Achieve'
            elif avg_diff < 0.6:
                option['feasibility'] = 'Easy to Achieve'
            elif avg_diff < 0.8:
                option['feasibility'] = 'Moderately Easy'
            elif avg_diff < 1.0:
                option['feasibility'] = 'Moderately Challenging'
            elif avg_diff < 1.3:
                option['feasibility'] = 'Challenging'
            elif avg_diff < 1.6:
                option['feasibility'] = 'Very Challenging'
            else:
                option['feasibility'] = 'Difficult to Achieve'
            
            max_time = option['overall_scores']['max_time_days']
            if max_time < 30:
                option['time_description'] = 'Short-term changes (< 1 month)'
            elif max_time < 60:
                option['time_description'] = 'Medium-term changes (1-2 months)'
            else:
                option['time_description'] = 'Long-term changes (2+ months)'
        
        return ranked_options


def get_ranked_counterfactuals(
    current_values: Dict[str, Dict[int, Union[int, float]]],
    counterfactuals: List[Dict[str, float]]
) -> Tuple[Dict, List[Dict[str, float]]]:
    """
    Main function to rank counterfactuals and return frontend-friendly format.
    
    Args:
        current_values: Dict of current feature values in nested format
        counterfactuals: List of counterfactual options
        
    Returns:
        Dict with ranked counterfactuals and metadata
    """
    ranker = CounterfactualRanker()
    ranked_options = ranker.rank_counterfactuals(
        current_values,
        counterfactuals
    )
    
    # Get only the valid counterfactuals
    valid_counterfactuals = []
    valid_ranked_options = []
    for option in ranked_options:
        if not option['invalid_changes'] and option['overall_scores']['average_difficulty'] < 0.8:
            valid_counterfactuals.append(option['changes'])
            valid_ranked_options.append(option)


    # valid_counterfactuals = [
    #     option['changes']
    #     for option in ranked_options
    #     if not option['invalid_changes'] and option['overall_scores']['average_difficulty'] < 0.8  # Filter out extremely difficult changes
    # ]

    result = {
        'ranked_options': valid_ranked_options,
        'metadata': {
            'feature_metadata': {
                name: {
                    'changeability': meta.changeability,
                    'typical_time_days': meta.time_to_change,
                    'effort_level': meta.effort_level,
                    'risk_level': meta.risk_level,
                    'healthy_range': [meta.min_healthy, meta.max_healthy]
                } for name, meta in ranker.feature_metadata.items()
            }
        }
    }

    return result, valid_counterfactuals

# Example usage:
if __name__ == "__main__":
    # Example data in the correct format
    current_values = {
        'Pregnancies': {39: 4}, 
        'Glucose': {39: 111}, 
        'BloodPressure': {39: 72}, 
        'SkinThickness': {39: 47}, 
        'Insulin': {39: 207}, 
        'BMI': {39: 37.1}, 
        'DiabetesPedigreeFunction': {39: 1.39}, 
        'Age': {39: 56}
    }
    
    counterfactuals = [
        {'Glucose': 67.0, 'BMI': 20.4},
        {'BMI': 1.6},
        {'BMI': 11.6}
    ]
    
    result, valid_counterfactuals = get_ranked_counterfactuals(current_values, counterfactuals)
    
    # Print results
    print("\nRanked Counterfactual Options:")
    for i, option in enumerate(result['ranked_options'], 1):
        print(f"\nOption {i}:")
        print(f"Feasibility: {option['feasibility']}")
        print(f"Timeline: {option['time_description']}")
        print("Changes required:")
        for feature, details in option['feature_difficulties'].items():
            print(f"  {feature}:")
            print(f"    Current: {details['current_value']:.1f}")
            print(f"    Target:  {details['target_value']:.1f}")
            print(f"    Difficulty score: {details['difficulty_score']:.2f}")
            print(f"    Estimated days: {details['estimated_days']:.0f}")
            
        if option['invalid_changes']:
            print("\nWarnings:")
            for warning in option['invalid_changes']:
                print(f"  ⚠️ {warning}")