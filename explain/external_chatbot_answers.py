import os
from typing import Optional, Dict, Any, List, Tuple
import json
import anthropic
from datetime import datetime
import re

api_key = os.environ.get('ANTHROPIC_API_KEY')

class ClaudeAPIHandler:
    """Handles interactions with Claude 3.5 Haiku via Anthropic's API."""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-haiku-20240307"

    def format_spacing(self, text: str) -> str:
        """
        Enforces proper spacing and formatting in the response text with robust error handling.
        """
        formatted_parts = []
        
        try:
            # Process the heading
            heading_match = re.search(r'<h4>(.*?)</h4>', text)
            if heading_match:
                formatted_parts.append(f"<h4>{heading_match.group(1)}</h4>\n\n")
                text = text[heading_match.end():].strip()
            
            # Split content into main text and references if references exist
            if '<h4>References</h4>' in text:
                parts = text.split('<h4>References</h4>')
                main_content = parts[0].strip()
                references = parts[1].strip() if len(parts) > 1 else ''
            else:
                main_content = text.strip()
                references = ''
            
            # Process main text paragraphs
            paragraphs = [p.strip() for p in main_content.split('\n') if p.strip()]
            for paragraph in paragraphs:
                # Remove existing <p> tags if present
                paragraph = paragraph.replace('<p>', '').replace('</p>', '')
                formatted_parts.append(f"<p>{paragraph}</p>\n\n")
            
            # Add references section if it exists
            if references:
                formatted_parts.append("<h4>References</h4>\n\n")
                ref_lines = [r.strip() for r in references.split('\n') if r.strip()]
                for ref in ref_lines:
                    formatted_parts.append(f"{ref}\n")
        
        except Exception as e:
            print(f"Error in format_spacing: {str(e)}")
            # Return original text if formatting fails
            return text
        
        return ''.join(formatted_parts)

    def clean_response(self, response_text: str) -> str:
        """
        Cleans up the response text and enforces proper spacing.
        """
        try:
            # Remove any introductory phrases
            if isinstance(response_text, str) and response_text.startswith('Here is'):
                response_text = response_text.split('\n', 1)[1].strip()
            
            # Remove any div classes
            response_text = response_text.replace(' class="key-findings"', '')
            response_text = response_text.replace(' class="critical-factors"', '')
            response_text = response_text.replace(' class="implications"', '')
            response_text = response_text.replace(' class="observed-patterns"', '')
            response_text = response_text.replace(' class="pattern-analysis"', '')
            response_text = response_text.replace(' class="references"', '')
            
            # Apply proper spacing
            return self.format_spacing(response_text)
            
        except Exception as e:
            print(f"Error in clean_response: {str(e)}")
            return response_text

    def construct_prompt(self, 
                        current_query: str,
                        previous_exchanges: Optional[List[Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]]] = None,
                        current_context: Optional[Dict] = None) -> str:
        """
        Constructs the full prompt to send to Claude.
        
        Args:
            current_query: Current user question
            previous_exchanges: Previous chat exchanges (if any)
            current_context: Current dashboard context including patient data and visualization
        """
        prompt_parts = [
            "I need help providing a response to a user's question in a medical AI explanatory system. "
            "The system analyzes diabetes risk using patient data and various visualizations. "
            "Here's the relevant context and current query:\n"
        ]
        
        # Add current context if available
        if current_context:
            if current_context.get("current_patient"):
                patient_data = current_context["current_patient"]
                prompt_parts.append("\nCurrent Patient Data:")
                prompt_parts.append(f"Feature Values:\n{json.dumps(patient_data['patient_features'], indent=2)}")
                
                if patient_data.get("model_prediction"):
                    pred = patient_data["model_prediction"]
                    prompt_parts.append(f"\nModel Prediction: {pred['prediction']}")
                    prompt_parts.append(f"Prediction Probability: {pred['probability']:.2f}")
            
            if current_context.get("active_visualization"):
                viz = current_context["active_visualization"]
                if viz["type"]:
                    prompt_parts.append(f"\nActive Visualization Type: {viz['type']}")
                if viz["data"]:
                    prompt_parts.append(f"Visualization Data:\n{json.dumps(viz['data'], indent=2)}")
                if viz["component_code"]:
                    prompt_parts.append(f"\nVisualization Component Code:\n{viz['component_code']}")
        
        # Add chat history context
        if previous_exchanges:
            prompt_parts.append("\nPrevious Conversation:")
            for i, (query, response, vis_data, frontend_code) in enumerate(previous_exchanges, 1):
                prompt_parts.append(f"\nExchange {i}:")
                prompt_parts.append(f"User Query: {query}")
                prompt_parts.append(f"System Response: {response}")
                
                if vis_data:
                    prompt_parts.append(f"Visualization Data:\n{json.dumps(vis_data, indent=2)}")
                    
                if frontend_code:
                    prompt_parts.append(f"Frontend Visualization Component:\n{frontend_code}")
                    
                prompt_parts.append("-" * 50)
        
        # Add current query and response requirements
        prompt_parts.extend([
            f"\nCurrent User Query: {current_query}\n",
            "Please provide a concise response following these requirements:",
            "1. Content:",
            "   - Explain the AI's decision-making process clearly and accurately",
            "   - Important: Integrate with any visualization data provided",
            "   - Keep the total response under 60 words",
            "   - Reference specific patient data values when relevant",
            "   - Explain visualizations in context of the patient's case",
            "   - Include only the most relevant medical context",
            "2. Structure:",
            "   - Use <h4> for the main heading (no h1, h2, or h3)",
            "   - Write 1-2 short paragraphs only",
            "   - Each paragraph should be 2-3 sentences maximum",
            "   - Start each paragraph with a clear topic sentence",
            "   - Use line breaks between paragraphs",
            "3. Format:",
            "   - No CSS styles or classes",
            "   - No introductory phrases",
            "   - Simple HTML tags only (<h4>, <p>, <ul>, <li>)\n",
            "Keep the response focused, clear, and concise while maintaining accuracy and personalizing to the current patient's case when relevant."
        ])
        
        return "\n".join(prompt_parts)

    def get_response(self,
                    current_query: str,
                    previous_exchanges: Optional[List[Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]]] = None,
                    current_context: Optional[Dict] = None) -> str:
        """
        Sends the prompt to Claude and gets the response.
        
        Args:
            current_query: Current user question
            previous_exchanges: Previous chat exchanges (if any)
            current_context: Current dashboard context including patient data and visualization
        
        Returns:
            str: Claude's response
            
        Raises:
            Exception: If there's an error communicating with the API
        """
        try:
            prompt = self.construct_prompt(
                current_query=current_query,
                previous_exchanges=previous_exchanges,
                current_context=current_context
            )
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract the response text from the message
            response = message.content[0].text if isinstance(message.content, list) else message.content
            
            # Clean up and format the response
            cleaned_response = self.clean_response(response)
            
            return cleaned_response
            
        except Exception as e:
            error_msg = f"Error getting response from Claude API: {str(e)}"
            print(f"[{datetime.now()}] {error_msg}")
            raise Exception(error_msg)

# Example usage:
# def main():
#     # Initialize the handler with your API key
#     api_key = os.getenv("ANTHROPIC_API_KEY")
#     if not api_key:
#         raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
#     handler = ClaudeAPIHandler(api_key)
    
#     # Example data showing multiple exchanges
#     previous_exchanges = [
#         # Exchange 1
#         (
#             "What are the most important features for the predictions?",
#             "Based on our analysis, glucose and BMI are the most significant features...",
#             {
#                 "data": {
#                     "ClassName": "likely to have diabetes",
#                     "Feature": ["Glucose", "BMI", "Age"],
#                     "Importance": [35, 25, 15]
#                 },
#                 "type": "feature_importance"
#             },
#             None  # No frontend code for this exchange
#         ),
#         # Exchange 2
#         (
#             "Can you explain why the prediction of sample 39 was made?",
#             "The model predicted diabetes for sample 39 primarily due to elevated glucose...",
#             {
#                 "data": {
#                     "sample_id": 39,
#                     "features": {"glucose": 180, "bmi": 32}
#                 },
#                 "type": "individual_explanation"
#             },
#             None
#         )
#     ]
    
#     current_query = "How do these factors compare to medical standards?"
    
#     try:
#         response = handler.get_response(
#             current_query=current_query,
#             previous_exchanges=previous_exchanges
#         )
#         print(response)
#     except Exception as e:
#         print(f"Error: {str(e)}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())