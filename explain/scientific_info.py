# from exa_py import Exa
from pprint import pprint
import re
import google.generativeai as genai
import time

def extract_titles_and_urls(text):
    """
    Extract titles and URLs from document search results text and format them together.
    
    Args:
        text (str): Input text containing document search results
        
    Returns:
        str: Formatted string of titles and URLs
    """
    # Extract titles and URLs using regex
    pattern = r'Title:\s*(.*?)\s*\nURL:\s*(.*?)\s*\nID:'
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
    
    # Format matches into desired string
    formatted_string = ""
    for title, url in matches:
        # Clean up any extra whitespace
        title = title.strip()
        url = url.strip()
        formatted_string += f"{title}: {url} , "
        
    return formatted_string


def get_scientific_info(topNImpFeatures, otherFeatures, scientific_file_path):

    topNImpFeatures_str = ""
    for topFeat in topNImpFeatures:
      topNImpFeatures_str += f"\n- {topFeat}"

    otherFeatures_str = ""
    for othFeat in otherFeatures:
      otherFeatures_str += f"\n- {othFeat}"

    # start_time = time.perf_counter()

    # Configure API
    gemini_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=gemini_key)  # Replace with your API key

    # Setup model with parameters optimized for accuracy
    model = genai.GenerativeModel('gemini-1.5-pro')
    # model = genai.GenerativeModel('gemini-1.5-flash')
    config = genai.types.GenerationConfig(
        temperature=0.2,
        top_p=0.2,
        max_output_tokens=8192  
    )

    prompt_str = f"""Please find and analyze reliable academic research papers and clinical guidelines about diabetes factors. Then provide a structured JSON response analyzing these sources following these specifications:

1. First, identify and use recent, authoritative sources such as:
- Peer-reviewed research papers from medical journals
- Clinical practice guidelines from organizations like WHO, ADA
- Systematic reviews and meta-analyses
- Large-scale epidemiological studies

IMPORTANT CITATION REQUIREMENTS:
- Every claim must have a specific citation linked to a specific paper or guideline
- Do NOT use generic citations like "multiple sources" or "generally accepted knowledge"
- Do NOT mention missing citations or need for additional references
- If you cannot find a specific citation for a claim, do not include that claim
- Each citation must correspond to a specific paper listed in the references section
- When citing guidelines (WHO, ADA, etc.), cite the specific published document with its year and title

2. Focus on analyzing the relative importance of these SPECIFICALLY CATEGORIZED factors:

Primary factors (MUST be categorized exactly as listed):{topNImpFeatures_str}

Secondary factors (MUST be categorized exactly as listed):{otherFeatures_str}
"""

    prompt_str += """3. Structure your analysis as a JSON object with the following format:

{
  "primary_factors": {
    "Glucose": {
      "importance_analysis": {
        "analysis": "detailed discussion",
        "citations": {
          // REQUIRED: Each claim must have a specific citation and verification
          // GOOD EXAMPLES:
          "elevated fasting glucose >126 mg/dL indicates diabetes": {
            "citation": "[1]",
            "verification": {
              "type": "direct_quote",  // Options: "direct_quote", "interpretation", "misaligned"
              "evidence": "The criteria for the diagnosis of diabetes are... fasting plasma glucose (FPG) ≥126 mg/dL (7.0 mmol/L)"  // For direct quotes
              // OR
              "type": "interpretation",
              "evidence": "While not directly stated, this interpretation is supported by the paper's discussion of..."  // For interpretations
              // OR
              "type": "misaligned",
              "evidence": "This claim is not supported by or conflicts with the cited paper's findings"  // For misaligned claims
            }
          },
          ... // Additional claims and citations as needed
        }
      },
      "typical_ranges": {
        "values": {
          "min": "value",
          "max": "value",
          "unit": "mg/dL"
        },
        "citations": {
          "specific claim 1": "[citation_number]",
          ... // Additional claims and citations as needed
        }
      },
      "diagnostic_thresholds": {
        "values": {
          "prediabetes": "value",
          "diabetes": "value",
          "unit": "mg/dL"
        },
        "citations": {
          "specific claim 1": "[citation_number]",
          ... // Additional claims and citations as needed
        }
      }
    },
    "BMI": {
      // Similar structure as glucose
    }
  },
  "secondary_factors": {
    "Diabetes Pedigree Function": {  // REQUIRED: All secondary factors must have complete analysis
      "importance_analysis": {
        "analysis": "detailed discussion that MUST include:
        1. The factor's specific role in diabetes
        2. Why it is considered less important than primary factors (Glucose and BMI)
        3. How it complements or relates to primary factors",
        "citations": {
          "specific claim about factor's importance": {
            "citation": "[citation_number]",
            "verification": {
              "type": "direct_quote",  // OR "interpretation" OR "misaligned"
              "evidence": "exact quote or interpretation explanation"
            }
          },
          "specific claim about why this factor is less critical than primary factors": {
            "citation": "[citation_number]",
            "verification": {
              "type": "direct_quote",  // OR "interpretation" OR "misaligned"
              "evidence": "exact quote or interpretation explanation"
            }
          },
          ... // Additional claims and citations as needed
        }
      }
    },
    // REQUIRED: All other secondary factors must follow the same complete structure
    "Age": { ... },
    "Pregnancies": { ... },
    "Blood Pressure": { ... },
    "Skin Thickness": { ... },
    "Insulin": { ... }
  }
  "references": [
    {
      "id": 1,
      // REQUIRED: Use proper scientific citation format
      // GOOD EXAMPLE:
      "citation": "American Diabetes Association. (2024). Standards of Medical Care in Diabetes-2024. Diabetes Care, 47(Supplement 1), S1-S289."
      
      // BAD EXAMPLES - DO NOT USE:
      // "citation": "WHO guidelines"
      // "citation": "Journal of Diabetes Research paper about glucose"
      // "citation": "https://..."
    },
    ... // Additional references as needed
  ]
}

Please ensure:
1. Each analytical claim, range, and threshold has its own citation
2. All numerical values include their units
3. References are numbered sequentially
4. Citations are directly linked to the specific claims they support within each section
5. Include as many claims and citations as necessary to comprehensively cover the research findings
6. Use only reliable, peer-reviewed sources
7. Prefer recent publications (within the last 5-10 years) when available
8. Every single claim must have a specific citation that maps to a specific reference
9. References must be actual papers or guidelines with titles and sources, not generic mentions
10. Do not include any claims that you cannot support with a specific citation
11. When citing organizations like WHO or ADA, cite their specific published guidelines with year and title
12. No placeholder or generic citations are allowed
13. References must follow scientific citation format:
    - For journal articles: Authors, (Year), Title, Journal Name, Volume(Issue), Pages
    - For clinical guidelines: Organization, (Year), Title, Publication Name, Volume(Supplement), Pages
    - For books: Authors, (Year), Title, Edition, Publisher
14. URLs should not be used in place of proper scientific citations
15. Each reference must be complete with all required citation elements
16. References should include DOI when available
17. Strictly follow the provided categorization of primary and secondary factors
18. Do not recategorize or move factors between primary and secondary categories
19. Include all listed factors in their specified categories, even if you find evidence suggesting different relative importance
20. Each citation must include a verification that:
    - Identifies whether the claim is directly quoted, reasonably interpreted, or misaligned
    - For direct quotes: Provides the exact text from the cited paper
    - For interpretations: Explains how the claim relates to the paper's findings
    - For misaligned claims: Explains why the claim is not supported
21. Only include claims that are either directly supported or can be reasonably interpreted from the cited papers
22. Remove or revise any claims that are misaligned with their citations
23. Every factor (both primary and secondary) MUST have:
    - A complete importance_analysis section
    - At least one specific claim with citation and verification
    - No empty or placeholder sections allowed
24. Empty JSON objects ({}) are not acceptable for any factor
25. If insufficient evidence is found for a factor, explain this in the analysis and provide citations for what is known
26. Use exact original factor names as keys in the JSON structure:
    - Primary factors: "Glucose", "BMI"
    - Secondary factors: "Diabetes Pedigree Function", "Age", "Pregnancies", "Blood Pressure", "Skin Thickness", "Insulin"
27. Maintain consistent capitalization as provided in the original factor names
8. For each secondary factor, the importance_analysis MUST:
    - Explicitly compare its importance to primary factors (Glucose and BMI)
    - Provide specific citations supporting why it is considered a secondary rather than primary factor
    - Explain how it interacts with or complements primary factors
29. Each secondary factor's analysis must maintain scientific rigor while clearly establishing its subordinate role to primary factors
"""

    response = model.generate_content(prompt_str, generation_config=config)

    with open(scientific_file_path, 'w', encoding='utf-8') as file:
        file.write(response.text)
    print(f"scientific info saved to {scientific_file_path}")
    
    return response.text
    # end_time = time.perf_counter()  # or time.time()
    # print(f"Total execution time: {end_time - start_time:.4f} seconds")


