"""Followup operation."""

# from explain.perplexity_prompting import prompt_facts
# from explain.scientific_explanations import prompt_facts




def followup_operation(conversation, parse_text, i, **kwargs):
    """Follow up operation.

    If there's an explicit option to followup, this command deals with it.
    """
    follow_up_info = conversation.get_followup_desc()
    
    if follow_up_info == "":
        return "Sorry, I'm a bit unsure what you mean... try again?", 0
    
    elif isinstance(follow_up_info, dict):

        if follow_up_info["topic"] == "scientific_explanation":

            [topNImpFeatures, otherFeatures] = follow_up_info["info"]
            prompt_response = prompt_facts(topNImpFeatures, otherFeatures, conversation)
            fact_summary = f"<markdown>{prompt_response}"
            return fact_summary, 1

        elif follow_up_info["topic"] == "scientific_explanation_verification":

            [response_text, config, chat] = follow_up_info["info"]

            followup_question = f"""For the text provided about diabetes research, analyze each citation [X] as it appears in order and:

            1. State which specific claim in the text is associated with the citation
            2. Indicate whether that claim appears to be:
            - Directly supported by the cited paper's findings
            - A reasonable interpretation of the paper's findings
            - Misaligned with or not found in the cited paper
            
            Do not include direct quotes from the papers. Instead, focus on verifying whether the text's claims match the general findings and conclusions of each cited work.

            Here is the text to analyze:
            {response_text}
            """

            # Send follow-up and save response
            followup_response = chat.send_message(followup_question, generation_config=config)
            followup_response_text = f"<markdown>{followup_response.candidates[0].content.parts[0].text}"

            return followup_response_text, 1
    
    else:
        return follow_up_info, 1
