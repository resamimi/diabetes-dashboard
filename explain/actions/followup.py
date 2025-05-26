"""Followup operation."""

# from explain.perplexity_prompting import prompt_facts
# from explain.scientific_explanations import prompt_facts
import json



def followup_operation(conversation, parse_text, i, **kwargs):
    """Follow up operation.

    If there's an explicit option to followup, this command deals with it.
    """
    follow_up_info = conversation.get_followup_desc()
    
    if follow_up_info == "":
        return "Sorry, I'm a bit unsure what you mean... try again?", 0
    
    # elif isinstance(follow_up_info, dict):

    #     if follow_up_info["topic"] == "feature_range_explanation":

    #         ranges_report = follow_up_info["info"]

    #         return ranges_report, 1
    
    else:
        return follow_up_info, 1
