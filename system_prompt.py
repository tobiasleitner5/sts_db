def get_system_prompt(prompt_instruction: str) -> str:
    """Build the system prompt for STS pair generation.
    
    Central definition so that main.py and main_batch.py
    always use the same prompt wording.
    """
    return (
        f"System Prompt: {prompt_instruction} "
        f"When being asked to generate a sentence based on an input sentence, make sure the writing style remains consistent with the input. "
        f"Here are some examples of news articles that follow a similar style: "
        f"- 'Year to December 31 2000 EARLIEST PROJECTED FORECAST AGO Sales 8.25 16.55 Current 1.46 1.53 Net 1.08 587 mln EPS 89.48 yen 59.46 yen Ord div 20.00 yen 20.00 yen NOTE Sumida Corp is a minor specialised coil manufacturer.'"
        f"- 'Six months to September 30 2000 Sales 23.44 Operating 2.01 Current 1.65 Net loss 2.20 EPS loss 29.29 yen Cash flow 1.10 from operations NOTE Nippon Chemical Industrial Co Ltd manufactures inorganic industrial chemicals.'"
        f"- 'Year to March 31 2001 LATEST ACTUAL FORECAST AGO Sales nil 107.55 Current loss 2.50 prft 10.53 Net 3.00 53.59 EPS 9.08 yen 495.11 yen Ord div 7.00 yen 20.00 yen NOTE Softbank Corp is wholesaler of PC software.'"
        f"Your output must always be a valid JSON object. "
        f"Do not include any conversational text, explanations, or markdown code blocks. "
        f"The JSON must follow this schema: "
        "{"
        '  "output_sentence": <output>'
        "}"
    )
