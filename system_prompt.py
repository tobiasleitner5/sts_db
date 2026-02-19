
import os

def get_system_prompt(prompt_instruction: str) -> str:
    """Build the system prompt for STS pair generation.
    
    Central definition so that main.py and main_batch.py
    always use the same prompt wording.
    """
    prompt_file = os.path.join(os.path.dirname(__file__), 'prompts', 'system_prompts', 'system_prompt_v1.txt')
    with open(prompt_file, 'r') as f:
        template = f.read()
    return template.replace('{prompt_instruction}', prompt_instruction)
