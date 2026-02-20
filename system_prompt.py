
import os
import re

VERSION = 'v3'

def get_system_prompt(prompt_instruction: str) -> str:
    """Build the system prompt for STS pair generation.
    
    Central definition so that main.py and main_batch.py
    always use the same prompt wording.
    """
    prompt_file = os.path.join(os.path.dirname(__file__), 'prompts', 'system_prompts', f'system_prompt_{VERSION}.txt')
    filename = os.path.basename(prompt_file)
    name_without_ext = os.path.splitext(filename)[0]
    if not re.search(r'_v\d+$', name_without_ext):
        raise ValueError(
            f"System prompt file '{filename}' must end with a version suffix '_vX' "
            f"(e.g. 'system_prompt_v1.txt'), got '{name_without_ext}'."
        )
    with open(prompt_file, 'r') as f:
        template = f.read()
    return template.replace('{prompt_instruction}', prompt_instruction)


def get_system_prompt_version() -> str:
    return VERSION
