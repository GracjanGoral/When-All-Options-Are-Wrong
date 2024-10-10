from typing import Dict

TYPE_OF_PROMPTS: Dict[str, str] = {
    "easy": "The answer may not be in the options.",
    "standard": "",
    "baseline": "",
    "hard": "You must choose exactly one option.",
}

TYPE_OF_INSTRUCTIONS: Dict[int, str] = {
    1: "Limit your response to A or B only.",
    2: "Choose only A or B as your response.",
    3: "Select either A or B, nothing else.",
    4: "Respond exclusively with A or B.",
    5: "Pick A or B as the only answer.",
}
