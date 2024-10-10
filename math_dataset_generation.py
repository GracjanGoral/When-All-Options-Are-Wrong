import json
import random
from typing import List, Tuple


def generate_example(difficulty: int, seed: int) -> Tuple[str, str, str]:
    """Generate a math problem based on the given difficulty level."""
    random.seed(seed)
    if difficulty == 2:
        return _generate_medium_example()
    elif difficulty == 3:
        return _generate_hard_example()
    else:
        raise ValueError("Difficulty must be one of {1, 2, 3}")


def generate_easy_dataset() -> None:
    """Generate an easy math problem dataset."""
    questions: List[str] = []
    choices: List[str] = []
    answers: List[str] = []

    first_numbers = range(10)
    second_numbers = range(10)

    pairs = {(min(a, b), max(a, b)) for a in first_numbers for b in second_numbers}

    for a, b in pairs:
        correct_answer = str(a + b)
        options = _generate_answer_choices(correct_answer, 1, 10)

        questions.append(f"What is {a} + {b}?")
        choices.append(", ".join(options))
        answers.append(correct_answer)

    _save_dataset("easy_dataset.json", questions, choices, answers)


def _generate_medium_example() -> Tuple[str, str, str]:
    """Generate a medium-level math problem."""
    a: int = random.randint(10, 99)
    b: int = random.randint(10, 99)
    correct_answer: str = str(a + b)
    options = _generate_answer_choices(correct_answer, 1, 10)

    return f"What is {a} + {b}?", ", ".join(options), correct_answer


def _generate_hard_example() -> Tuple[str, str, str]:
    """Generate a hard-level math problem."""
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    correct_answer = str(a + b)
    options = _generate_answer_choices(correct_answer, 1, 100)

    return f"What is {a} + {b}?", ", ".join(options), correct_answer


def _generate_answer_choices(
    correct_answer: str, min_offset: int, max_offset: int
) -> List[str]:
    """Generate a list of answer choices including the correct answer."""
    options = [
        correct_answer,
        str(int(correct_answer) + random.randint(min_offset, max_offset)),
        str(abs(int(correct_answer) - random.randint(min_offset, max_offset))),
        str(int(correct_answer) + random.randint(min_offset, max_offset)),
    ]
    random.shuffle(options)
    return options


def generate_dataset(
    num_examples: int, difficulty: int, seed: int
) -> Tuple[List[str], List[str], List[str]]:
    """Generate a dataset of math problems for a given difficulty level."""
    questions: List[str] = []
    choices: List[str] = []
    answers: List[str] = []

    for _ in range(num_examples):
        question, choice, answer = generate_example(difficulty, seed)
        questions.append(question)
        choices.append(choice)
        answers.append(answer)
        seed += 1

    return questions, choices, answers


def _save_dataset(
    filename: str, questions: List[str], choices: List[str], answers: List[str]
) -> None:
    """Save the dataset to a JSON file."""
    with open(filename, "w") as f:
        json.dump({"question": questions, "choices": choices, "answer": answers}, f)
