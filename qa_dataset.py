import itertools
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from type_of_prompts import TYPE_OF_PROMPTS

rng = np.random.default_rng(42)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class QADataset(Dataset):
    """A dataset class for question and answer data.

    Attributes:
        dataset: A pandas DataFrame containing the QA data.
    """

    def __init__(
        self,
        path_to_dataset: str,
        suffix: str,
        number_of_presented_options: int,
        with_gold_answer: bool,
        how_close_to_correct_answer: float,
        type_of_dataset: str,
    ) -> None:
        """
        Args:
            dataset: The dataset to preprocess.
            suffix: The suffix to append to the instruction, for example "There is no correct answer.".
            number_of_presented_options: The number of choices to present.
            with_gold_answer: A boolean indicating if the transformed dataset will contain the gold answer.
            how_close_to_correct_answer: A float value to adjust the correct answer. This is used to test if the model struggles with values closer to the correct answer. Also, this option is used for Basic Addition Dataset (or any other dataset with similar structure). Also, with_gold_answer should be set to False.
            type_of_dataset: The type of dataset, for now it can be "MMLU" or "BAD" (Basic Addition Dataset). But it can be extended to other types. You can provide MMUL type of dataset (or BAD) and preprocess data.

        """
        self.path_to_dataset = path_to_dataset
        self.suffix = suffix
        self.number_of_presented_options = number_of_presented_options
        self.with_gold_answer = with_gold_answer
        self.how_close_to_correct_answer = how_close_to_correct_answer
        self.type_of_dataset = type_of_dataset
        self.name_of_dataset = path_to_dataset.split("/")[-1].split(".")[0]

        try:
            dataset = pd.read_json(self.path_to_dataset)
        except FileNotFoundError as e:
            logging.error(f"Failed to load dataset: {str(e)}")
            raise

        self.transformed_dataset = self.pre_process_raw_dataset(
            dataset=dataset,
            suffix=self.suffix,
            number_of_presented_options=self.number_of_presented_options,
            with_gold_answer=self.with_gold_answer,
            how_close_to_correct_answer=self.how_close_to_correct_answer,
            type_of_dataset=self.type_of_dataset,
        )

    def __len__(self) -> int:
        return len(self.transformed_dataset)

    def __getitem__(self, idx: int) -> Tuple[str, str, str, str, str, int, str]:
        question = self.transformed_dataset.iloc[idx]["question"]
        answer = self.transformed_dataset.iloc[idx]["correct_answer"]
        adjusted_answers = self.transformed_dataset.iloc[idx]["adjusted_answers"]
        question_type = self.transformed_dataset.iloc[idx]["question_type"]
        dataset_type = self.transformed_dataset.iloc[idx]["dataset_type"]
        presented_options = self.transformed_dataset.iloc[idx]["presented_options"]
        options = self.transformed_dataset.iloc[idx]["options"]
        return (
            question,
            question_type,
            answer,
            adjusted_answers,
            dataset_type,
            presented_options,
            options,
        )

    def pre_process_raw_dataset(
        self,
        dataset: pd.DataFrame,
        suffix: str,
        number_of_presented_options: int,
        with_gold_answer: bool,
        how_close_to_correct_answer: Optional[float],
        type_of_dataset: str,
    ) -> pd.DataFrame:
        """
        This function preprocesses the dataset by processing the choices and formatting the questions. It can be used for both MMLU and Basic Addition Dataset (or any other dataset with similar structure).

        Args:
            dataset: The dataset to preprocess.
            suffix: The suffix to append to the instruction, for example "There is no correct answer.".
            number_of_presented_options: The number of choices to present.
            with_gold_answer: A boolean indicating if the dataset contains the gold answer.
            how_close_to_correct_answer: A float value to adjust the correct answer. Also, if we don't use this option then we return in adjusted_answers the same value as the correct answer.
            type_of_dataset:  The type of dataset, for now it can be "MMLU" or "BAD" (Basic Addition Dataset). But it can be extended to other types. You can provide MMUL type of dataset (or BAD) and preprocess data.

        Returns:
            A pandas DataFrame containing the preprocessed dataset.
        """
        logging.basicConfig(level=logging.INFO)
        logging.info(
            "Preprocessing dataset: presented_options={}, how_close={}, type={}, suffix={}, with_gold_answer={}".format(
                number_of_presented_options,
                how_close_to_correct_answer,
                type_of_dataset,
                suffix,
                with_gold_answer,
            )
        )

        function_args = {
            "number_of_presented_options": number_of_presented_options,
            "type_of_dataset": type_of_dataset,
        }

        if with_gold_answer:
            process_func = self.process_choices_with_gold_answer
            args = lambda c, a: {**function_args, "choices": c, "answer": a}
        else:
            process_func = self.process_choices
            args = lambda c, a: {
                **function_args,
                "choices": c,
                "answer": a,
                "adjustment": (
                    how_close_to_correct_answer if type_of_dataset == "BAD" else None
                ),
            }

        processed_and_permuted_choices = [
            process_func(**args(choices, answer))
            for choices, answer in zip(dataset["choices"], dataset["answer"])
        ]

        adjusted_answers = [
            (
                str(float(answer) + how_close_to_correct_answer)
                if type_of_dataset == "BAD" and how_close_to_correct_answer
                else answer
            )
            for answer in dataset["answer"]
        ]
        formatted_questions = self.format_questions(
            questions=dataset["question"],
            choices=processed_and_permuted_choices,
            number_of_presented_options=number_of_presented_options,
            suffix="baseline" if with_gold_answer else suffix,
        )

        results = {
            "question": [q["question"] for q in formatted_questions],
            "question_type": [q["type"] for q in formatted_questions],
            "options": [q["options"] for q in formatted_questions],
            "adjusted_answers": [
                element for element in adjusted_answers for _ in range(2)
            ],
            "correct_answer": [
                element for element in dataset["answer"].tolist() for _ in range(2)
            ],
            "dataset_type": [type_of_dataset] * len(dataset) * 2,
            "presented_options": [number_of_presented_options] * len(dataset) * 2,
        }
        return pd.DataFrame(results)

    @staticmethod
    def process_choices(
        choices: Union[str, List[str]],
        answer: str,
        type_of_dataset: str,
        adjustment: Optional[float],
        number_of_presented_options: int,
        negative: bool = False,
    ) -> List[List[str]]:
        """
        This function processes and permutes choices based on the dataset type. It can be used for both MMLU and Basic Addition Dataset. It can select number of presented options.
        Also, it can adjust some value to the correct answer (this can we helpful for testing if the model struggle with values closer to the correct answer).

        Args:
            choices: A string of comma-separated choices or a list of choices.
            answer: The correct answer.
            type_of_dataset: The type of dataset, for now it can be "MMLU" or "BAD" (Basic Addition Dataset). But it can be extended to other types. You can provide MMUL type of dataset (or BAD) and preprocess data.
            adjustment: A float value to adjust the correct answer.
            number_of_presented_options: The number of choices to present.

        Returns:
            A list containing two lists of choices, potentially permuted, with the correct answer removed.
        """

        assert not (
            type_of_dataset == "MMLU" and adjustment
        ), "If you use MMLU dataset, you cannot use adjustment."

        if isinstance(answer, float | int):
            answer = str(answer)

        choices_list = choices if type_of_dataset == "MMLU" else choices.split(", ")

        choices_list = [
            choice for choice in choices_list if choice.strip() != answer.strip()
        ]

        print(choices_list)

        if type_of_dataset == "BAD" and adjustment:
            close_answer: str = str((int(float(answer) + adjustment)))
            final_choices: List[str] = choices_list[
                : number_of_presented_options - 1
            ] + [close_answer]
        else:
            final_choices: List[str] = choices_list[:number_of_presented_options]

        if negative:
            final_choices = [str(-int(choice)) for choice in final_choices]

        if len(final_choices) > 2:
            permuted_choices = list(itertools.permutations(final_choices))
            rng.shuffle(permuted_choices)

            for permutation in permuted_choices:
                if list(permutation) != final_choices:
                    return [final_choices, list(permutation)]

        return [final_choices, final_choices[::-1]]

    @staticmethod
    def process_choices_with_gold_answer(
        choices: str | List[str],
        type_of_dataset: str,
        answer: str,
        number_of_presented_options: int,
    ) -> List[List[str]]:
        """
        This function processes and permutes choices based on the dataset type. It can be used for both MMLU and Basic Addition Dataset. It can select number of presented options.

        Args:
            choices: A string of comma-separated choices or a list of choices.
            type_of_dataset: The type of dataset, for now it can be "MMLU" or "BAD" (Basic Addition Dataset). But it can be extended to other types. You can provide MMUL type of dataset (or BAD) and preprocess data.
            answer: The correct answer.
            number_of_presented_options: The number of choices to present.

        Returns:
            A list containing two lists of choices, potentially permuted, with the correct answer.
        """
        choices_list = choices if type_of_dataset == "MMLU" else choices.split(", ")

        answer = str(answer)

        if answer not in choices_list[:number_of_presented_options]:
            answer_index = choices_list.index(answer)
            replace_index = rng.choice(list(range(0, number_of_presented_options)))
            choices_list[replace_index], choices_list[answer_index] = (
                choices_list[answer_index],
                choices_list[replace_index],
            )

        final_choices = choices_list[:number_of_presented_options]

        permuted_choices = list(itertools.permutations(final_choices))
        rng.shuffle(permuted_choices)

        for permutation in permuted_choices:
            if list(permutation) != final_choices:
                return [final_choices, list(permutation)]

        return [final_choices, final_choices[::-1]]

    def format_questions(
        self,
        questions: List[str],
        choices: List[List[List[str]]],
        number_of_presented_options: int,
        suffix: str,
    ) -> List[Dict[str, str]]:
        """
        Formats a list of questions with choices into a structured dictionary format.

        Args:
            questions: A list of questions.
            choices: A list of choices for each question.
            number_of_presented_options: The number of choices to present.
            suffix: A string to append to the instruction. Now is implemented 4 types: 'easy', 'standard', 'baseline', and 'hard'. For example for 'easy' we have 'The answer may not be in the options.'

        Returns:
            A list of dictionaries containing the formatted questions with indicator for original or permuted question.

        Example:
            questions = ["What is 1 + 1?", "What is 1 + 1?"]
            choices = [[["1", "2"]], [["2", "1"]]]
            number_of_presented_options = 2
            suffix = ""

            Result:
            [
                {
                    "question": "What is 1 + 1? Respond solely with A or B. Options:\nA. 1\nB. 2\nAnswer:",
                    "type": "original_question",
                    options: "Options:\nA. 1\nB. 2"
                },
                {
                    "question": "What is 1 + 1? Respond solely with A or B. Options:\nA. 2\nB. 1\nAnswer:",
                    "type": "permuted_question",
                    "options": " Options:\nA. 2\nB. 1"
                }
            ]
        """

        formatted_questions = []
        # Generate the instruction with the option letters
        instruction = (
            f"Respond solely with {' or '.join(chr(65 + i) for i in range(number_of_presented_options))}."
            + TYPE_OF_PROMPTS[suffix]
        )
        for question, choices_pair in zip(questions, choices):
            # Ensure the question ends with proper punctuation. This is important for MMLU questions.
            if question[-1] not in ".?!:":
                question += "... "

            # Format each set of choices and append the formatted question to the result
            for choice_set in choices_pair:
                options_text = "\n".join(
                    f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choice_set)
                )
                question_text = (
                    f"{question} {instruction}\nOptions:\n{options_text}\nAnswer: "
                )
                formatted_questions.append(
                    {
                        "question": question_text,
                        "type": (
                            "permuted_question"
                            if choice_set != choices_pair[0]
                            else "original_question"
                        ),
                        "options": f"Options:\n{options_text}",
                    }
                )

        return formatted_questions


class QADataloader:
    def __init__(
        self, dataset: QADataset, batch_size: int, num_workers: int, shuffle: bool
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def return_dataloader(self):
        return self.dataloader
