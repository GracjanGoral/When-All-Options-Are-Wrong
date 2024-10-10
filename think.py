import argparse
import itertools
import logging

from answer_mapping import Mapper
from evaluator import EvalCriticalThinking
from mapper_selector import mapper_model_selector
from model_selector import model_selector
from models import LLM
from qa_dataset import QADataloader, QADataset


def run_thinking(
    name_of_model: str,
    temperature_of_tested_model: float,
    max_token_of_tested_model: int,
    dataloader: QADataloader,
    mapper: str,
    temperature_of_mapper: float,
    max_token_of_mapper: int,
) -> None:
    model_to_test: LLM = model_selector(
        name_of_model=name_of_model,
        temperature=temperature_of_tested_model,
        max_tokens=max_token_of_tested_model,
    )
    mapper_to_use: Mapper = mapper_model_selector(
        name_of_model=mapper,
        temperature=temperature_of_mapper,
        max_tokens=max_token_of_mapper,
    )

    # Evaluate model
    eval_model = EvalCriticalThinking(
        model=model_to_test, dataloader=dataloader, mapper=mapper_to_use
    )
    eval_model.think()
    logging.info("Evaluation completed successfully.")


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Run critical thinking evaluation")
    parser.add_argument(
        "--dataset_paths", nargs="+", type=str, help="List of paths to the datasets"
    )
    parser.add_argument(
        "--suffixes", nargs="+", type=str, help="List of suffixes for dataset files"
    )
    parser.add_argument(
        "--options_num",
        type=int,
        default=2,
        help="Number of options presented in the dataset",
    )
    parser.add_argument(
        "--with_gold_answer",
        type=bool,
        default=False,
        help="Include gold answer in the dataset",
    )
    parser.add_argument(
        "--closeness_score",
        type=int,
        default=0,
        help="Score for how close answers are to correct",
    )
    parser.add_argument(
        "--dataset_type", type=str, default="BAD", help="Type of dataset"
    )
    parser.add_argument(
        "--models", nargs="+", type=str, help="List of models to evaluate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature of the tested model"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum tokens for the tested model",
    )
    parser.add_argument(
        "--mapper", type=str, default="gpt-4", help="Mapper model to use"
    )
    parser.add_argument(
        "--mapper_temperature",
        type=float,
        default=0.0,
        help="Temperature of the mapper",
    )
    parser.add_argument(
        "--mapper_max_tokens",
        type=int,
        default=16,
        help="Maximum tokens for the mapper",
    )

    args = parser.parse_args()

    # Iterate over the Cartesian product of datasets, suffixes, and models
    for dataset_path, suffix, model in itertools.product(
        args.dataset_paths, args.suffixes, args.models
    ):
        print(
            f"Evaluating model {model} on dataset {dataset_path} with suffix {suffix}"
        )

        # Create the dataset object
        dataset = QADataset(
            path_to_dataset=dataset_path,
            suffix=suffix,
            number_of_presented_options=args.options_num,
            with_gold_answer=args.with_gold_answer,
            how_close_to_correct_answer=args.closeness_score,
            type_of_dataset=args.dataset_type,
        )

        # Create the dataloader
        dataloader = QADataloader(
            dataset=dataset, batch_size=1, num_workers=1, shuffle=False
        )

        # Run the evaluation for the current combination
        run_thinking(
            name_of_model=model,
            temperature_of_tested_model=args.temperature,
            max_token_of_tested_model=args.max_tokens,
            dataloader=dataloader,
            mapper=args.mapper,
            temperature_of_mapper=args.mapper_temperature,
            max_token_of_mapper=args.mapper_max_tokens,
        )