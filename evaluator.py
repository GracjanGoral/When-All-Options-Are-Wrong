import logging
import os
from datetime import datetime

import pandas as pd
from answer_mapping import Mapper
from models import LLM
from qa_dataset import QADataloader, QADataset
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EvalCriticalThinking:
    def __init__(self, model: LLM, dataloader: QADataloader, mapper: Mapper) -> None:
        self.model = model
        self.mapper = mapper
        self.dataloader = dataloader

    def think(self):
        try:
            os.makedirs("evaluation_results/", exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create results folder: {str(e)}")
            raise

        type_of_dataset: str = self.dataloader.dataset.type_of_dataset
        name_of_dataset: str = self.dataloader.dataset.name_of_dataset
        name_of_evaluated_model: str = str(self.model)
        suffix: str = self.dataloader.dataset.suffix

        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results: pd.DataFrame = pd.DataFrame(
            columns=[
                "prompt",
                "permuted",
                "model_answer",
                "gold_answer",
                "adjusted_answers",
                "mapped_answer",
                "type_of_dataset",
                "suffix" "model_id",
            ]
        )
        dataloaders: DataLoader = self.dataloader.return_dataloader()

        for question in dataloaders:
            prompt: str
            permuted: str
            gold_answer: str
            adjusted_answers: str
            dataset_type: str
            presented_options: int

            (
                prompt,
                permuted,
                gold_answer,
                adjusted_answers,
                dataset_type,
                presented_options,
                options,
            ) = (
                question[0][0],
                question[1][0],
                question[2][0],
                question[3][0],
                question[4][0],
                question[5][0],
                question[6][0],
            )

            output: str = self.model.eval(prompt)
            map_answer = self.mapper.map_answer(options, output)
            result_row = {
                "prompt": prompt,
                "permuted": permuted,
                "model_answer": output,
                "gold_answer": gold_answer,
                "adjusted_answers": adjusted_answers,
                "mapped_answer": map_answer,
                "type_of_dataset": dataset_type,
                "suffix": suffix,
                "model_id": name_of_dataset,
            }
            print(f"Prompt: {prompt}")
            print(f"Gold answer: {gold_answer}")
            print(f"Model answer: {output}")
            print(f"Mapped answer: {map_answer}")
            print("-" * 1000)
            results = pd.concat(
                [results, pd.DataFrame([result_row])], ignore_index=True
            )
        end_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = f"evaluation_results/{suffix}_{name_of_dataset}_{type_of_dataset}_{name_of_evaluated_model}_{start_time}__{end_time}.csv"
        results.to_csv(results_file, index=False)
