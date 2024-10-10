from typing import Dict, List

from answer_mapping import MapAnswerOpenAI, MapAnswerToNone, Mapper
from supported_models import MAPPER_MODELS
from tabulate import tabulate


def supported_models_for_mapping(models: Dict[str, List[str]], model_name: str = None):
    if not models:
        raise ValueError(
            "No models provided. Please check the company's website for the latest supported models."
        )

    print(
        "The following models are supported for the Mapper model, which maps the model's answers to one of the options provided:"
    )

    table_data = []
    for company, model_list in models.items():
        for model in model_list:
            table_data.append([company, model])

    print(tabulate(table_data, headers=["Company", "Model"], tablefmt="pretty"))

    if model_name:
        model_found = any(model_name in model_list for model_list in models.values())
        if not model_found:
            raise ValueError(
                f"The model '{model_name}' is not found. Please check the company's website for the latest supported models."
            )
        else:
            print(f"Model '{model_name}' is supported.")


def mapper_model_selector(
    name_of_model: str, temperature: float = 0.0, max_tokens: int = 16
) -> Mapper:
    supported_models_for_mapping(MAPPER_MODELS, name_of_model)
    if name_of_model in MAPPER_MODELS["OpenAI"]:
        return MapAnswerOpenAI(
            name_of_model=name_of_model, temp=temperature, max_tokens=max_tokens
        )
    elif name_of_model in MAPPER_MODELS["Manually"]:
        return MapAnswerToNone()
    else:
        raise ValueError(
            f"Model '{name_of_model}' is not supported. Please check the company's website for the latest supported models."
        )
