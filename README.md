# Wait, that's not an option
This repository tests the robustness of Large Language Models (LLMs) using multiple-choice questions where all provided options are incorrect. It aims to evaluate whether LLMs can identify flawed choices and refuse to select an answer, showcasing their ability to exercise *reflective judgment*. This work is based on the research article **["Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options"](https://arxiv.org/abs/2409.00113)**. More details can be found on our [project website](https://sites.google.com/view/reflective-jugment/strona-g%C5%82%C3%B3wna).
## Project Setup
1. **Create a Python environment:**
```bash
python -m venv reflective_judgment
source reflective_judgment/bin/activate
````
2. **Clone the repository:**
```bash
git clone https://github.com/GracjanGoral/When-All-Options-Are-Wrong.git
cd When-All-Options-Are-Wrong
```
3. **Install requirements:**

```bash
pip install -r requirements.txt
```
4. **API Token Configuration**:

Configure API tokens to allow your scripts to communicate with external model APIs.
This setup is essential for integrating services such as OpenAI, Google (Gemini), Anthropic, and Replicate.

```bash
# Example configuration for multiple API tokens
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
export GOOGLE_API_KEY="your_google_api_key_here"
export REPLICATE_API_TOKEN="your_replicate_api_token_here"
```
**Note:**
If you want to avoid manually mapping model answers, you can utilize another language model like GPT-4.
To do so, you will need to provide an OpenAI token (currently, only GPT models are supported for mapping).
Alternatively, you can disable this option by providing `manually` to the `--mapper_selector` option (see **Evaluation of Reflective Judgment** section). This will result in all answers being mapped to `None`.
# Dataset Format/Construction

Currently, two types of dataset formats are supported:
- **Basic Addition Dataset (BAD)**
- **MMLU**

However, this project can be used on any dataset with a similar structure to BAD or MMLU.

## Dataset Examples

>### Dataset Example: BAD
>- **Question:** What is 0+0? Respond solely with A or B.
>- **Options:**
>  - A. 1
>  - B. 0

> ### Dataset Example: MMLU
> - **Question:** The period of a 10-Hz wave is ... Respond solely with A or B.
> - **Options:**
>   - A. 1/10 s
>   - B. 2/7 s

Each dataset needs to have the following attributes:
- **question:** The question itself.
- **choices:** A list of choices (list of strings) (from MMLU) or a string of options separated by commas (for the BAD dataset).
- **answer:** The correct answer (or `None` if all options are incorrect).

> **Note:** In the future, all choices will be in the format of a list of strings.

## Evaluation of Reflective Judgment

```bash
 python3 think.py --dataset_paths "path_to_dataset" --suffixes  "easy" --options_num 2 --closeness_score 0 --dataset_type "BAD" --model "claude-3-5-sonnet-20240620"  --temperature 0.0 --max_tokens 128 --mapper "gpt-4" --mapper_temperature 0.0 --mapper_max_tokens 16
```

- `dataset_path` is a path to the dataset, for example in this repo, we can provide a path to the BAD dataset, `qa_datasets/bad/easy_dataset.json`.

- `suffix` is a type of prompt added to the question. Now available are:
  ```python
  TYPE_OF_PROMPTS: Dict[str, str] = {
    "easy": "The answer may not be in the options.",
    "baseline": "",
    "hard": "You must choose exactly one option."
  }
    ```
>**Note:** If you want to add a new suffix, just put it in this dictionary.

- **with_gold_answer**: Indicates if a gold answer is provided in the options.

- **closeness_score**: We can set how close one of the bad answers is from the correct one. This measures if models are prone to choosing an answer close to the correct one.

- **model**: This is the model which we evaluated (for example, gpt-4).

- **temperature**: This is the temperature of the model that we evaluated.

- **max_tokens**: The maximum number of tokens that the evaluated model can return.

- **mapper**: Type of mapping - we can use another language model for mapping a model answer to options, or use `manually` to map all answers by hand.

- **mapper_temperature**: Temperature of the model for mapping. If mapping is done `manually`, this is not used.

- **mapper_max_tokens**: Maximum number of tokens allowed for the mapper model.
