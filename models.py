import os
from abc import ABC, abstractmethod
from time import sleep
from typing import Callable, Dict, Generator, List

import google.generativeai as genai
import requests
from anthropic import Anthropic
from loguru import logger
from openai import OpenAI
from replicate import Client


class LLM(ABC):
    """
    Abstract base class for Language Model interfaces.
    """

    def __init__(
        self, model_name: str, temperature: float = 0.0, max_tokens: int = 100
    ) -> None:
        """
        Initializes the LLM instance.

        Args:
            model_name (str): The name of the model.
            temperature (float): The sampling temperature.
            max_tokens (int): The maximum number of tokens to generate.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __str__(self) -> str:
        """
        Cleans the model name by removing any prefix if present.

        Returns:
            str: The cleaned model name.
        """
        return (
            self.model_name.split("/")[1] if "/" in self.model_name else self.model_name
        )

    @abstractmethod
    def eval(self, prompt: str) -> str:
        """
        Abstract method to evaluate a prompt using the model.

        Args:
            prompt (str): The input prompt to be processed by the model.

        Returns:
            str: The generated response from the model.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError(
            "The eval method must be implemented by the subclass."
        )

    @staticmethod
    def _retry_logic(func: Callable[[], str], max_retries: int, delay: int) -> str:
        """
        Implements a retry logic for executing a function.

        Args:
            func (Callable): The function to execute with retries.
            max_retries (int): The maximum number of retries on failure.
            delay (int): The delay between retries in seconds.

        Returns:
            str: The result of the function call if successful.

        Raises:
            Exception: If the function fails after max_retries.
        """
        attempts = 0
        while attempts < max_retries:
            try:
                return func()
            except Exception as e:
                attempts += 1
                logger.info("Attempt %d failed with error: %s", attempts, e)
                sleep(delay)
                if attempts == max_retries:
                    logger.error("Max retries reached. Raising exception.")
                    raise e


class GPTModel(LLM):
    """
    GPT model implementation using the OpenAI API.
    """

    def __init__(self, model_name: str, temperature: float, max_tokens: int) -> None:
        """
        Initializes the GPTModel instance.

        Args:
            model_name (str): The name of the GPT model.
            temperature (float): The sampling temperature.
            max_tokens (int): The maximum number of tokens to generate.
        """
        super().__init__(model_name, temperature, max_tokens)
        try:
            self.api_key: str = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise KeyError("The 'OPENAI_API_KEY' environment variable is not set.")
        self.open_ai = OpenAI(api_key=self.api_key)

    def eval(self, prompt: str, max_retries: int = 50, delay: int = 1) -> str:
        """
        Evaluates a prompt using the GPT model with retry logic.

        Args:
            prompt (str): The input prompt to be processed.
            max_retries (int): The maximum number of retries on failure.
            delay (int): The delay between retries in seconds.

        Returns:
            str: The generated response from the model.

        Raises:
            Exception: If the model fails to generate a response after max_retries.
        """
        return self._retry_logic(
            lambda: self._generate_response(prompt), max_retries, delay
        )

    def _generate_response(self, prompt: str) -> str:
        if self.model_name in ["o1-preview", "o1-mini"]:
            response = self.open_ai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            return response.choices[0].message.content
        else:
            response = self.open_ai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=0,
            )
            return response.model_dump()["choices"][0]["message"]["content"]


class ReplicateModel(LLM):
    """
    Replicate model implementation using the Replicate API.
    """

    def __init__(self, model_name: str, temperature: float, max_tokens: int) -> None:
        """
        Initializes the ReplicateModel instance.

        Args:
            model_name (str): The name of the Replicate model.
            temperature (float): The sampling temperature.
            max_tokens (int): The maximum number of tokens to generate.
        """
        super().__init__(model_name, temperature, max_tokens)
        try:
            self.api_key: str = os.environ["REPLICATE_API_TOKEN"]
        except KeyError:
            raise KeyError("The 'REPLICATE_API_TOKEN' environment variable is not set.")
        self.replicate_client = Client(api_token=self.api_key)

    def eval(self, prompt: str, max_retries: int = 50, delay: int = 1) -> str:
        """
        Evaluates a prompt using the Replicate model with retry logic.

        Args:
            prompt (str): The input prompt to be processed.
            max_retries (int): The maximum number of retries on failure.
            delay (int): The delay between retries in seconds.

        Returns:
            str: The generated response from the model.

        Raises:
            Exception: If the model fails to generate a response after max_retries.
        """
        prompt_dict = self._create_prompt_dict(prompt)
        return self._retry_logic(
            lambda: self._generate_response(prompt_dict), max_retries, delay
        )

    def _create_prompt_dict(self, prompt: str) -> dict:
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "system_prompt": "",
            "prompt": prompt,
            "seed": 0,
        }

    def _generate_response(self, prompt_dict: dict) -> str:
        output: Generator = self.replicate_client.run(
            self.model_name, input=prompt_dict
        )
        return "".join(output)


class AnthropicModel(LLM):
    """
    Anthropic model implementation using the Anthropic API.
    """

    def __init__(self, model_name: str, temperature: float, max_tokens: int) -> None:
        """
        Initializes the AnthropicModel instance.

        Args:
            model_name (str): The name of the Anthropic model.
            temperature (float): The sampling temperature.
            max_tokens (int): The maximum number of tokens to generate.
        """
        super().__init__(model_name, temperature, max_tokens)
        try:
            self.api_key: str = os.environ["ANTHROPIC_API_KEY"]
        except KeyError:
            raise KeyError("The 'ANTHROPIC_API_KEY' environment variable is not set.")
        self.anthropic_client = Anthropic(api_key=self.api_key)

    def eval(self, prompt: str, max_retries: int = 50, delay: int = 1) -> str:
        """
        Evaluates a prompt using the Anthropic model with retry logic.

        Args:
            prompt (str): The input prompt to be processed.
            max_retries (int): The maximum number of retries on failure.
            delay (int): The delay between retries in seconds.

        Returns:
            str: The generated response from the model.

        Raises:
            Exception: If the model fails to generate a response after max_retries.
        """
        return self._retry_logic(
            lambda: self._generate_response(prompt), max_retries, delay
        )

    def _generate_response(self, prompt: str) -> str:
        message = self.anthropic_client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system="",
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class GoogleModel(LLM):
    """
    Google model implementation using the Google Generative AI API.
    """

    def __init__(self, model_name: str, temperature: float, max_tokens: int) -> None:
        """
        Initializes the GoogleModel instance.

        Args:
            model_name (str): The name of the Google model.
            temperature (float): The sampling temperature.
            max_tokens (int): The maximum number of tokens to generate.
        """
        super().__init__(model_name, temperature, max_tokens)
        try:
            self.api_key: str = os.environ["GOOGLE_API_KEY"]
        except KeyError:
            raise KeyError("The 'GOOGLE_API_KEY' environment variable is not set.")
        genai.configure(api_key=self.api_key)
        self.google_client = genai.GenerativeModel(self.model_name)

    def eval(self, prompt: str, max_retries: int = 50, delay: int = 1) -> str:
        """
        Evaluates a prompt using the Google model with retry logic.

        Args:
            prompt (str): The input prompt to be processed.
            max_retries (int): The maximum number of retries on failure.
            delay (int): The delay between retries in seconds.

        Returns:
            str: The generated response from the model.

        Raises:
            Exception: If the model fails to generate a response after max_retries.
        """
        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        return self._retry_logic(
            lambda: self._generate_response(prompt, safety_settings), max_retries, delay
        )

    def _generate_response(self, prompt: str, safety_settings: list) -> str:
        response = self.google_client.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                stop_sequences=["x"],
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        return response.text


class MetaModel(LLM):
    """
    Meta model implementation using the DeepInfra API.
    """

    def __init__(self, model_name: str, temperature: float, max_tokens: int) -> None:
        """
        Initializes the MetaModel instance.

        Args:
            model_name (str): The name of the Meta model.
            temperature (float): The sampling temperature.
            max_tokens (int): The maximum number of tokens to generate.
        """
        super().__init__(model_name, temperature, max_tokens)
        try:
            self.api_key: str = os.environ["DEEPINFRA_TOKEN"]
        except KeyError:
            raise KeyError("The 'DEEPINFRA_TOKEN' environment variable is not set.")
        self.open_ai = OpenAI(
            base_url="https://api.deepinfra.com/v1/openai", api_key=self.api_key
        )

    def eval(self, prompt: str, max_retries: int = 50, delay: int = 1) -> str:
        """
        Evaluates a prompt using the Meta model with retry logic.

        Args:
            prompt (str): The input prompt to be processed.
            max_retries (int): The maximum number of retries on failure.
            delay (int): The delay between retries in seconds.

        Returns:
            str: The generated response from the model.

        Raises:
            Exception: If the model fails to generate a response after max_retries.
        """
        return self._retry_logic(
            lambda: self._generate_response(prompt), max_retries, delay
        )

    def _generate_response(self, prompt: str) -> str:
        response = self.open_ai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=0,
        )
        return response.model_dump()["choices"][0]["message"]["content"]


class NvidiaModel(LLM):
    def __init__(self, model_name: str, temperature: float, max_tokens: int) -> None:
        super().__init__(model_name, temperature, max_tokens)
        try:
            self.api_key: str = os.environ["NVIDIA_API_KEY"]
        except KeyError:
            raise KeyError("The 'NVIDIA_API_KEY' environment variable is not set.")
        self.open_ai = OpenAI(api_key=self.api_key)

    def eval(self, prompt: str, max_retries: int = 50, delay: int = 1) -> str:
        return self._retry_logic(
            lambda: self._generate_response(prompt), max_retries, delay
        )

    def _generate_response(self, prompt: str) -> str:
        url: str = "https://integrate.api.nvidia.com/v1/chat/completions"

        payload: Dict[str, str | List[Dict[str, str]] | float | int | bool] = {
            "model": self.model_name,
            "messages": [{"content": prompt, "role": "user"}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        headers: Dict[str, str] = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Bearer " + self.api_key,
        }

        response = requests.post(url, json=payload, headers=headers)

        return response.json()["choices"][0]["message"]["content"]


class MapAnswerOpenAI:
    def __init__(self, name_of_model) -> None:
        self.open_ai: OpenAI = OpenAI()
        self.model_name = name_of_model

    def create_json_number(self, input_string, model_answer):
        instruction = f"""
        Use the following rules to map the model's answer to the letters 'A', 'B', or 'C':
        1. Return 'A' or 'B' if the model gives one of the options.
        2. If the model gives a letter 'A' or 'B' but then proposes a different answer, return the letter 'C'.
        3. If the model says that there is no correct answer, return the letter 'C'.
        Respond with the letter 'A', 'B', or 'C':

        Options: {input_string}
        Model's Answer: {model_answer}
        """

        return instruction

    def map_answer(self, context_question, model_answer):
        json_result = self.create_json_number(context_question, model_answer)
        response = self.open_ai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": json_result}],
            temperature=0,
            max_tokens=64,
        )
        return response.model_dump()["choices"][0]["message"]["content"]
