from __future__ import annotations
import http
import json
import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import kitoken
import requests
import tiktoken

logger = logging.getLogger("video_summarizer")

_file_dir = Path(__file__).parent


class LlmProvider:
    def __init__(self, model_config: dict[str, Any], out_dir: Path) -> None:
        self.model = model_config["name"]
        self._model_config = model_config
        if "options" not in self._model_config:
            self._model_config["options"] = {}
        self._encoding: None | tiktoken.Encoding | kitoken.Kitoken = None
        self._out_dir = out_dir
        self._base_url = self._model_config["base_url"].rstrip("/")

    @staticmethod
    def get_llm_provider(model: str, out_dir: Path) -> LlmProvider:
        with (_file_dir / "model_config.json").open(encoding="utf-8") as f:
            model_config: dict[str, dict[str, Any]] = json.load(f)
        if model not in model_config:
            raise ValueError(f"Model {model} not found in model_config.json")
        if model_config[model]["api"] == "ollama":
            return OllamaProvider(model_config[model], out_dir)
        else:
            return OpenAiProvider(model_config[model], out_dir)

    def call_llm(
        self,
        name: str,
        system_prompt: str,
        user_prompt: str,
        output_tokens: int,
    ) -> str:
        pass

    def encode(self, string: str) -> list[int]:
        return self._get_encoding().encode(string)

    def decode(self, encoded_string: list[int]) -> str:
        dec = self._get_encoding().decode(encoded_string)
        if isinstance(dec, bytes):
            return dec.decode("utf-8")
        return dec

    def _model_name_to_tokenizer_file(self) -> str:
        pass

    def _get_encoding(self) -> tiktoken.Encoding | kitoken.Kitoken:
        if self._encoding is None:
            _file_dir = Path(__file__).parent
            if self.model.startswith("mistral-"):
                self._encoding = kitoken.Kitoken.from_tekken_file(
                    str(_file_dir / "tokenizer" / "tekken.json")
                )
            else:
                tokenizer_path = (
                    _file_dir / "tokenizer" / self._model_name_to_tokenizer_file()
                )
                if tokenizer_path.exists():
                    logger.debug(f"Using custom tokenizer for {self.model}...")
                    self._encoding = kitoken.Kitoken.from_tokenizers_file(
                        str(tokenizer_path)
                    )
                else:
                    logger.info("Using default tokenizer...")
                    self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def count_tokens(self, string: str) -> int:
        return len(self.encode(string))


class OllamaProvider(LlmProvider):
    def __init__(self, model_config: dict[str, Any], out_dir: Path) -> None:
        super().__init__(model_config, out_dir)

    @staticmethod
    def _tokens_per_seconds(response: dict[str, Any]) -> float:
        return response["eval_count"] / response["eval_duration"] * 1e9

    def _model_name_to_tokenizer_file(self) -> str:
        return f"{self.model.split(':')[0]}.json"

    def call_llm(
        self,
        name: str,
        system_prompt: str,
        user_prompt: str,
        output_tokens: int,
    ) -> str:
        system_prompt_tokens = self.count_tokens(system_prompt)
        user_prompt_tokens = self.count_tokens(user_prompt)
        ctx_len = output_tokens + system_prompt_tokens + user_prompt_tokens
        model_ctx_len = self._model_config["max_context_len"]
        if ctx_len > model_ctx_len:
            logger.warning(
                f"Model's context length of {model_ctx_len} too short, needed {ctx_len}..."
            )
            ctx_len = model_ctx_len
        logger.info(f"Calling {self.model} for '{name}' (ctx_len={ctx_len})...")
        start_ts = time.time()
        data = {
            "model": self.model,
            "stream": False,
            "keep_alive": 5,
            "system": system_prompt,
            "options": {
                "num_ctx": ctx_len,
                "num_batch": 512,  # smaller num_batch lowers GPU memory usage and performance
            },
            "prompt": user_prompt,
        }
        resp = requests.post("http://localhost:11434/api/generate", json=data)
        while (
            resp.status_code == http.HTTPStatus.INTERNAL_SERVER_ERROR
            and data["options"]["num_batch"] > 1
        ):
            data["options"]["num_batch"] -= 64
            logger.warning(f"Reducing num_batch to {data['options']['num_batch']}...")
            resp = requests.post("http://localhost:11434/api/generate", json=data)
        resp.raise_for_status()
        response = resp.json()
        logger.info(
            f"Calling {self.model} for '{name}' took {timedelta(seconds=time.time() - start_ts)}, "
            f"{self._tokens_per_seconds(response):.1f} tokens/s, prompt tokens: {response['prompt_eval_count']} "
            f"(estimated: {system_prompt_tokens + user_prompt_tokens}), "
            f"response tokens: {response['eval_count']} (estimated {output_tokens})"
        )

        # output for debugging/analysis only
        clean_name = "".join(
            c for c in name.lower().replace(" ", "_") if c.isalnum() or c in "._- "
        )
        with (self._out_dir / f"{clean_name}_llm_call.txt").open(
            "w", encoding="utf-8"
        ) as f:
            f.write(
                f"# System Prompt ({system_prompt_tokens} tokens) #\n"
                f"{system_prompt}\n\n# User Prompt ({user_prompt_tokens} tokens)#\n"
                f"{user_prompt}\n\n"
                f"# Response ({response['eval_count']} tokens) #\n"
                f"{response['response']}"
            )
        return response["response"]


class OpenAiProvider(LlmProvider):
    def __init__(self, model_config: dict[str, Any], out_dir: Path) -> None:
        super().__init__(model_config, out_dir)
        models_info = self._get_models()
        model_names = []
        found = False
        for m in models_info:
            model_names.append(m["id"])
            if m["id"] == self.model:
                found = True
                break
        if not found:
            raise ValueError(
                f"Model {self.model} not found, available models: {model_names}"
            )

    @staticmethod
    def _tokens_per_seconds(response: dict[str, Any]) -> float:
        return response["timings"]["predicted_per_second"]

    def _model_name_to_tokenizer_file(self) -> str:
        # model name is e.g. "/models/gemma3/google_gemma-3-12b-it-IQ4_XS.gguf"
        return f"{self.model.strip('/').split('/')[1]}.json"

    def _get_models(self) -> dict[str, Any]:
        resp = requests.get(f"{self._base_url}/v1/models")
        resp.raise_for_status()
        return resp.json()["data"]

    def call_llm(
        self,
        name: str,
        system_prompt: str,
        user_prompt: str,
        output_tokens: int,
    ) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        prompt_tokens = self.count_tokens(prompt)
        ctx_len = output_tokens + prompt_tokens
        if ctx_len > self._model_config["max_context_len"]:
            logger.warning(
                f"Model's context length of {self._model_config['max_context_len']} too short, needed {ctx_len}..."
            )
        logger.info(f"Calling {self.model} for '{name}' (ctx_len={ctx_len})...")
        start_ts = time.time()
        data = {
            "prompt": prompt,
            "model": self._model_config["name"],
            "stream": False,
            "seed": 1,
            "max_tokens": output_tokens,
            **self._model_config["options"],
        }
        resp = requests.post(
            f"{self._base_url}/v1/completions", json=data, timeout=1800
        )
        resp.raise_for_status()
        response = resp.json()
        logger.info(
            f"Calling {self.model} for '{name}' took {timedelta(seconds=time.time() - start_ts)}, "
            f"{self._tokens_per_seconds(response):.1f} tokens/s, prompt tokens: {response['timings']['prompt_n']} "
            f"(estimated: {prompt_tokens}), "
            f"response tokens: {response['timings']['predicted_n']} (estimated {output_tokens})"
        )
        text = response["choices"][0]["text"]
        # output for debugging/analysis only
        clean_name = "".join(
            c for c in name.lower().replace(" ", "_") if c.isalnum() or c in "._- "
        )
        with (self._out_dir / f"{clean_name}_llm_call.txt").open(
            "w", encoding="utf-8"
        ) as f:
            f.write(
                f"# Prompt ({prompt_tokens} tokens) #\n"
                f"{prompt}\n\n"
                f"# Response ({response['timings']['predicted_n']} tokens) #\n"
                f"{text}"
            )
        return text
