import argparse
import functools
import json
import logging
import os
import sys
import time
from datetime import timedelta, date
import multiprocessing as mp
from pathlib import Path
import warnings
from typing import Any

import langdetect
import local_ffmpeg
from iso639 import Language
import kitoken
import nltk
import pandas as pd
import requests
import tiktoken
import torch
import yt_dlp

logger = logging.getLogger(__name__)
_file_dir = Path(os.path.dirname(os.path.realpath(__file__)))


try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")


class VideoSummarizer:
    _whisper_model = "turbo"  # turbo cannot translate, therefore consider "large-v3"
    _context_limit = 16384  # could be set higher, but often quality suffers (https://github.com/NVIDIA/RULER)

    def __init__(
        self,
        url: str,
        model_chunks: str,
        model_article: str,
        model_description: str,
        output_dir: Path,
        translate: bool = False,
    ) -> None:
        """
        Initialize the VideoSummarizer instance.

        Args:
            url (str): The URL of the video to summarize.
            model_chunks (str): The model to use for summarizing video chunks.
            model_article (str): The model to use for creating the article from summarized chunks.
            model_description (str): The model to use for cleaning the video description.
            output_dir (Path): The directory to store output files.
            translate (bool, optional): Whether to translate the transcription with Whisper.  Defaults to False.
        """
        self._out_dir = Path(output_dir)
        self._out_dir.mkdir(exist_ok=True)
        self._url = url
        self._translate = translate
        self._model_chunks = model_chunks
        self._model_article = model_article
        self._model_description = model_description
        self._audio_file: Path = self._out_dir / "audio.opus"
        self._video_info: Path = self._out_dir / "video_info.json"
        self._video_title_description = self._out_dir / "video_title_description.json"
        self._transcript = self._out_dir / "transcript.json"
        self._diarization = self._out_dir / "diarization.json"
        self._full_transcript: Path = self._out_dir / "full_transcript.json"
        self._summarized_chunks: Path = self._out_dir / "summarized_chunks.json"
        self.article: Path = self._out_dir / "article.md"
        self._text: None | str = None
        self._n_speakers: None | int = None
        self._summarization: str | None = None

    @staticmethod
    @functools.cache
    def _get_encoding(model: str) -> tiktoken.Encoding | kitoken.Kitoken:
        if model.startswith("mistral-"):
            return kitoken.Kitoken.from_tekken_file(
                str(_file_dir / "tokenizer" / "tekken.json")
            )
        else:
            tokenizer_path = _file_dir / "tokenizer" / f"{model.split(':')[0]}.json"
            if tokenizer_path.exists():
                logger.debug(f"Using custom tokenizer for {model}...")
                return kitoken.Kitoken.from_tokenizers_file(str(tokenizer_path))
        return tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def _count_tokens(model: str, string: str) -> int:
        return len(VideoSummarizer._get_encoding(model).encode(string))

    def _get_video_title_description(self) -> tuple[str, str, str]:
        if not self._video_title_description.exists():
            self._clean_description()
        with self._video_title_description.open(encoding="utf-8") as f:
            video_info = json.load(f)
        return (
            video_info.get("title", ""),
            video_info.get("description", ""),
            video_info.get("channel", ""),
        )

    def _get_full_transcript(self) -> list[str, str, int]:
        if self._full_transcript.exists():
            with self._full_transcript.open(encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError(
            "Full transcript not found, run first transcription method"
        )

    def _download_audio(self) -> None:
        start_ts = time.time()
        if not local_ffmpeg.check():
            logger.info("FFmpeg not found, installing locally...")
            success, message = local_ffmpeg.install()
            if success:
                logger.info(message)  # FFmpeg installed successfully
            else:
                logger.error(f"Error: {message}")
                sys.exit(1)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.splitext(self._audio_file)[0],
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "opus",
                    "nopostoverwrites": False,
                }
            ],
            "postprocessor_args": ["-c", "copy"],
            "quiet": True,
            "sponsorblock": True,
            "sponsorblock-remove": [
                "sponsor",
                "intro",
                "outro",
                "selfpromo",
                "interaction",
                "music_offtopic",
            ],
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
            },
        }
        logger.info("Downloading audio...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            if not self._video_info.exists():
                info = ydl.extract_info(self._url, download=False)
                with self._video_info.open("w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
            if not self._audio_file.exists():
                ydl.download([self._url])
        logger.info(
            f"Audio download finished in {timedelta(seconds=time.time() - start_ts)}"
        )

    @staticmethod
    def _transcribe_proc(in_file: Path, out_json: Path, translate: bool) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        task = "translate" if translate else "transcribe"
        start_ts = time.time()
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "int8_float16"
        else:
            device = "cpu"
            compute_type = "int8"
        whisper_model = WhisperModel(
            VideoSummarizer._whisper_model, device=device, compute_type=compute_type
        )
        batched_model = BatchedInferencePipeline(model=whisper_model)
        with warnings.catch_warnings(action="ignore"):
            whisper_segments, info = batched_model.transcribe(
                str(in_file),
                batch_size=8,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                task=task,
            )
            lang = (
                "english"
                if translate
                else Language.from_part1(info.language).name.lower()
            )
        sentence_timestamps = []
        for seg in whisper_segments:
            if seg.text.strip():
                sentences = nltk.sent_tokenize(seg.text, language=lang)
                words = seg.words
                word_idx = 0
                for sentence in sentences:
                    sentence_start = None
                    sentence_end = None
                    sentence_words = sentence.split()  # Tokenize sentence into words

                    # Find the first and last words of the sentence in the word-level timestamps
                    for word in sentence_words:
                        while (
                            word_idx < len(words)
                            and words[word_idx].word.strip() != word.strip()
                        ):
                            word_idx += 1
                        if word_idx < len(words):
                            if (
                                sentence_start is None
                            ):  # Assign start time of the first word in the sentence
                                sentence_start = words[word_idx].start
                            sentence_end = words[
                                word_idx
                            ].end  # Update end time with the current word's end time
                            word_idx += 1
                    if sentence_start is not None and sentence_end is not None:
                        sentence_timestamps.append(
                            (sentence_start, sentence_end, sentence)
                        )
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(sentence_timestamps, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Transcription finished in {timedelta(seconds=time.time() - start_ts)}"
        )

    @staticmethod
    def _diarization_proc(in_file_audio, out_file: Path) -> None:
        from pyannote.audio import Pipeline
        import torch

        def load_pipeline_from_pretrained() -> Pipeline:
            path_to_config = (
                Path(__file__).parent / "models" / "pyannote_diarization_config.yaml"
            )
            cwd = Path.cwd().resolve()  # store current working directory

            # first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
            cd_to = path_to_config.parent.parent.resolve()
            os.chdir(cd_to)
            pipeline = Pipeline.from_pretrained(path_to_config)
            os.chdir(cwd)
            return pipeline

        speechbrain_logger = logging.getLogger("speechbrain")
        speechbrain_logger.setLevel(logging.WARNING)

        start_ts = time.time()
        with warnings.catch_warnings(action="ignore"):
            pipeline = load_pipeline_from_pretrained()
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            diarization_result = pipeline(str(in_file_audio))  # speaker diarization
        diarization_json = []
        for s, _, speaker in diarization_result.itertracks(yield_label=True):
            diarization_json.append((s.start, s.end, speaker))
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(diarization_json, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Speaker diarization finished in {timedelta(seconds=time.time() - start_ts)}"
        )

    def _combine_transcript_diarization(
        self,
        transcription_segments: list[
            tuple[float, float, str]
        ],  # segments: start, end, text
        diarization: list[tuple[float, float, str]],  # start, end, speaker label
    ) -> None:
        df_text_segments = pd.DataFrame(
            transcription_segments, columns=["start_time_s", "end_time_s", "text"]
        )
        df_speakers = pd.DataFrame(
            diarization, columns=["start_time_s", "end_time_s", "speaker"]
        )

        def overlap(row1: pd.Series, row2: pd.Series) -> float:
            """Calculate overlap between two intervals."""
            overlap_start = max(row1["start_time_s"], row2["start_time_s"])
            overlap_end = min(row1["end_time_s"], row2["end_time_s"])
            return max(0.0, overlap_end - overlap_start)

        def find_best_speaker(text_row: pd.Series, speaker_df: pd.DataFrame) -> str:
            """For each text segment, find the speaker with the highest overlap"""
            speaker_df["overlap"] = speaker_df.apply(
                lambda speaker_row: overlap(text_row, speaker_row), axis=1
            )
            best_speaker_row = speaker_df.loc[speaker_df["overlap"].idxmax()]
            return best_speaker_row["speaker"]

        df_text_segments["best_speaker"] = df_text_segments.apply(
            lambda row: find_best_speaker(row, df_speakers), axis=1
        )
        df_text_segments["n_tokens"] = df_text_segments.text.apply(
            lambda x: self._count_tokens(self._model_chunks, x)
        )
        self._n_speakers = df_speakers["speaker"].nunique()
        with self._full_transcript.open("w", encoding="utf-8") as f:
            df_text_segments[["best_speaker", "text", "n_tokens"]].to_json(
                f, orient="values", indent=2, force_ascii=False
            )

    def _transcribe(self) -> None:
        logger.info("Transcribing and diarize video with Whisper...")
        start_ts = time.time()
        ctx = mp.get_context("spawn")
        processes = []
        if not self._transcript.exists():
            p_whisper = ctx.Process(
                target=VideoSummarizer._transcribe_proc,
                args=(self._audio_file, self._transcript, self._translate),
            )
            p_whisper.start()
            processes.append(p_whisper)
        if not self._diarization.exists():
            p_diar = ctx.Process(
                target=VideoSummarizer._diarization_proc,
                args=(self._audio_file, self._diarization),
            )
            p_diar.start()
            processes.append(p_diar)
        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("Transcription or diarization failed")
        with self._transcript.open(encoding="utf-8") as f:
            transcript_res = json.load(f)
        with self._diarization.open(encoding="utf-8") as f:
            diarization_res = json.load(f)
        self._combine_transcript_diarization(transcript_res, diarization_res)
        logger.info(
            f"Transcription finished in {timedelta(seconds=time.time() - start_ts)}"
        )

    def _create_text_from_list(
        self, sentence_list: list[list[str, str, int] | tuple[str, str, int]]
    ) -> str:
        if self._n_speakers == 1:
            return "\n".join([f"\n{text.lstrip()}" for _, text, _ in sentence_list])

        previous_speaker = None
        all_texts = []
        for speaker, text, _ in sentence_list:
            if previous_speaker == speaker:
                all_texts[-1] = f"{all_texts[-1]}\n{text.lstrip()}"
            else:
                speaker_name = speaker.replace("SPEAKER_0", "Speaker ").replace(
                    "SPEAKER_", "Speaker "
                )
                all_texts.append(f"\n{speaker_name}: {text.lstrip()}")
                previous_speaker = speaker
        return "\n".join(all_texts)

    def _split_for_summarize(self, max_tokens_seg: int, overlap: int) -> list[str]:
        all_segments = []
        current_text = []
        sum_tokens = 0
        with self._full_transcript.open(encoding="utf-8") as f:
            full_transcript = json.load(f)
        for i, (speaker, text, n_tokens) in enumerate(full_transcript):
            sum_tokens += n_tokens
            if sum_tokens < max_tokens_seg:
                current_text.append((speaker, text, n_tokens))
            else:
                all_segments.append(self._create_text_from_list(current_text))
                current_text = current_text[-overlap:]
                sum_tokens = 0
        all_segments.append(self._create_text_from_list(current_text))
        return all_segments

    @staticmethod
    def _trim_to_full_sentence(string: str) -> str:
        lang = Language.from_part1(langdetect.detect(string)).name.lower()
        sentences = nltk.sent_tokenize(string, language=lang)
        return " ".join(sentences[1:])

    @staticmethod
    def _tokens_per_seconds(response: dict[str, Any]) -> float:
        return response["eval_count"] / response["eval_duration"] * 1e9

    def _clean_description(self) -> None:
        if self._video_title_description.exists():
            return

        with self._video_info.open(encoding="utf-8") as f:
            video_info = json.load(f)

        system = """You are a Large Language Model (LLM) to clean up YouTube video descriptions from irrelevant 
        text like ads and self-promotion. Don't produce any output except the cleaned up video description."""
        user_content = f'''The YouTube video title is "{video_info["title"]}" and the description is:
        This is the video description to clean:
        ```
        {video_info["description"]}
        ```
        '''
        description = self._call_llm(
            "Clean video description",
            system,
            user_content,
            self._model_description,
            output_tokens=1000,
        )
        with self._video_title_description.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "title": video_info["title"],
                    "description": description,
                    "channel": video_info["uploader"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _summarize_chunks(self) -> None:
        max_output_tokens = 2000
        text_chunks = self._split_for_summarize(max_tokens_seg=4000, overlap=10)
        if self._n_speakers > 1:
            speakers = ", ".join(
                f'"Speaker {i:02d}: "' for i in range(self._n_speakers)
            )
            _, _, channel = self._get_video_title_description()
            speaker_info = (
                f"The video has {self._n_speakers} speakers. In the transcription, {speakers} was added"
                f" by pyannote for speaker diarization. "
                "In the video segment description, address which speaker says what. "
                f"The video is from the YouTube channel '{channel}', therefore this could the name of "
                f"one of the speakers. "
                f"If you can find out the speakers names, write at the beginning something like "
                f'"Speaker 1 (Prof. Dr. James Bush)" (if the name of Speaker 1 is probably Prof. Dr. James Bush). '
            )
        else:
            speaker_info = ""
        system_content = (
            "You are a Large Language Model (LLM) that creates a detailed descriptions from YouTube video "
            f"transcription segments. {speaker_info}"
            "Your description must cover all statements and points in detail from the video transcription segment. "
            "Make sure even small details discussed in the video are in your description. "
            "Your response must be only in English (translate all parts/words in other languages to English). "
            "It should only include the description, do not provide any further "
            "explanation and don't create any titles. As context, use the provided descriptions of the previous "
            "segments (if available), but take care to not repeat anything already described there. Do not repeat "
            "yourself."
        )
        previous_summary = ""
        summarized_chunks = []

        for i, chunk in enumerate(text_chunks):
            # Calculate available tokens for previous summary
            system_tokens = self._count_tokens(self._model_chunks, system_content)
            user_content = (
                f"Create the description of the following video transcript segment number {i + 1} of "
                f"{len(text_chunks)}:\n{chunk}"
            )
            user_prompt_tokens = self._count_tokens(self._model_chunks, user_content)
            available_tokens = (
                self._context_limit
                - max_output_tokens
                - system_tokens
                - user_prompt_tokens
            )
            previous_summary_tokens = self._count_tokens(
                self._model_chunks, previous_summary
            )

            if previous_summary_tokens > available_tokens:
                encoded_previous = self._get_encoding(self._model_chunks).encode(
                    previous_summary
                )
                trimmed_previous = self._get_encoding(self._model_chunks).decode(
                    encoded_previous[-available_tokens:]
                )
                previous_summary = self._trim_to_full_sentence(trimmed_previous)

            system = (
                f"{system_content}\n\n## Previous descriptions ##\n{previous_summary}"
            )
            chunk_sum = self._call_llm(
                f"Chunk {i + 1}/{len(text_chunks)}",
                system,
                user_content,
                self._model_chunks,
                max_output_tokens,
            )
            summarized_chunks.append(chunk_sum)
            previous_summary = " ".join(summarized_chunks)
        with self._summarized_chunks.open("w", encoding="utf-8") as f:
            json.dump(summarized_chunks, f, ensure_ascii=False, indent=2)

    def _create_article_from_chunks(self) -> None:
        with self._summarized_chunks.open(encoding="utf-8") as f:
            summarized_chunks = json.load(f)
        combined_content = "\n\n".join(s for i, s in enumerate(summarized_chunks))
        title, description, channel = self._get_video_title_description()
        user_prompt = (
            f"## Official YouTube Channel Name ##\n{channel}\n\n## Official video online title ##"
            f"\n{title}\n\n## Official video online description ##\n{description}\n\n"
            f"## Video description ##\n{combined_content}"
        )
        user_prompt_tokens = self._count_tokens(self._model_article, user_prompt)
        max_output_tokens = round(user_prompt_tokens * 1.2)
        if self._n_speakers > 1:
            speaker_info = (
                f"The video has {self._n_speakers} speakers. In the transcription, 'Speaker ...' was added"
                f" by pyannote for speaker diarization. "
                "If you can guess the names of the speakers, use the full names instead of 'Speaker ...'. "
                f"Address which speaker says what. The YouTube online description could also contain names of "
                " speakers. "
            )
        else:
            speaker_info = ""
        system = (
            "You are a Large Language Model (LLM) that creates a cohesive, long, detailed article "
            "(with title, introduction and subtiles) from a YouTube video. "
            "Your knowledge base was last updated at 2023 or even before. "
            f"The current date is {date.today().isoformat()}. The video was transcribed with Whisper. {speaker_info}"
            "The video was chunked arbitrary by token size in segments and each segment was described with an LLM. "
            "Use the YouTube online title, YouTube online description and the concatenate LLM video segment "
            "descriptions. "
            "Your article must be extensive to cover all topics in detail and contain even small details discussed in "
            "the video. "
            "Your response must be only in English (translate all parts/words in other languages to English). "
            "The article should be formatted as Markdown document. Do not create hyperlinks in the article. "
            "Do not add your own explanations or disclaimers (the only output should be the article) and make "
            "sure to not repeat yourself."
        )
        article = self._call_llm(
            "Article", system, user_prompt, self._model_article, max_output_tokens
        )
        with self.article.open("w", encoding="utf-8") as f:
            f.write(article)

    def _call_llm(
        self,
        name: str,
        system_prompt: str,
        user_prompt: str,
        model: str,
        output_tokens: int,
    ) -> str:
        system_prompt_tokens = self._count_tokens(model, system_prompt)
        user_prompt_tokens = self._count_tokens(model, user_prompt)
        ctx_len = output_tokens + system_prompt_tokens + user_prompt_tokens
        model_ctx_len = self._get_model_ctx_len(model)
        if ctx_len > model_ctx_len:
            logger.warning(
                f"Model's context length of {model_ctx_len} too short, needed {ctx_len}..."
            )
            ctx_len = model_ctx_len
        logger.info(f"Calling {model} for '{name}' (ctx_len={ctx_len})...")
        start_ts = time.time()
        data = {
            "model": model,
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
        while resp.status_code == 500 and data["options"]["num_batch"] > 1:
            data["options"]["num_batch"] -= 64
            logger.warning(f"Reducing num_batch to {data['options']['num_batch']}...")
            resp = requests.post("http://localhost:11434/api/generate", json=data)
        resp.raise_for_status()
        response = resp.json()
        logger.info(
            f"Calling {model} for '{name}' took {timedelta(seconds=time.time() - start_ts)}, "
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

    def _create_dirname(self, title: str) -> str:
        system = "You are an LLM that creates a single, short folder name (snakecase) from a video title. Don't output anything else."
        prompt = f'Video title to create folder name: "{title}"'
        return self._call_llm(
            "Directory name", system, prompt, self._model_description, output_tokens=100
        )

    @staticmethod
    def _get_model_ctx_len(model: str) -> int:
        resp = requests.post("http://localhost:11434/api/show", json={"model": model})
        if resp.status_code == 404:
            logger.info(f"Pulling model {model}...")
            resp2 = requests.post(
                "http://localhost:11434/api/pull",
                json={"model": model, "stream": False},
            )
            if resp2 != 200:
                raise RuntimeError(
                    f"Could not pull model {model} ({resp2.text}), please try manually 'ollama pull {model}'"
                )
            resp = requests.post(
                "http://localhost:11434/api/show", json={"model": model}
            )
        resp.raise_for_status()
        model_info = resp.json()
        return model_info["model_info"][
            f"{model_info['details']['family']}.context_length"
        ]

    def summarize(self) -> str:
        self._download_audio()
        self._transcribe()
        self._summarize_chunks()
        self._create_article_from_chunks()
        return self._summarization


def main() -> None:
    def _clean_model_name(name: str) -> str:
        return name.replace(":", "@").rsplit("/", 1)[-1]

    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    parser = argparse.ArgumentParser(
        description="Summarize a YouTube video locally with Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        help="Video URL, e.g. https://www.youtube.com/watch?v=...",
        required=True,
    )

    parser.add_argument(
        "-p",
        "--parts",
        type=str,
        default="mistral-small",
        help="Ollama model for video parts summaries. Working models best to lowest quality: mistral-small > phi4 > granite3.2:8b > granite3.2:2b",
    )
    parser.add_argument(
        "-a",
        "--article",
        help="Ollama model to write article from video parts summaries. Working models best to lowest quality: mistral-small > phi4 > granite3.2:8b",
        type=str,
        default="mistral-small",
    )
    parser.add_argument(
        "-c",
        "--clean",
        help="Ollama model to clean video description. Working models best to lowest quality: mistral-small > phi4",
        type=str,
        default="mistral-small",
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Output directory. If not provided: <parts_model>_<article_model> ",
        type=Path,
    )
    args = parser.parse_args()

    out_dir = (
        args.out
        if args.out
        else Path(f"{_clean_model_name(args.parts)}_{_clean_model_name(args.article)}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    vt = VideoSummarizer(
        args.url, args.parts, args.article, args.clean, output_dir=out_dir
    )
    vt.summarize()
    logger.info(f"Article written to {vt.article}")


if __name__ == "__main__":
    main()
