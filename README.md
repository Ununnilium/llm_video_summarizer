# Create Articles from YouTube Videos

1. Install Ollama on your computer: https://ollama.com/download
   For an Intel GPU, you can use [this version](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/ollama_portable_zip_quickstart.md)
2. This application uses pyannote-audio. You have to create a Huggingface account, accept the license agreement 
   [here](https://huggingface.co/pyannote/segmentation-3.0) and [here](https://huggingface.co/pyannote/speaker-diarization-3.1),
   create a Huggingface token and put it the environmental variable `HUGGINGFACE_TOKEN`.
3. You can pre-download some LLMs with
   ```bash
   ollama pull mistral-small:24b-instruct-2501-q4_K_M
   ollama pull granite3.2:8b-instruct-q4_K_M
   ollama pull phi4:14b-q4_K_M
    ```
4. Clone this repo and create a new environment with uv:
   ```bash
   git clone https://github.com/Ununnilium/llm_video_summarizer
   cd llm_video_summarizer
   uv sync
   ```
5. Check the usage with `python video_summarizer.py --help`:
   ```
   usage: video_summarizer.py [-h] -u URL [-p PARTS] [-a ARTICLE] [-c CLEAN] [-o OUT]
   
   Summarize a YouTube video locally with Ollama.
   
   options:
     -h, --help            show this help message and exit
     -u URL, --url URL     Video URL, e.g. https://www.youtube.com/watch?v=... (default: None)
     -p PARTS, --parts PARTS
                           Ollama model for video parts summaries. Working models best to lowest quality: mistral-small > phi4 > granite3.2:8b > granite3.2:2b (default: mistral-small)
     -a ARTICLE, --article ARTICLE
                           Ollama model to write article from video parts summaries. Working models best to lowest quality: mistral-small > phi4 > granite3.2:8b (default: mistral-small)
     -c CLEAN, --clean CLEAN
                           Ollama model to clean video description. Working models best to lowest quality: mistral-small > phi4 (default: mistral-small)
     -o OUT, --out OUT     Output directory. If not provided: <parts_model>_<article_model> (default: None)
   ```
