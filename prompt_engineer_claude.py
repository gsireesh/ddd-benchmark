import anthropic
from dotenv import load_dotenv
import httpx
from pathlib import Path
import os, time, logging, base64, json

file_logger = logging.getLogger('claude_file_log')
stream_logger = logging.getLogger('claude_stream_log')

# Create Claude API client
load_dotenv()
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
client = anthropic.Anthropic(api_key=api_key)

def log_and_print(msg: str, exc_info: bool = False, level: str = "info"):
    """Logs and prints a message."""
    match level:
        case "info":
            file_logger.info(msg, exc_info=exc_info)
        case "warning":
            file_logger.warning(msg, exc_info=exc_info)
        case "error":
            file_logger.error(msg, exc_info=exc_info)
        case _:
            file_logger.info(msg, exc_info=exc_info)
    print(msg)

def timeit(func):
    """Decorator to time the execution of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log_and_print(f"Finished processing in {elapsed_time:.2f} seconds.")
        return result
    return wrapper

class Config:
    # --- Dataset parameters ---
    FULL_DATASET = True
    MODE:str = "html"  # pdf, xml, md, html, txt
    MODIFICATION:str = ""  # Version of the dataset
    DATASET_CATEGORY:str = "zeolite"  # zeolite or aluminum

    # --- Model parameters ---
    MODEL:str = "claude-3-7-sonnet-20250219"
    RESULT_TOKENS:int = 4000
    RETRY_LIMIT:int = 5
    RETRY_DELAY_SECONDS:int = 60
    WAIT_TIME_BETWEEN_ATTEMPTS:int = 15

    XML_CHUNK_WIDTH:int = 2048
    XML_CHUNK_STRIDE:int = 1024

    # --- Model Specific Configs ---
    RESULT_TOKENS:int = 8000
    THINKING_TOKENS:int = 1024
    MODEL_INPUT_TOKEN_LIMIT = 200000

    # --- Generate paths ---
    BASE_PATH = Path(__file__).parent
    DATA_DIR = BASE_PATH / "data" / f"{DATASET_CATEGORY.lower()}" / (MODE + MODIFICATION)
    OUT_DIR  = BASE_PATH / "claude" / f"{DATASET_CATEGORY.lower()}" / (MODE + MODIFICATION) / "responses"
    PROMTPT_PATH  = BASE_PATH / "prompts" / f"{DATASET_CATEGORY.lower()}" /  MODE / "prompt"
    LOG_DIR = BASE_PATH / "claude" / f"{DATASET_CATEGORY.lower()}" /  (MODE + MODIFICATION) / "logs"

    # --- Training set DOIs ---
    TRAIN_SET_DOIS = {
        "zeolite": [
            "10.1016/j.micromeso.2006.10.023",
            "10.1016/j.solidstatesciences.2007.08.002",
            "10.1007/s10934-015-0051-5",
            "10.1002/anie.200461911",
            "10.1007/s11244-013-0170-7",
        ],
        "aluminum": [
            "10.1016/j.scriptamat.2004.07.020",
            "10.1016/j.engfailanal.2010.08.007",
            "10.1016/j.jallcom.2013.08.214",
            "10.1007/s11661-010-0395-z",
            "10.1007/s11661-008-9739-3",
            "10.1007/s11837-016-1896-z",
        ],
    }.get(DATASET_CATEGORY, [])

def check_config():
    # --- Check for valid config parameters ---
    assert Config.TRAIN_SET_DOIS, "TRAIN_SET_DOIS is empty."
    assert Config.DATA_DIR.is_dir(), f"Data directory not found: {Config.DATA_DIR}"
    Config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    assert Config.PROMTPT_PATH.is_file(), f"Prompt file not found: {Config.PROMTPT_PATH}"
    # assert Config.MODE in ["pdf", "xml", "md", "html", "txt"], f"Invalid mode: {Config.MODE}. Expected one of ['pdf', 'xml', 'md', 'html', 'txt']."
    assert Config.RESULT_TOKENS > 0, "RESULT_TOKENS must be greater than 0."
    assert Config.THINKING_TOKENS >= 1024, "THINKING_TOKENS must be greater than or equal to 1024."

class ClaudeAPI:
    def __init__(self) -> None:
        self.filename = ""
        self.session_name = ""

    def count_tokens(self, message: list[dict]) -> int:
        """
        Counts the number of tokens in a message using the Anthropic API.
        Returns -1 if an error occurs.
        """
        try:
            response = client.messages.count_tokens(
                model=Config.MODEL,
                messages=message,   # type: ignore
            )
            return response.input_tokens
        except anthropic.APIStatusError as e:
            log_and_print(f"Anthropic API Status Error: Status Code={e.status_code}, Message={e.message}")
        except Exception as e:
            log_and_print(f"Unexpected error during token counting: {e}", exc_info=True)
        return -1
    
    def check_input_token_limit(self, message: list[dict]) -> bool:
        """
        Checks if the number of tokens in a message exceeds the model's input token limit.
        Returns True if within limit, False otherwise.
        """
        # input_tokens = self.count_tokens(message)
        # if input_tokens == -1:
        #     log_and_print("Error counting tokens. Skipping file.")
        #     return False
        # if input_tokens >= Config.MODEL_INPUT_TOKEN_LIMIT:
        #     log_and_print(f"Input tokens ({input_tokens}) exceed the model limit ({Config.MODEL_INPUT_TOKEN_LIMIT}). Skipping file.")
        #     return False
        # log_and_print(f"Input tokens: {input_tokens} (limit: {Config.MODEL_INPUT_TOKEN_LIMIT})")
        return True

    def streaming(self, message: list[dict]) -> str:
        """
        Streams the response from the Claude API for a given message.
        Returns the full response text.
        """
        full_response_text = ""

        with client.beta.messages.stream(
            model=Config.MODEL,
            max_tokens=Config.RESULT_TOKENS + Config.THINKING_TOKENS,
            messages=message,   # type: ignore
            # betas=["files-api-2025-04-14"],
            thinking={
                "type": "enabled",
                "budget_tokens": Config.THINKING_TOKENS,
            },
        ) as stream:
            for event in stream:
                match event.type:
                    case "content_block_delta" if event.delta.type == "thinking_delta":
                        stream_logger.info(event.delta.thinking)
                    case "content_block_delta" if event.delta.type == "text_delta":
                        full_response_text += event.delta.text
                        stream_logger.info(event.delta.text)
                    case "message_delta":
                        stream_logger.info(f"\nTotal token used: {event.usage.output_tokens}\n")
        return full_response_text

    def _handle_stream_exception(self, e):
        """Handles exceptions during streaming and returns True if should break, False otherwise."""
        if isinstance(e, (anthropic.RateLimitError, anthropic.APIConnectionError, httpx.RemoteProtocolError)):
            log_and_print(f"RateLimit or Connection error processing {self.filename}: {e}.")
            return False
        elif isinstance(e, anthropic.APIStatusError):
            log_and_print(
                f"Anthropic API Status Error processing {self.filename}: "
                f"Status Code={e.status_code}, Message={e.message}"
            )
            return e.status_code != 200
        else:
            log_and_print(f"Unexpected error during streaming for {self.filename}: {e}", exc_info=True)
            return True

    def retry_stream(self, message: list[dict]) -> str | None:
        """
        Retries streaming the response from the Claude API for a given message.
        Handles exceptions and retries if necessary.
        Returns the full response text or None on error.
        """
        for attempt in range(1, Config.RETRY_LIMIT + 1):
            file_logger.info("Waiting for Claude to respond...")
            try:
                response = self.streaming(message)
                if response:
                    log_and_print(f"Response received for {self.session_name}.")
                    return response
                log_and_print(f"Empty response for {self.session_name}.")
            except Exception as e:
                if self._handle_stream_exception(e):
                    return None

            if attempt < Config.RETRY_LIMIT:
                log_and_print(f"Retrying in {Config.RETRY_DELAY_SECONDS}s for {self.session_name} (Attempt {attempt}/{Config.RETRY_LIMIT})...")
                time.sleep(Config.RETRY_DELAY_SECONDS)

        log_and_print(f"Max retries reached for {self.session_name}. Skipping file.")
        return None

    @timeit
    def API_call(self, message: list[dict]) -> str | None:
        if not self.check_input_token_limit(message):
            log_and_print(f"Token limit exceeded for {self.session_name}. Skipping.")
            return None
        
        file_logger.info(f"--- Streaming response for {self.session_name} ---")
        result = self.retry_stream(message)
        file_logger.info(f"--- Finished stream for {self.session_name} ---")
        time.sleep(Config.WAIT_TIME_BETWEEN_ATTEMPTS)
        return result

    def API_call_with_PDF(self, prompts: list[str], path: Path) -> str | None:
        self.filename = path.name
        self.session_name = f"{self.filename} (PDF)"
        message = Message(prompts[0]).PDF_base64_message(path)
        return self.API_call(message)
    
    def API_call_with_XML(self, prompts: list[str], path: Path) -> str | None:
        self.filename = path.name
        prompt, secondary_prompt = prompts
        if not secondary_prompt:
            raise ValueError("Secondary prompt is empty. Cannot proceed.")
        chunk_messages = Message(prompt).XML_message(path)
        chunk_responses = Response()
        for idx, chunk_message in enumerate(chunk_messages):
            self.session_name = f"{self.filename} (chunk {idx+1})"
            log_and_print(f"--- Processing chunk {idx+1}/{len(chunk_messages)} ---")
            response = self.API_call([chunk_message])
            if response is not None:
                chunk_responses._stack_response(response)
                log_and_print("Success")
            else:
                log_and_print(f"Error processing chunk {idx+1}")
            time.sleep(Config.WAIT_TIME_BETWEEN_ATTEMPTS)
        merged_response = chunk_responses._merge_responses()

        secondary_message = Message(secondary_prompt).text_message(merged_response)
        log_and_print(f"--- Processing secondary prompt ---")
        return self.API_call(secondary_message)

    def API_call_with_text_file(self, prompts: list[str], path: Path) -> str | None:
        self.filename = path.name
        self.session_name = f"{self.filename} (text file)"
        message = Message(prompts[0]).text_based_message(path)
        return self.API_call(message)

class Message:
    def __init__(self, prompt:str):
        self.prompt = prompt

    def File_upload(self, file_path: Path, is_PDF: bool=False) -> str:
        """
        Uploads a PDF file to the Claude API and returns the file ID.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"{Config.MODE.upper()} file not found: {file_path}")
        try:
            MIME_type = "application/pdf" if is_PDF else "text/plain"
            file_upload = client.beta.files.upload(
                file=(file_path.name, open(file_path, "rb"), MIME_type),
            )
        except anthropic.APIStatusError as e:
            log_and_print(f"Anthropic API Status Error: Status Code={e.status_code}, Message={e.message}")
            raise
        except Exception as e:
            log_and_print(f"Unexpected error during file upload: {e}", exc_info=True)
            raise
        return file_upload.id

    def PDF_embed(self, file_path: Path) -> str:
        """
        Reads a PDF file and encodes its content in base64.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"{Config.MODE.upper()} file not found: {file_path}")
        content_b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        return content_b64
    
    def XML_chunker(self, file_path: Path) -> list[str]:
        """
        Splits an XML file into overlapping text chunks.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"{Config.MODE.upper()} file not found: {file_path}")
        content = file_path.read_text(encoding="utf-8")
        total_len = len(content)
        log_and_print(f"XML file size: {total_len} characters")

        width = Config.XML_CHUNK_WIDTH
        stride = Config.XML_CHUNK_STRIDE

        return [content[start:start + width] for start in range(0, len(content), stride)]

    def text_message(self, text:str) -> list[dict]:
        """
        Creates a message for the Claude API using a text input along with a prompt.
        """
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "text", "text": self.prompt}
            ]
        }]
    
    def PDF_base64_message(self, file_path: Path) -> list[dict]:
        """
        Creates a message for the Claude API using a PDF file and a prompt.
        The PDF is read as base64 and attached in the message according to the provider's format.
        """
        content_b64 = self.PDF_embed(file_path)
        return [{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": content_b64,
                    }
                },
                {"type": "text", "text": self.prompt}
            ]
        }]

    def PDF_message(self, file_path: Path) -> list[dict]:
        """
        Creates a message for the Claude API using a PDF file and a prompt.
        The PDF is uploaded to the API and the file ID is included in the message.
        """
        file_id = self.File_upload(file_path, is_PDF=True)
        return [{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "id": file_id,
                    }
                },
                {"type": "text", "text": self.prompt}
            ]
        }]
    
    def XML_message(self, file_path: Path) -> list[dict]:
        """
        Splits an XML file into overlapping text chunks and creates API messages for each chunk.
        """
        return [self.text_message(chunk)[0] for chunk in self.XML_chunker(file_path)]
    
    def text_based_message(self, file_path: Path) -> list[dict]:
        """
        Creates a message for the Claude API using a text-based file (e.g., .txt, .md, .html).
        The file is read as text and included in the message.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"{Config.MODE.upper()} file not found: {file_path}")
        content = file_path.read_text(encoding="utf-8")
        return self.text_message(content)

    def text_based_file_message(self, file_path: Path) -> list[dict]:
        """
        Creates a message for the Claude API using a text-based file (e.g., .txt, .md, .html) and a prompt.
        The file is read as text and included in the message.
        """
        fileID = self.File_upload(file_path, is_PDF=False)
        return [{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": fileID,
                    },
                },
                {"type": "text", "text": self.prompt}
            ]
        }]

class Response:
    def __init__(self):
        self.response_collection:dict[str,list[str]] = {
            "table csv": [],
            "extracted text": [],
            "property mapping": [],
            "formula abbreviations": [],
        }
        self.response_aggregation:dict[str,str] = {
            "table csv": "",
            "extracted text": "",
            "property mapping": "",
            "formula abbreviations": "",
        }
        
    def _parse_json_response(self, response_text) -> dict[str, str] | None:
        """
        Extracts and parses the first valid JSON object from the response text.
        Returns the parsed dictionary, or None if parsing fails.
        """
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(response_text):
            try:
                obj, end = decoder.raw_decode(response_text, idx)
                if isinstance(obj, dict):
                    return obj
                idx = end
            except json.JSONDecodeError:
                idx += 1
        return None
    
    def _stack_response(self, response_text: str) -> None:
        """
        Stacks the responses from multiple chunks into a single response.
        """
        chunk_info = self._parse_json_response(response_text)
        if chunk_info is None: return
        for key in self.response_collection:
            if key not in chunk_info: continue
            self.response_collection[key].append(chunk_info[key])

    def _merge_responses(self) -> str:
        """
        Merges the stacked responses into a single JSON string.
        """
        for key in self.response_collection:
            self.response_aggregation[key] = "\n".join(self.response_collection[key])
        merged_response = json.dumps(self.response_aggregation, indent=4)
        file_logger.info(f"Merged response: {merged_response}")
        return merged_response

def process_file(prompts: list[str], path: Path) -> str | None:
    """
    Processes a single PDF/XML file using the Claude API via streaming.
    Handles both PDF and XML (with chunking and secondary prompt).
    Returns the final response or None on error.
    """
    match Config.MODE:
        case "md" | "html" | "txt":
            return ClaudeAPI().API_call_with_text_file(prompts, path)
        case "pdf":
            return ClaudeAPI().API_call_with_PDF(prompts, path)
        case "xml":
            # if not prompts or len(prompts) < 2:
            #     log_and_print("Insufficient prompts provided for XML processing. Expected at least 2 prompts.")
            #     return None
            return ClaudeAPI().API_call_with_text_file(prompts, path)
        case _:
            log_and_print(f"Unsupported mode: {Config.MODE}")
            return None

def process_files(file_paths: list[Path]) -> int:
    """
    Processes a list of files using Anthropic's Claude API via streaming.
    Logs status and handles output file management.
    Returns the number of files that encountered errors.
    """
    error_count = 0
    total_files = len(file_paths)
    
    prompt = load_prompt(Config.PROMTPT_PATH, "Primary")
    prompts = [prompt, ]

    
    stream_handler = None
    stream_formatter = logging.Formatter('%(message)s')


    for idx, path in enumerate(file_paths, 1):
        filename = path.name
        safe_doi = path.stem
        out_path = Config.OUT_DIR / f"response_{safe_doi}.json"

        if stream_handler is not None:
            stream_logger.removeHandler(stream_handler)
            stream_handler.close()
        stream_handler = logging.FileHandler(Config.LOG_DIR / f"{safe_doi}.log", encoding='utf-8', mode='a')
        stream_handler.setFormatter(stream_formatter)
        stream_handler.terminator = ""
        stream_logger.addHandler(stream_handler)
        stream_logger.setLevel(logging.INFO)
        stream_logger.info(f"--- Processing file {idx}/{total_files}: {filename} ---\n")
        # break

        log_and_print(f"Processing file {idx}/{total_files}: {filename}...")

        if out_path.exists():
            if out_path.stat().st_size > 0:
                log_and_print("Output file already exists. Skipping file.")
                continue    # output already exists and is non-empty
            else:
                log_and_print("Output file is empty. Deleting and reprocessing.")
                out_path.unlink()
                file_logger.info(f"Output file deleted: {out_path.name}")

        response_text = process_file(prompts, path)
        
        if not response_text:
            error_count += 1
            log_and_print(f"Error processing {filename}. Skipping file.")
            continue

        log_and_print("Saving response")
        try:
            out_path.write_text(response_text, encoding="utf-8")
            file_logger.info(f"Response saved to {out_path.name}")
        except Exception as e:
            error_count += 1
            print(f"Failed to save response for {filename}: {e}")
            file_logger.error(f"Failed to save response for {filename}: {e}")

    return error_count

def setup_file_logger():
    file_logger.setLevel(logging.INFO)
    while file_logger.handlers:
        handler = file_logger.handlers.pop()
        handler.close()
    file_handler = logging.FileHandler(Config.LOG_DIR / f"main_log.log", encoding='utf-8', mode='a')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_handler)

def log_config():
    file_logger.info(f"--- Logging started ---")
    file_logger.info(f"File logging configured to: {Config.LOG_DIR / f"main_log.log"}")
    file_logger.info("--- Config Parameters ---")
    file_logger.info(f"Model: {Config.MODEL}")
    file_logger.info(f"Result tokens: {Config.RESULT_TOKENS}")
    file_logger.info(f"Thinking tokens: {Config.THINKING_TOKENS}")
    file_logger.info(f"File mode for processing: {Config.MODE}")
    if Config.MODE == "xml":
        file_logger.info(f"XML chunk width: {Config.XML_CHUNK_WIDTH}")
        file_logger.info(f"XML chunk stride: {Config.XML_CHUNK_STRIDE}")

def load_prompt(path: Path, description: str) -> str:
    if not path.is_file():
        file_logger.warning(f"{description} prompt file not found: {path}")
        return ""
    text = path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"{description} prompt file is empty: {path}")
    file_logger.info(f"Loaded {description} prompt from: {path}")
    return text

def get_files_to_process() -> list[Path]:
    if Config.FULL_DATASET:
        file_logger.info(f"Using full dataset: {Config.DATA_DIR}")
        print("RUNNING FULL DATASET!")
        return list(Config.DATA_DIR.glob(f"*.{Config.MODE}"))
    else:
        file_logger.info(f"Using training set: {Config.TRAIN_SET_DOIS}")
        expected_files = [
            Config.DATA_DIR / f"{doi.replace('/', '_')}.{Config.MODE}"
            for doi in Config.TRAIN_SET_DOIS
        ]
        return [f for f in expected_files if f.exists()]

def initialize_logging_and_validate_files() -> list[Path]:
    setup_file_logger()
    log_config()
    files_to_process = get_files_to_process()

    if not files_to_process:
        raise FileNotFoundError(f"No files found in {Config.DATA_DIR} matching the expected DOIs.")

    file_logger.info(f"Found {len(files_to_process)} file(s) to process.")

    return files_to_process

def main():
    """Sets up, runs the file processing, and saves the results."""
    check_config()
    files_to_process = initialize_logging_and_validate_files()

    log_and_print(f"Processing {Config.MODE.upper()} files")
    error_count = process_files(files_to_process)
    
    log_and_print(f"Processing complete. Saved: {len(files_to_process)-error_count}, Errors/Skipped: {error_count}")

if __name__ == "__main__":
    main()
