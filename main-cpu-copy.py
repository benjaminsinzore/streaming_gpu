import asyncio
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import platform
import sqlite3
import time
import threading
import json
import queue
from fastapi.websockets import WebSocketState
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text
### Fix the SQLAlchemy Warning - (deprecated since: 2.0)
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional
from generator import Segment, load_csm_1b_local
from llm_interface import LLMInterface
from rag_system import RAGSystem
from vad import AudioStreamProcessor
from pydantic import BaseModel
import logging
from config import ConfigManager
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import re

##NEW APPROACH
from pathlib import Path

# --- Define CompanionConfig BEFORE using it ---
class CompanionConfig(BaseModel):
    system_prompt: str
    reference_audio_path: str
    reference_text: str
    reference_audio_path2: Optional[str] = None
    reference_text2: Optional[str] = None
    reference_audio_path3: Optional[str] = None
    reference_text3: Optional[str] = None
    model_path: str
    llm_path: str
    max_tokens: int = 8192
    voice_speaker_id: int = 0
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    embedding_model: str = "all-MiniLM-L6-v2"

# --- Load Default Configuration from JSON file ---
CONFIG_FILE_PATH = "config/app_config.json" # Adjust path if necessary
try:
    with open(CONFIG_FILE_PATH, 'r') as f:
        config_dict = json.load(f)
    # Add default values for fields not in the JSON if needed
    config_dict.setdefault('max_tokens', 8192)
    config_dict.setdefault('voice_speaker_id', 0)
    config_dict.setdefault('vad_enabled', True)
    config_dict.setdefault('vad_threshold', 0.5)
    config_dict.setdefault('embedding_model', "all-MiniLM-L6-v2")
    DEFAULT_CONFIG = CompanionConfig(**config_dict)
    print(f"Loaded default configuration from {CONFIG_FILE_PATH}")
except FileNotFoundError:
    print(f"Configuration file {CONFIG_FILE_PATH} not found. Using hardcoded defaults.")
    # Fallback to hardcoded defaults if the file is missing
    DEFAULT_CONFIG = CompanionConfig(
        system_prompt="You are a friendly AI.",
        model_path="./finetuned_model", # Update this path if needed
        llm_path="./models/llama3-8b-instruct.gguf", # Update this path if needed
        reference_audio_path="./reference.wav", # Update this path if needed
        reference_text="Hi, how can I help?",
        reference_audio_path2="", # Optional
        reference_text2="",       # Optional
        reference_audio_path3="", # Optional
        reference_text3="",       # Optional
        # Add other default values if your CompanionConfig model has them
        max_tokens=8192,
        voice_speaker_id=0,
        vad_enabled=True,
        vad_threshold=0.5,
        embedding_model="all-MiniLM-L6-v2"
    )
except json.JSONDecodeError as e:
    print(f"Error parsing {CONFIG_FILE_PATH}: {e}. Using hardcoded defaults.")
    DEFAULT_CONFIG = CompanionConfig(
        system_prompt="You are a friendly AI.",
        model_path="./finetuned_model", # Update this path if needed
        llm_path="./models/llama3-8b-instruct.gguf", # Update this path if needed
        reference_audio_path="./reference.wav", # Update this path if needed
        reference_text="Hi, how can I help?",
        reference_audio_path2="", # Optional
        reference_text2="",       # Optional
        reference_audio_path3="", # Optional
        reference_text3="",       # Optional
        # Add other default values if your CompanionConfig model has them
        max_tokens=8192,
        voice_speaker_id=0,
        vad_enabled=True,
        vad_threshold=0.5,
        embedding_model="all-MiniLM-L6-v2"
    )

# --- Rest of your constants ---
speaking_start_time = 0.0
MIN_BARGE_LATENCY = 0.9
speaker_counters = {
    0: 0,  # AI
    1: 0   # User
}
current_generation_id = 1
pending_user_inputs = []
user_input_lock = threading.Lock()
audio_fade_duration = 0.3
last_interrupt_time = 0
interrupt_cooldown = 6.0
audio_chunk_buffer = []
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
model_thread = None
model_queue = queue.Queue()
model_result_queue = queue.Queue()
model_thread_running = threading.Event()
llm_lock = threading.Lock()
audio_gen_lock = threading.Lock()

Base = declarative_base()
engine = create_engine("sqlite:///companion.db")
SessionLocal = sessionmaker(bind=engine)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    timestamp = Column(String)
    user_message = Column(Text)
    ai_message = Column(Text)
    audio_path = Column(String)

Base.metadata.create_all(bind=engine)

# ... (Rest of the code remains the same from here onwards) ...

# Define your default configuration here
# DEFAULT_CONFIG = CompanionConfig( # <-- This was moved to the top
#     system_prompt="You are a friendly AI.",
#     model_path="./finetuned_model",
#     llm_path="./models/llama3-8b-instruct.gguf",
#     reference_audio_path="./reference.wav",
#     reference_text="Hi, how can I help?",
#     reference_audio_path2="", # Optional
#     reference_text2="",       # Optional
#     reference_audio_path3="", # Optional
#     reference_text3="",       # Optional
#     # Add other default values if your CompanionConfig model has them
#     max_tokens=8192,
#     voice_speaker_id=0,
#     vad_enabled=True,
#     vad_threshold=0.5,
#     embedding_model="all-MiniLM-L6-v2"
# )

# ... (The rest of the code continues as before, ensuring CompanionConfig is defined before any function that uses it) ...

conversation_history = []
config = None
audio_queue = queue.Queue()
is_speaking = False
interrupt_flag = threading.Event()
generator = None
llm = None
rag = None
vad_processor = None
reference_segments = []
active_connections = []
message_queue = asyncio.Queue()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates") # This might still be needed if you use chat.html
config_manager = ConfigManager()

# --- Whisper Model Loading Code (as before) ---
model_id = "openai/whisper-large-v3-turbo"
cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
model_cache_dir = cache_dir / 'models--openai--whisper-large-v3-turbo' / 'snapshots'
snapshots = list(model_cache_dir.iterdir())
if not snapshots:
    raise ValueError("No model snapshots found in cache!")
snapshot_path = snapshots[0]
print(f"Using Whisper model from: {snapshot_path}")
config_path = snapshot_path / "config.json"
if not config_path.exists():
    raise FileNotFoundError(f"config.json not found at {config_path}")
model_id = str(snapshot_path)

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    local_files_only=True
)
whisper_model.to("cpu")

processor = AutoProcessor.from_pretrained(
    model_id,
    local_files_only=True
)

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float32,
    device=-1,
)
# --- End Whisper Model Loading ---

async def process_message_queue():
    while True:
        message = await message_queue.get()
        for client in active_connections[:]:
            try:
                if client.client_state == WebSocketState.CONNECTED:
                    await client.send_json(message)
            except Exception as e:
                logger.error(f"Error in message queue for client: {e}")
                if client in active_connections:
                    active_connections.remove(client)
        message_queue.task_done()

def load_reference_segments(config_data: CompanionConfig):
    global reference_segments
    reference_segments = []
    if os.path.isfile(config_data.reference_audio_path):
        logger.info(f"Loading primary reference audio: {config_data.reference_audio_path}")
        wav, sr = torchaudio.load(config_data.reference_audio_path)
        wav = torchaudio.functional.resample(wav.squeeze(0), orig_freq=sr, new_freq=24_000)
        reference_segments.append(Segment(text=config_data.reference_text, speaker=config_data.voice_speaker_id, audio=wav))
    else:
        logger.warning(f"Primary reference audio '{config_data.reference_audio_path}' not found.")
    if config_data.reference_audio_path2 and os.path.isfile(config_data.reference_audio_path2):
        logger.info(f"Loading second reference audio: {config_data.reference_audio_path2}")
        wav, sr = torchaudio.load(config_data.reference_audio_path2)
        wav = torchaudio.functional.resample(wav.squeeze(0), orig_freq=sr, new_freq=24_000)
        reference_segments.append(Segment(text=config_data.reference_text2, speaker=config_data.voice_speaker_id, audio=wav))
    if config_data.reference_audio_path3 and os.path.isfile(config_data.reference_audio_path3):
        logger.info(f"Loading third reference audio: {config_data.reference_audio_path3}")
        wav, sr = torchaudio.load(config_data.reference_audio_path3)
        wav = torchaudio.functional.resample(wav.squeeze(0), orig_freq=sr, new_freq=24_000)
        reference_segments.append(Segment(text=config_data.reference_text3, speaker=config_data.voice_speaker_id, audio=wav))
    logger.info(f"Loaded {len(reference_segments)} reference audio segments.")

def transcribe_audio(audio_data, sample_rate):
    global whisper_pipe
    audio_np = np.array(audio_data).astype(np.float32)
    if sample_rate != 16000:
        try:
            audio_tensor = torch.tensor(audio_np).unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
            audio_np = audio_tensor.squeeze(0).numpy()
        except:
            pass
    try:
        result = whisper_pipe(audio_np, generate_kwargs={"language": "english"})
        return result["text"]
    except:
        return "[Transcription error]"

def initialize_models(config_data: CompanionConfig):
    global generator, llm, rag, vad_processor, config
    config = config_data
    logger.info("Loading LLM …")
    print("Loading LLM …") # Print to terminal
    llm = LLMInterface(config_data.llm_path, config_data.max_tokens)
    logger.info("Loading RAG …")
    print("Loading RAG …") # Print to terminal
    rag = RAGSystem("companion.db", model_name=config_data.embedding_model)
    logger.info("Loading VAD model …")
    print("Loading VAD model …") # Print to terminal
    vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    vad_processor = AudioStreamProcessor(
        model=vad_model,
        utils=vad_utils,
        sample_rate=16_000,
        vad_threshold=config_data.vad_threshold,
        callbacks={"on_speech_start": on_speech_start, "on_speech_end": on_speech_end},
    )
    load_reference_segments(config_data)
    start_model_thread()
    logger.info("Warming up voice model …")
    print("Warming up voice model …") # Print to terminal
    t0 = time.time()
    model_queue.put((
        "warm-up.", config_data.voice_speaker_id, [], 500, 0.7, 40,
    ))
    while True:
        r = model_result_queue.get()
        if r is None:
            break
    logger.info(f"Voice model ready in {time.time() - t0:.1f}s")
    print(f"Voice model ready in {time.time() - t0:.1f}s") # Print to terminal
    print("--- Initialization Complete ---") # Print to terminal
    logger.info("--- Initialization Complete ---")

def on_speech_start():
    asyncio.run_coroutine_threadsafe(
        message_queue.put(
            {
                "type": "vad_status",
                "status": "speech_started",
                "should_interrupt": False,
            }
        ),
        loop,
    )

def on_speech_end(audio_data, sample_rate):
    try:
        logger.info("Transcription starting")
        user_text = transcribe_audio(audio_data, sample_rate)
        logger.info(f"Transcription completed: '{user_text}'")
        session_id = "default"
        speaker_id = 1
        index = speaker_counters[speaker_id]
        user_audio_path = f"audio/user/{session_id}_user_{index}.wav"
        os.makedirs(os.path.dirname(user_audio_path), exist_ok=True)
        audio_tensor = torch.tensor(audio_data).unsqueeze(0)
        save_audio_and_trim(user_audio_path, session_id, speaker_id, audio_tensor.squeeze(0), sample_rate)
        add_segment(user_text, speaker_id, audio_tensor.squeeze(0))
        logger.info(f"User audio saved and segment appended: {user_audio_path}")
        speaker_counters[speaker_id] += 1
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "transcription", "text": user_text}),
            loop
        )
        threading.Thread(target=lambda: process_user_input(user_text, session_id), daemon=True).start()
    except Exception as e:
        logger.error(f"VAD callback failed: {e}")

def process_pending_inputs():
    global pending_user_inputs, is_speaking, interrupt_flag
    time.sleep(0.2)
    is_speaking = False
    interrupt_flag.clear()
    with user_input_lock:
        if not pending_user_inputs:
            logger.info("No pending user inputs to process")
            return
        latest_input = pending_user_inputs[-1]
        logger.info(f"Processing only latest input: '{latest_input[0]}'")
        pending_user_inputs = []
        user_text, session_id = latest_input
        process_user_input(user_text, session_id)

def process_user_input(user_text, session_id="default"):
    global config, is_speaking, pending_user_inputs, interrupt_flag
    if not user_text or user_text.strip() == "":
        logger.warning("Empty user input received, ignoring")
        return
    interrupt_flag.clear()
    is_speaking = False
    if is_speaking:
        logger.info(f"AI is currently speaking, adding input to pending queue: '{user_text}'")
        with user_input_lock:
            pending_user_inputs = [(user_text, session_id)]
            logger.info(f"Added user input as the only pending input: '{user_text}'")
        if not interrupt_flag.is_set():
            logger.info("Automatically interrupting current speech for new input")
            interrupt_flag.set()
            asyncio.run_coroutine_threadsafe(
                message_queue.put({"type": "audio_status", "status": "interrupted"}),
                loop
            )
            time.sleep(0.3)
            process_pending_inputs()
        return
    interrupt_flag.clear()
    logger.info(f"Processing user input: '{user_text}'")
    context = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in conversation_history[-5:]])
    rag_context = rag.query(user_text)
    system_prompt = config.system_prompt
    if rag_context:
        system_prompt += f"\n\nRelevant context:\n{rag_context}"
    asyncio.run_coroutine_threadsafe(
        message_queue.put({"type": "status", "message": "Thinking..."}),
        loop
    )
    try:
        with llm_lock:
            ai_response = llm.generate_response(system_prompt, user_text, context)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        conversation_history.append({
            "timestamp": timestamp,
            "user": user_text,
            "ai": ai_response
        })
        try:
            db = SessionLocal()
            conv = Conversation(
                session_id=session_id,
                timestamp=timestamp,
                user_message=user_text,
                ai_message=ai_response,
                audio_path=""
            )
            db.add(conv)
            db.commit()
            index = speaker_counters[0]
            output_file = f"audio/ai/{session_id}_response_{index}.wav"
            speaker_counters[0] += 1
            conv.audio_path = output_file
            db.commit()
            db.close()
        except Exception as e:
            logger.error(f"Database error: {e}")
        threading.Thread(target=lambda: rag.add_conversation(user_text, ai_response), daemon=True).start()
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "audio_status", "status": "preparing"}),
            loop
        )
        time.sleep(0.2)
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "response", "text": ai_response}),
            loop
        )
        time.sleep(0.5)
        if is_speaking:
            logger.warning("Still speaking when trying to start new audio - forcing interrupt")
            interrupt_flag.set()
            is_speaking = False
            time.sleep(0.5)
        interrupt_flag.clear()
        is_speaking = False
        threading.Thread(target=lambda: audio_generation_thread(ai_response, output_file), daemon=True).start()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "error", "message": "Failed to generate response"}),
            loop
        )

def model_worker(cfg: CompanionConfig):
    global generator, model_thread_running
    logger.info("Model worker thread started")
    print("Model worker thread started") # Print to terminal
    if generator is None:
        logger.info("Loading voice model inside worker thread …")
        print("Loading voice model inside worker thread …") # Print to terminal
        generator = load_csm_1b_local(cfg.model_path, "cpu")
        logger.info("Voice model ready")
        print("Voice model ready") # Print to terminal
    while model_thread_running.is_set():
        try:
            request = model_queue.get(timeout=0.1)
            if request is None:
                break
            text, speaker_id, context, max_ms, temperature, topk = request
            for chunk in generator.generate_stream(
                    text=text,
                    speaker=speaker_id,
                    context=context,
                    max_audio_length_ms=max_ms,
                    temperature=temperature,
                    topk=topk):
                model_result_queue.put(chunk)
                if not model_thread_running.is_set():
                    break
            model_result_queue.put(None)
        except queue.Empty:
            continue
        except Exception as e:
            import traceback
            logger.error(f"Error in model worker: {e}\n{traceback.format_exc()}")
            print(f"Error in model worker: {e}\n{traceback.format_exc()}") # Print to terminal
            model_result_queue.put(Exception(f"Generation error: {e}"))
    logger.info("Model worker thread exiting")
    print("Model worker thread exiting") # Print to terminal

def start_model_thread():
    global model_thread, model_thread_running
    if model_thread is not None and model_thread.is_alive():
        return
    model_thread_running.set()
    model_thread = threading.Thread(target=model_worker, args=(config,), daemon=True, name="model_worker")
    model_thread.start()
    logger.info("Started dedicated model worker thread")

async def run_audio_generation(text, output_file):
    audio_generation_thread(text, output_file)

def send_to_all_clients(message: dict):
    for client in active_connections[:]:
        try:
            if client.client_state == WebSocketState.CONNECTED:
                asyncio.run_coroutine_threadsafe(client.send_json(message), loop)
                logger.info(f"Sent message to client: {message}")
            else:
                logger.warning("Detected non-connected client; removing from active_connections")
                active_connections.remove(client)
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            if client in active_connections:
                active_connections.remove(client)

saved_audio_paths = {
    "default": {
        0: [],
        1: []
    }
}
MAX_AUDIO_FILES = 8


def save_audio_and_trim(path, session_id, speaker_id, tensor, sample_rate):
    # Explicitly specify the format as 'wav'
    torchaudio.save(path, tensor.unsqueeze(0), sample_rate, format='wav')
    # ... rest of the function logic remains the same ...
    saved_audio_paths.setdefault(session_id, {}).setdefault(speaker_id, []).append(path)
    paths = saved_audio_paths[session_id][speaker_id]
    while len(paths) > MAX_AUDIO_FILES:
        old_path = paths.pop(0)
        if os.path.exists(old_path):
            os.remove(old_path)
            logger.info(f"Removed old audio file: {old_path}")
    other_speaker_id = 1 if speaker_id == 0 else 0
    if other_speaker_id in saved_audio_paths[session_id]:
        other_paths = saved_audio_paths[session_id][other_speaker_id]
        while len(other_paths) > MAX_AUDIO_FILES:
            old_path = other_paths.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
                logger.info(f"Removed old audio file from other speaker: {old_path}")

MAX_SEGMENTS = 8

def add_segment(text, speaker_id, audio_tensor):
    global reference_segments, generator, config
    num_reference_segments = 1
    if hasattr(config, 'reference_audio_path2') and config.reference_audio_path2:
        num_reference_segments += 1
    if hasattr(config, 'reference_audio_path3') and config.reference_audio_path3:
        num_reference_segments += 1
    new_segment = Segment(text=text, speaker=speaker_id, audio=audio_tensor)
    protected_segments = reference_segments[:num_reference_segments] if len(reference_segments) >= num_reference_segments else reference_segments.copy()
    dynamic_segments = reference_segments[num_reference_segments:] if len(reference_segments) > num_reference_segments else []
    dynamic_segments.append(new_segment)
    while len(protected_segments) + len(dynamic_segments) > MAX_SEGMENTS:
        if dynamic_segments:
            dynamic_segments.pop(0)
        else:
            break
    reference_segments = protected_segments + dynamic_segments
    if hasattr(generator, '_text_tokenizer'):
        total_tokens = 0
        for segment in reference_segments:
            tokens = generator._text_tokenizer.encode(f"[{segment.speaker}]{segment.text}")
            total_tokens += len(tokens)
            if segment.audio is not None:
                audio_frames = segment.audio.size(0) // 285
                total_tokens += audio_frames
        while dynamic_segments and total_tokens > 2048:
            removed = dynamic_segments.pop(0)
            reference_segments.remove(removed)
            removed_tokens = len(generator._text_tokenizer.encode(f"[{removed.speaker}]{removed.text}"))
            if removed.audio is not None:
                removed_audio_frames = removed.audio.size(0) // 285
                removed_tokens += removed_audio_frames
            total_tokens -= removed_tokens
        logger.info(f"Segments: {len(reference_segments)} ({len(protected_segments)} protected, {len(dynamic_segments)} dynamic), total tokens: {total_tokens}/2048")
    else:
        logger.warning("Unable to access tokenizer - falling back to word-based estimation")
        def estimate_tokens(segment):
            words = segment.text.split()
            punctuation = sum(1 for char in segment.text if char in ".,!?;:\"'()[]{}")
            text_tokens = len(words) + punctuation
            audio_tokens = 0
            if segment.audio is not None:
                audio_frames = segment.audio.size(0) // 300
                audio_tokens = audio_frames
            return text_tokens + audio_tokens
        total_estimated_tokens = sum(estimate_tokens(segment) for segment in reference_segments)
        while dynamic_segments and total_estimated_tokens > 2048:
            removed = dynamic_segments.pop(0)
            idx = reference_segments.index(removed)
            reference_segments.pop(idx)
            total_estimated_tokens -= estimate_tokens(removed)
        logger.info(f"Segments: {len(reference_segments)} ({len(protected_segments)} protected, {len(dynamic_segments)} dynamic), estimated tokens: {total_estimated_tokens}/2048")

def preprocess_text_for_tts(text):
    pattern = r'[^\w\s.,!?\']'
    cleaned_text = re.sub(pattern, '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'([.,!?])(\S)', r'\1 \2', cleaned_text)
    return cleaned_text.strip()

def audio_generation_thread(text, output_file):
    global is_speaking, interrupt_flag, audio_queue, model_thread_running, current_generation_id, speaking_start_time
    current_generation_id += 1
    this_id = current_generation_id
    interrupt_flag.clear()
    logger.info(f"Starting audio generation for ID: {this_id}")
    if not audio_gen_lock.acquire(blocking=False):
        logger.warning(f"Audio generation {this_id} - lock acquisition failed, another generation is in progress")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({
                "type": "error",
                "message": "Audio generation busy, skipping synthesis",
                "gen_id": this_id
            }),
            loop
        )
        return
    try:
        start_model_thread()
        interrupt_flag.clear()
        is_speaking = True
        speaking_start_time = time.time()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        all_audio_chunks = []
        text_lower = text.lower()
        text_lower = preprocess_text_for_tts(text_lower)
        asyncio.run_coroutine_threadsafe(
            message_queue.put({
                "type": "audio_status",
                "status": "preparing_generation",
                "gen_id": this_id
            }),
            loop
        )
        time.sleep(0.2)
        logger.info(f"Sending generating status with ID {this_id}")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({
                "type": "audio_status",
                "status": "generating",
                "gen_id": this_id
            }),
            loop
        )
        time.sleep(0.2)
        words = text.split()
        avg_wpm = 100
        words_per_second = avg_wpm / 60
        estimated_seconds = len(words) / words_per_second
        max_audio_length_ms = int(estimated_seconds * 1000)
        model_queue.put((
            text_lower,
            config.voice_speaker_id,
            reference_segments,
            max_audio_length_ms,
            0.8,
            50
        ))
        generation_start = time.time()
        chunk_counter = 0
        while True:
            try:
                if interrupt_flag.is_set():
                    logger.info(f"Audio generation {this_id} - interrupt detected, stopping")
                    model_thread_running.clear()
                    time.sleep(0.1)
                    model_thread_running.set()
                    start_model_thread()
                    while not model_result_queue.empty():
                        try:
                            model_result_queue.get_nowait()
                        except queue.Empty:
                            pass
                    break
                result = model_result_queue.get(timeout=0.1)
                if result is None:
                    logger.info(f"Audio generation {this_id} - complete")
                    break
                if isinstance(result, Exception):
                    logger.error(f"Audio generation {this_id} - error: {result}")
                    raise result
                if chunk_counter == 0:
                    first_chunk_time = time.time() - generation_start
                    logger.info(f"Audio generation {this_id} - first chunk latency: {first_chunk_time*1000:.1f}ms")
                chunk_counter += 1
                if interrupt_flag.is_set():
                    logger.info(f"Audio generation {this_id} - interrupt flag set during chunk processing")
                    break
                audio_chunk = result
                all_audio_chunks.append(audio_chunk)
                chunk_array = audio_chunk.cpu().numpy().astype(np.float32)
                audio_queue.put(chunk_array)
                if chunk_counter == 1:
                    logger.info(f"Sending first audio chunk with ID {this_id}")
                    asyncio.run_coroutine_threadsafe(
                        message_queue.put({
                            "type": "audio_status",
                            "status": "first_chunk",
                            "gen_id": this_id
                        }),
                        loop
                    )
                    time.sleep(0.1)
                asyncio.run_coroutine_threadsafe(
                    message_queue.put({
                        "type": "audio_chunk",
                        "audio": chunk_array.tolist(),
                        "sample_rate": generator.sample_rate,
                        "gen_id": this_id,
                        "chunk_num": chunk_counter
                    }),
                    loop
                )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio generation {this_id} - error processing result: {e}")
                break
        if all_audio_chunks and not interrupt_flag.is_set():
            try:
                complete_audio = torch.cat(all_audio_chunks)
                save_audio_and_trim(output_file, "default", config.voice_speaker_id, complete_audio, generator.sample_rate)
                add_segment(text.lower(), config.voice_speaker_id, complete_audio)
                total_time = time.time() - generation_start
                total_audio_seconds = complete_audio.size(0) / generator.sample_rate
                rtf = total_time / total_audio_seconds
                logger.info(f"Audio generation {this_id} - completed in {total_time:.2f}s, RTF: {rtf:.2f}x")
            except Exception as e:
                logger.error(f"Audio generation {this_id} - error saving complete audio: {e}")
    except Exception as e:
        import traceback
        logger.error(f"Audio generation {this_id} - unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        is_speaking = False
        audio_queue.put(None)
        try:
            logger.info(f"Audio generation {this_id} - sending completion status")
            asyncio.run_coroutine_threadsafe(
                message_queue.put({
                    "type": "audio_status",
                    "status": "complete",
                    "gen_id": this_id
                }),
                loop
            )
        except Exception as e:
            logger.error(f"Audio generation {this_id} - failed to send completion status: {e}")
        with user_input_lock:
            if pending_user_inputs:
                logger.info(f"Audio generation {this_id} - processing pending inputs")
                process_pending_inputs()
        logger.info(f"Audio generation {this_id} - releasing lock")
        audio_gen_lock.release()

def handle_interrupt(websocket):
    global is_speaking, last_interrupt_time, interrupt_flag, model_thread_running, speaking_start_time
    logger.info(f"Interrupt requested. Current state: is_speaking={is_speaking}")
    current_time = time.time()
    time_since_speech_start = current_time - speaking_start_time if speaking_start_time > 0 else 999
    time_since_last_interrupt = current_time - last_interrupt_time
    if time_since_last_interrupt < interrupt_cooldown and time_since_speech_start > 3.0:
        logger.info(f"Ignoring interrupt: too soon after previous interrupt ({time_since_last_interrupt:.1f}s < {interrupt_cooldown}s)")
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({
                "type": "audio_status",
                "status": "interrupt_acknowledged",
                "success": False,
                "reason": "cooldown"
            }),
            loop
        )
        return False
    last_interrupt_time = current_time
    if is_speaking or not model_result_queue.empty():
        logger.info("Interruption processing: we are speaking or generating")
        interrupt_flag.set()
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "audio_status", "status": "interrupted"}),
            loop
        )
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({
                "type": "audio_status",
                "status": "interrupt_acknowledged"
            }),
            loop
        )
        try:
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break
            audio_queue.put(None)
            logger.info("Audio queue cleared")
        except Exception as e:
            logger.error(f"Error clearing audio queue: {e}")
        if vad_processor:
            try:
                vad_processor.reset()
                logger.info("VAD processor reset")
            except Exception as e:
                logger.error(f"Error resetting VAD: {e}")
        if model_thread and model_thread.is_alive():
            try:
                model_thread_running.clear()
                time.sleep(0.1)
                model_thread_running.set()
                start_model_thread()
                logger.info("Model thread restarted")
            except Exception as e:
                logger.error(f"Error restarting model thread: {e}")
        return True
    logger.info("No active speech to interrupt")
    return False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global is_speaking, audio_queue
    await websocket.accept()
    active_connections.append(websocket)
    # Removed loading saved config request
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "config":
                # Optional: Allow dynamic config via WebSocket if needed, but not for startup init
                # For startup, we use the DEFAULT_CONFIG
                try:
                    config_data = data["config"]
                    conf = CompanionConfig(**config_data)
                    # Optional: Save config if received via WebSocket
                    # saved = config_manager.save_config(config_data)
                    initialize_models(conf)
                    await websocket.send_json({"type": "status", "message": "Models initialized and configuration saved"})
                except Exception as e:
                    logger.error(f"Error processing config: {str(e)}")
                    await websocket.send_json({"type": "error", "message": f"Configuration error: {str(e)}"})
            elif data["type"] == "request_saved_config":
                # Optional: Send current config if requested
                if config:
                    ##THIS IS DEPRECATED AND WILL NOT BE INCLUDED IN THE UPCOMING Pydantic V3 REPLACE '.dict()' with 'model_dump()'
                    await websocket.send_json({"type": "saved_config", "config": config.model_dump()})
                else:
                    await websocket.send_json({"type": "saved_config", "config": None})
            elif data["type"] == "text_message":
                user_text = data["text"]
                session_id = data.get("session_id", "default")
                logger.info(f"TEXT-MSG from client: {user_text!r}")
                if is_speaking:
                    with user_input_lock:
                        if len(pending_user_inputs) >= 3:
                            pending_user_inputs = pending_user_inputs[-2:]
                        pending_user_inputs.append((user_text, session_id))
                    await websocket.send_json(
                        {"type": "status", "message": "Queued – I'll answer in a moment"})
                    continue
                await message_queue.put({"type": "transcription", "text": user_text})
                threading.Thread(
                    target=lambda: process_user_input(user_text, session_id),
                    daemon=True).start()
            elif data["type"] == "audio":
                audio_data = np.asarray(data["audio"], dtype=np.float32)
                sample_rate = data["sample_rate"]
                if sample_rate != 16000:
                    audio_tensor = torch.tensor(audio_data).unsqueeze(0)
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, orig_freq=sample_rate, new_freq=16000
                    )
                    audio_data = audio_tensor.squeeze(0).numpy()
                    sample_rate = 16000
                if config and config.vad_enabled:
                    vad_processor.process_audio(audio_data)
                else:
                    text = transcribe_audio(audio_data, sample_rate)
                    await websocket.send_json({"type": "transcription", "text": text})
                    await message_queue.put({"type": "transcription", "text": text})
                    if is_speaking:
                        with user_input_lock:
                            pending_user_inputs.append((text, "default"))
                    else:
                        process_user_input(text)
            elif data["type"] == "interrupt":
                logger.info("Explicit interrupt request received")
                await websocket.send_json({
                    "type": "audio_status",
                    "status": "interrupt_acknowledged"
                })
                success = handle_interrupt(websocket)
                if success:
                    await asyncio.sleep(0.3)
                    with user_input_lock:
                        if pending_user_inputs:
                            user_text, session_id = pending_user_inputs.pop(0)
                            pending_user_inputs.clear()
                            threading.Thread(
                                target=lambda: process_user_input(user_text, session_id),
                                daemon=True
                            ).start()
                await websocket.send_json({
                    "type": "audio_status",
                    "status": "interrupted",
                    "success": success
                })
            elif data["type"] == "mute":
                await websocket.send_json({"type": "mute_status", "muted": data["muted"]})
                if not data["muted"] and config and config.vad_enabled:
                    vad_processor.reset()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

# Removed the /setup route
# @app.get("/setup", response_class=HTMLResponse)
# async def setup_page(request: Request):
#     return templates.TemplateResponse("setup.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Changed redirect from /setup to /chat
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    os.makedirs("static", exist_ok=True)
    os.makedirs("audio/user", exist_ok=True)
    os.makedirs("audio/ai", exist_ok=True)
    os.makedirs("embeddings_cache", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    # Changed index.html to redirect to /chat instead of /setup
    with open("templates/index.html", "w") as f:
        f.write("""<meta http-equiv="refresh" content="0; url=/chat" />""")
    try:
        torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    except:
        pass
    asyncio.create_task(process_message_queue())

    # --- NEW: Automatic Initialization ---
    print("--- Starting Automatic Initialization ---")
    logger.info("--- Starting Automatic Initialization ---")
    # Run the initialization in a separate thread so the server startup doesn't block
    init_thread = threading.Thread(target=initialize_models, args=(DEFAULT_CONFIG,), daemon=True)
    init_thread.start()
    # --- END NEW ---

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server shutting down...")

@app.get("/api/conversations")
async def get_conversations(request: Request):
    conn = sqlite3.connect("companion.db")
    cur = conn.cursor()
    cur.execute("SELECT id, user_message, ai_message FROM conversations ORDER BY id DESC")
    data = [{"id": row[0], "user_message": row[1], "ai_message": row[2]} for row in cur.fetchall()]
    conn.close()
    return JSONResponse(content=data)

@app.put("/api/conversations/{conv_id}")
async def update_conversation(conv_id: int, data: dict):
    try:
        conn = sqlite3.connect("companion.db")
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET user_message=?, ai_message=? WHERE id=?",
                    (data["user_message"], data["ai_message"], conv_id))
        conn.commit()
        conn.close()
        return JSONResponse(content={"status": "updated", "id": conv_id})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.delete("/api/conversations")
async def delete_all_conversations():
    try:
        conn = sqlite3.connect("companion.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()
        return JSONResponse(content={"status": "all deleted"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: int):
    try:
        conn = sqlite3.connect("companion.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        conn.close()
        return JSONResponse(content={"status": "deleted", "id": conv_id})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/crud", response_class=HTMLResponse)
async def crud_ui(request: Request):
    return templates.TemplateResponse("crud.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=lambda: asyncio.run(loop.run_forever()), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)