import asyncio
import os

# 1. Set compile-disable environment variables FIRST
os.environ['NO_TORCH_COMPILE'] = '1'
os.environ['TORCH_COMPILE'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['ONEDNN_PRIMITIVE_CACHE_CAPACITY'] = '0'

# Performance thread settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 2. Now import PyTorch
import torch
import torchaudio

# 3. Configure PyTorch's dynamo AFTER importing torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()



import platform
import sqlite3
import time
import threading
import json
import queue
from fastapi.websockets import WebSocketState
import sounddevice as sd
import numpy as np
import whisper

import bcrypt

import uuid # Ensure this import is present at the top if not already there
from fastapi import Depends, Request
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Optional
from generator import Segment, load_csm_1b_local
from llm_interface import LLMInterface
from rag_system import RAGSystem
from vad import AudioStreamProcessor
from pydantic import BaseModel
import logging
import bcrypt
from config import ConfigManager
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import re
from passlib.context import CryptContext
from jose import JWTError, jwt
from pathlib import Path


from fastapi.responses import RedirectResponse
from fastapi import status

from fastapi import Response

from fastapi import Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt


from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Password hashing

from sqlalchemy import Column, Integer, String, Text
from datetime import datetime, timedelta
import time



##NEW APPROACH
from pathlib import Path

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

# Initialize Base after importing declarative_base
Base = declarative_base()
engine = create_engine("sqlite:///companion.db")
SessionLocal = sessionmaker(bind=engine)

# Update your User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    session_token = Column(String, unique=True, index=True)
    expires_at = Column(String)
    created_at = Column(String, default=lambda: datetime.now().isoformat())



# Inside your Python code, update the Conversation model
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # Add this line - links to users.id
    session_id = Column(String, index=True)
    timestamp = Column(String)
    user_message = Column(Text)
    ai_message = Column(Text)
    audio_path = Column(String)
    # Add this line for the starred status
    starred = Column(Integer, default=0) # Use Integer (0 or 1) for boolean-like behavior in SQLite



# Add the Feedback model
class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # Link to the user who submitted feedback
    email = Column(String, index=True)     # Store the email (redundant but useful for reporting/lookup)
    feedback_type = Column(String)         # e.g., bug, feature, comment
    message = Column(Text)                 # The feedback content
    timestamp = Column(String, default=lambda: datetime.now().isoformat()) # When submitted
    # Add other fields as needed, e.g., rating, subject



# Create all tables
Base.metadata.create_all(bind=engine)




class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

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

# Global variables for models
whisper_model = None
processor = None
whisper_pipe = None
generator = None
llm = None
rag = None
vad_processor = None
config = None
models_loaded = False  # Flag to track if models are loaded

conversation_history = []
audio_queue = queue.Queue()
is_speaking = False
interrupt_flag = threading.Event()
reference_segments = []
active_connections = []
message_queue = asyncio.Queue()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
config_manager = ConfigManager()



# Session management functions
def create_session_token():
    return str(uuid.uuid4())



def create_user_session(db, user_id: int):
    session_token = create_session_token()
    expires_at = (datetime.now() + timedelta(days=7)).isoformat()  # 7-day session
    
    session = UserSession(
        user_id=user_id,
        session_token=session_token,
        expires_at=expires_at
    )
    db.add(session)
    db.commit()
    return session_token


def get_user_from_session(db, session_token: str):
    if not session_token:
        return None
    
    # Clean up expired sessions first
    db.query(UserSession).filter(
        UserSession.expires_at <= datetime.now().isoformat()
    ).delete()
    db.commit()
    
    session = db.query(UserSession).filter(
        UserSession.session_token == session_token,
        UserSession.expires_at > datetime.now().isoformat()
    ).first()
    
    if session:
        user = db.query(User).filter(User.id == session.user_id).first()
        return user
    return None



async def get_current_user(request: Request):
    # Get session token from cookie
    session_token = request.cookies.get("session_token")
    
    if not session_token:
        return None
    
    db = SessionLocal()
    try:
        user = get_user_from_session(db, session_token)
        if user:
            # Return the ORM object but extract values when needed
            return user
        return None
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return None
    finally:
        db.close()



def get_password_hash(password: str) -> str:
    # Using bcrypt for more secure password hashing
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password=pwd_bytes, salt=salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode('utf-8')
    hashed_password_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password=password_bytes, hashed_password=hashed_password_bytes)


def create_user(db, email: str, password: str):
    hashed_password = get_password_hash(password)
    user = User(email=email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db, email: str):
    return db.query(User).filter(User.email == email).first()




def authenticate_user(db, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        # To prevent timing attacks, we should still call verify_password even if user doesn't exist
        verify_password("dummy_password", "dummy_hash") # Simulate verification time
        return None
    
    # Truncate the provided password before verifying
    truncated_password = password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
    if not verify_password(truncated_password, user.hashed_password):
        return None
    
    # Refresh the user to ensure it's attached to the session
    db.refresh(user)
    return user




class SessionManager:
    def __init__(self):
        self.sessions = {}  # {session_id: {'connections': [], 'user_data': {}}}
        self.connection_to_session = {}  # {websocket: session_id}

    def create_session(self, session_id: str = None, user_data: dict = None):
        """Create a new session"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'connections': [], 
                'user_data': user_data or {},
                'created_at': time.time()
            }
        return session_id

    def add_connection(self, session_id: str, websocket, user_data: dict = None):
        """Add a WebSocket connection to a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'connections': [], 
                'user_data': user_data or {},
                'created_at': time.time()
            }
        
        self.sessions[session_id]['connections'].append(websocket)
        self.connection_to_session[websocket] = session_id
        
        # Update user data if provided
        if user_data:
            self.sessions[session_id]['user_data'].update(user_data)

    def remove_connection(self, websocket):
        """Remove a WebSocket connection"""
        session_id = self.connection_to_session.get(websocket)
        if session_id and session_id in self.sessions:
            if websocket in self.sessions[session_id]['connections']:
                self.sessions[session_id]['connections'].remove(websocket)
            
            # Remove session if no connections left
            if not self.sessions[session_id]['connections']:
                del self.sessions[session_id]
        
        # Remove from connection mapping
        if websocket in self.connection_to_session:
            del self.connection_to_session[websocket]

    def get_session_connections(self, session_id: str):
        """Get all connections for a session"""
        return self.sessions.get(session_id, {}).get('connections', [])

    def get_connection_session(self, websocket):
        """Get session ID for a connection"""
        return self.connection_to_session.get(websocket)

    def get_session_data(self, session_id: str):
        """Get user data for a session"""
        return self.sessions.get(session_id, {}).get('user_data', {})

    def update_session_data(self, session_id: str, user_data: dict):
        """Update user data for a session"""
        if session_id in self.sessions:
            self.sessions[session_id]['user_data'].update(user_data)

    def cleanup_expired_sessions(self, max_age_seconds: int = 3600):
        """Clean up sessions older than max_age_seconds"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            if current_time - session_data['created_at'] > max_age_seconds:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            # Close all connections in expired session
            for websocket in self.sessions[session_id]['connections']:
                try:
                    asyncio.create_task(websocket.close())
                except:
                    pass
                if websocket in self.connection_to_session:
                    del self.connection_to_session[websocket]
            del self.sessions[session_id]

session_manager = SessionManager()


session_manager = SessionManager()

def load_whisper_model():
    """Load Whisper model for speech recognition"""
    global whisper_model, processor, whisper_pipe
    
    logger.info("Loading Whisper model...")
    
    # Find the exact snapshot path
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    model_cache_dir = cache_dir / 'models--openai--whisper-large-v3-turbo' / 'snapshots'

    # Get all snapshots and use the first one
    snapshots = list(model_cache_dir.iterdir())
    if not snapshots:
        raise ValueError("No model snapshots found in cache!")

    # Use the first snapshot (usually the only one or most recent)
    snapshot_path = snapshots[0]
    logger.info(f"Using Whisper model from: {snapshot_path}")

    # Verify config.json exists
    config_path = snapshot_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")

    # Load using the direct local path
    model_id = str(snapshot_path)

    devIce = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch.float16 if devIce == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True
    )
    whisper_model.to(devIce)

    

    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=True
    )

    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1  # 0 for first GPU, -1 for CPU
    dtype = torch.float16 if device != -1 else torch.float32  # Use float16 for GPU

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,  # Use GPU if available
    )
    
    logger.info("Whisper model loaded successfully")

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
    global generator, llm, rag, vad_processor, config, models_loaded
    config = config_data

    # --- Log model paths early ---
    logger.info(f"LLM model path: {os.path.abspath(config_data.llm_path)}")
    logger.info(f"Embedding model for RAG: {config_data.embedding_model}")
    logger.info(f"Voice model speaker ID: {config_data.voice_speaker_id}")
    
    # Add voice model path if it's in config (adjust field name as needed)
    if hasattr(config_data, 'tts_model_path'):
        logger.info(f"Voice/TTS model path: {os.path.abspath(config_data.tts_model_path)}")
    # ---

    logger.info("Loading LLM...")
    llm = LLMInterface(config_data.llm_path, config_data.max_tokens)

    logger.info("Loading RAG...")
    rag = RAGSystem("companion.db", model_name=config_data.embedding_model)

    logger.info("Loading VAD model...")
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
    
    logger.info("Warming up voice model...")
    t0 = time.time()
    model_queue.put((
        "warm-up.", config_data.voice_speaker_id, [], 500, 0.7, 40,
    ))
    
    try:
        r = model_result_queue.get(timeout=90)  # Prevent infinite hang
        if r is None:
            logger.error("Warm-up returned None")
    except queue.Empty:
        logger.error("Voice model warm-up timed out after 90s!")
        raise RuntimeError("Voice model failed to respond during warm-up")
        
    logger.info(f"Voice model ready in {time.time() - t0:.1f}s")
    
    models_loaded = True
    logger.info("All models initialized successfully")


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

# ... (rest of your functions remain the same - process_pending_inputs, process_user_input, model_worker, etc.)

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
    
    logger.info(f"process_user_input called with: '{user_text}', session: {session_id}")
    
    if not user_text or user_text.strip() == "":
        logger.warning("Empty user input received, ignoring")
        return
        
    logger.info(f"Current state - is_speaking: {is_speaking}, pending_inputs: {len(pending_user_inputs)}")
    
    interrupt_flag.clear()
    is_speaking = False
    
    if is_speaking:
        logger.info(f"AI is currently speaking, adding input to pending queue: '{user_text}'")
        with user_input_lock:
            pending_user_inputs = [(user_text, session_id)]
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
    
    # Get conversation context
    context = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in conversation_history[-5:]])
    rag_context = rag.query(user_text) if rag else ""
    
    system_prompt = config.system_prompt if config else "You are a helpful AI assistant."
    if rag_context:
        system_prompt += f"\n\nRelevant context:\n{rag_context}"
        
    logger.info(f"System prompt prepared, context length: {len(context)}")
    
    # Send thinking status
    asyncio.run_coroutine_threadsafe(
        message_queue.put({"type": "status", "message": "Thinking..."}),
        loop
    )
    
    try:
        logger.info("Generating AI response...")
        with llm_lock:
            ai_response = llm.generate_response(system_prompt, user_text, context)
        logger.info(f"AI response generated: '{ai_response[:100]}...'")
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        conversation_history.append({
            "timestamp": timestamp,
            "user": user_text,
            "ai": ai_response
        })
        
        # Save to database
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
            logger.info(f"Conversation saved to database, audio path: {output_file}")
        except Exception as e:
            logger.error(f"Database error: {e}")
            
        # Add to RAG system
        if rag:
            threading.Thread(target=lambda: rag.add_conversation(user_text, ai_response), daemon=True).start()
            
        # Send response to client
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "audio_status", "status": "preparing"}),
            loop
        )
        
        time.sleep(0.2)
        
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "response", "text": ai_response}),
            loop
        )
        
        logger.info("Response sent to client, starting audio generation...")
        
        time.sleep(0.5)
        
        if is_speaking:
            logger.warning("Still speaking when trying to start new audio - forcing interrupt")
            interrupt_flag.set()
            is_speaking = False
            time.sleep(0.5)
            
        interrupt_flag.clear()
        is_speaking = False
        
        # Start audio generation
        threading.Thread(
            target=lambda: audio_generation_thread(ai_response, output_file), 
            daemon=True
        ).start()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "error", "message": "Failed to generate response"}),
            loop
        )



def model_worker(cfg: CompanionConfig):
    global generator, model_thread_running
    logger.info("Hello! Model worker thread started")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        if generator is None:
            logger.info(f"About to load voice model from: {os.path.abspath(cfg.model_path)}")
            # Verify path exists
            if not os.path.exists(cfg.model_path):
                raise FileNotFoundError(f"Model path not found: {cfg.model_path}")
            generator = load_csm_1b_local(cfg.model_path, device)

            # Apply patch to bypass the problematic compiled method
            if hasattr(generator._audio_tokenizer, '_to_encoder_framerate'):
                original_method = generator._audio_tokenizer._to_encoder_framerate
                
                def patched_to_framerate(emb):
                    # Force eager execution without compilation
                    with torch.inference_mode(), torch.autocast(device_type='cpu', enabled=False):
                        return original_method(emb)
                
                generator._audio_tokenizer._to_encoder_framerate = patched_to_framerate
                print("Patched _to_encoder_framerate to prevent compilation errors")


            logger.info(f"Voice model successfully loaded on {device}")
        else:
            logger.info("Voice model already loaded")

        while model_thread_running.is_set():
            try:
                request = model_queue.get(timeout=0.1)
                if request is None:
                    break
                # ... rest of generation logic ...
                model_result_queue.put(None)
            except queue.Empty:
                continue
            except Exception as e:
                import traceback
                logger.error(f"Error during generation: {e}\n{traceback.format_exc()}")
                model_result_queue.put(Exception(str(e)))

    except Exception as e:
        import traceback
        logger.critical(f"CRITICAL: model_worker failed during model loading: {e}\n{traceback.format_exc()}")
        # Signal failure to main thread
        try:
            model_result_queue.put(None)  # or put an exception
        except:
            pass

def start_model_thread():
    global model_thread, model_thread_running
    if model_thread is not None and model_thread.is_alive():
        return
    model_thread_running.set()
    model_thread = threading.Thread(target=model_worker, args=(config,), daemon=True, name="model_worker")
    model_thread.start()
    logger.info("Started dedicated model worker thread")

# ... (rest of your functions remain the same - send_to_all_clients, save_audio_and_trim, add_segment, etc.)

saved_audio_paths = {
    "default": {
        0: [],
        1: []
    }
}
MAX_AUDIO_FILES = 8

def save_audio_and_trim(path, session_id, speaker_id, tensor, sample_rate):
    torchaudio.save(path, tensor.unsqueeze(0), sample_rate)
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
    global is_speaking, interrupt_flag, audio_queue, model_thread_running, current_generation_id, speaking_start_time, generator
    current_generation_id += 1
    this_id = current_generation_id
    interrupt_flag.clear()
    logger.info(f"Starting audio generation for ID: {this_id}")
    
    # Check if generator is on GPU
    device = "cuda" if hasattr(generator, 'device') and generator.device.type == 'cuda' else "cpu"
    logger.info(f"Generator device: {device}")
    
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
                
                # IMPORTANT: Move to CPU if on GPU, then convert to numpy
                if audio_chunk.is_cuda:
                    audio_chunk = audio_chunk.cpu()  # Move from GPU to CPU
                
                all_audio_chunks.append(audio_chunk)
                
                # Convert to numpy
                chunk_array = audio_chunk.numpy().astype(np.float32)  # Use .numpy() instead of .cpu().numpy()
                
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
                import traceback
                logger.error(traceback.format_exc())
                break
        
        if all_audio_chunks and not interrupt_flag.is_set():
            try:
                # Ensure all chunks are on CPU before concatenation
                cpu_chunks = []
                for chunk in all_audio_chunks:
                    if chunk.is_cuda:
                        cpu_chunks.append(chunk.cpu())
                    else:
                        cpu_chunks.append(chunk)
                
                complete_audio = torch.cat(cpu_chunks)
                
                # Save audio
                save_audio_and_trim(output_file, "default", config.voice_speaker_id, complete_audio, generator.sample_rate)
                add_segment(text.lower(), config.voice_speaker_id, complete_audio)
                
                total_time = time.time() - generation_start
                total_audio_seconds = complete_audio.size(0) / generator.sample_rate
                rtf = total_time / total_audio_seconds
                logger.info(f"Audio generation {this_id} - completed in {total_time:.2f}s, RTF: {rtf:.2f}x")
                
                # Optional: Log GPU memory usage
                if device == "cuda":
                    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
            except Exception as e:
                logger.error(f"Audio generation {this_id} - error saving complete audio: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    except Exception as e:
        import traceback
        logger.error(f"Audio generation {this_id} - unexpected error: {e}\n{traceback.format_exc()}")
    
    finally:
        is_speaking = False
        audio_queue.put(None)
        
        # Clean up GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
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
    global is_speaking, last_interrupt_time, interrupt_flag, model_thread_running, speaking_start_time, vad_processor
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
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())
    
    # Get user from session
    user_id = None
    user_email = "anonymous"
    try:
        # Try to get session token from query params or cookies
        session_token = websocket.query_params.get("session_token")
        if not session_token and "cookie" in websocket.headers:
            cookies = websocket.headers["cookie"]
            for cookie in cookies.split(";"):
                if "session_token" in cookie:
                    session_token = cookie.split("=")[1].strip()
                    break
        
        if session_token:
            db = SessionLocal()
            user = get_user_from_session(db, session_token)
            if user:
                user_id = user.id
                user_email = user.email
                logger.info(f"Authenticated user: {user_email} (ID: {user_id})")
            db.close()
    except Exception as e:
        logger.error(f"Error getting user from session: {e}")
    
    session_manager.add_connection(session_id, websocket, {
        "connected_at": time.time(),
        "ip_address": getattr(websocket.client, 'host', 'unknown'),
        "user_id": user_id,
        "user_email": user_email
    })
    logger.info(f"New WebSocket connection for session: {session_id}, user: {user_email} (ID: {user_id})")

    try:
        # Send immediate welcome message
        await websocket.send_json({
            "type": "test_response",
            "message": f"WebSocket connected successfully! User: {user_email}",
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"Sent welcome message to session: {session_id}")

    except Exception as e:
        logger.error(f"Error sending initial messages to session {session_id}: {e}")

    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received WebSocket message from session {session_id}, user {user_email}: {data.get('type')}")
            
            message_type = data.get("type")
            
            if message_type == "text_message":
                user_text = data.get("text", "").strip()
                if user_text:
                    logger.info(f"Processing text input from user {user_email}: '{user_text}'")
                    
                    # Send immediate acknowledgment
                    await websocket.send_json({
                        "type": "status",
                        "message": "Processing your message...",
                        "session_id": session_id,
                        "user_id": user_id
                    })
                    
                    # Process in thread with user context
                    threading.Thread(
                        target=process_user_input_direct,
                        args=(user_text, session_id, websocket, user_id),
                        daemon=True
                    ).start()
                else:
                    logger.warning(f"Received empty text input from user {user_email}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Empty message received",
                        "session_id": session_id
                    })
                    
            elif message_type == "interrupt":
                logger.info(f"Interrupt request received from user {user_email}")
                try:
                    success = handle_interrupt(websocket)
                    if success:
                        await websocket.send_json({
                            "type": "audio_status",
                            "status": "interrupt_acknowledged",
                            "session_id": session_id,
                            "user_id": user_id
                        })
                except Exception as e:
                    logger.error(f"Error handling interrupt for user {user_email}: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to process interrupt",
                        "session_id": session_id
                    })
                
            elif message_type == "test":
                test_message = data.get("message", "No message")
                logger.info(f"Test message received from user {user_email}: {test_message}")
                await websocket.send_json({
                    "type": "test_response", 
                    "message": f"Test successful - Backend received: {test_message} (User: {user_email})",
                    "session_id": session_id,
                    "user_id": user_id
                })
                
            elif message_type == "request_conversation_history":
                logger.info(f"Conversation history request from user {user_email}")
                if user_id:
                    try:
                        db = SessionLocal()
                        conversations = db.query(Conversation).filter(
                            Conversation.user_id == user_id
                        ).order_by(Conversation.timestamp.desc()).limit(50).all()
                        db.close()
                        
                        await websocket.send_json({
                            "type": "conversation_history",
                            "conversations": [{
                                "id": conv.id,
                                "timestamp": conv.timestamp,
                                "user_message": conv.user_message,
                                "ai_message": conv.ai_message,
                                "audio_path": conv.audio_path
                            } for conv in conversations],
                            "session_id": session_id,
                            "user_id": user_id
                        })
                    except Exception as e:
                        logger.error(f"Error fetching conversation history for user {user_email}: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to fetch conversation history",
                            "session_id": session_id
                        })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Authentication required to access conversation history",
                        "session_id": session_id
                    })
                
            else:
                logger.warning(f"Unknown message type from user {user_email}: {message_type}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "session_id": session_id
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}, user: {user_email}")
        session_manager.remove_connection(websocket)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for session {session_id}, user {user_email}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid JSON data received",
                "session_id": session_id
            })
        except:
            pass
        session_manager.remove_connection(websocket)
    except Exception as e:
        logger.error(f"Error in WebSocket connection for session {session_id}, user {user_email}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Server error occurred",
                "session_id": session_id
            })
        except:
            pass
        session_manager.remove_connection(websocket)







def process_user_input_direct(user_text: str, session_id: str, websocket: WebSocket, user_id: int = None):
    """
    Process user input and send responses directly to the WebSocket with user context
    """
    try:
        logger.info(f"process_user_input_direct called with: '{user_text}', session: {session_id}, user_id: {user_id}")
        
        # Get conversation context
        context = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in conversation_history[-5:]])
        rag_context = rag.query(user_text) if rag else ""
        
        system_prompt = config.system_prompt if config else "You are a helpful AI assistant."
        if rag_context:
            system_prompt += f"\n\nRelevant context:\n{rag_context}"
            
        logger.info("Generating AI response...")
        with llm_lock:
            ai_response = llm.generate_response(system_prompt, user_text, context)
        logger.info(f"AI response generated: '{ai_response}'")
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        conversation_history.append({
            "timestamp": timestamp,
            "user": user_text,
            "ai": ai_response
        })
        
        # Save to database WITH USER ID
        try:
            db = SessionLocal()
            conv = Conversation(
                user_id=user_id,  # This can be None for anonymous users
                session_id=session_id,
                timestamp=timestamp,
                user_message=user_text,
                ai_message=ai_response,
                audio_path=""
            )
            db.add(conv)
            db.commit()
            
            index = speaker_counters[0]
            # Include user_id in filename for better organization
            if user_id:
                output_file = f"audio/ai/user_{user_id}/{session_id}_response_{index}.wav"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            else:
                output_file = f"audio/ai/{session_id}_response_{index}.wav"
                
            speaker_counters[0] += 1
            
            conv.audio_path = output_file
            db.commit()
            db.close()
            
            if user_id:
                logger.info(f"Conversation saved to database for user {user_id}, audio path: {output_file}")
            else:
                logger.info(f"Conversation saved to database (anonymous user), audio path: {output_file}")
                
        except Exception as e:
            logger.error(f"Database error: {e}")
            
        # Send the AI response immediately to the WebSocket
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({
                "type": "response",
                "text": ai_response,
                "session_id": session_id,
                "user_id": user_id
            }),
            loop
        )
        
        logger.info("AI response sent via WebSocket")
        
        # Start audio generation
        threading.Thread(
            target=audio_generation_thread_direct,
            args=(ai_response, output_file, session_id, websocket, user_id),
            daemon=True
        ).start()
        
    except Exception as e:
        logger.error(f"Error in process_user_input_direct: {e}")
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({
                "type": "error",
                "message": f"Failed to generate response: {str(e)}",
                "session_id": session_id,
                "user_id": user_id
            }),
            loop
        )











def audio_generation_thread_direct(text, output_file, session_id, websocket, user_id=None):
    """
    Generate audio and send chunks directly to WebSocket with user context
    """
    global current_generation_id
    current_generation_id += 1
    this_id = current_generation_id
    
    logger.info(f"Starting audio generation for ID: {this_id}, session: {session_id}, user_id: {user_id}")
    
    try:
        # Send generating status
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({
                "type": "audio_status",
                "status": "generating",
                "gen_id": this_id,
                "session_id": session_id,
                "user_id": user_id
            }),
            loop
        )
        
        text_lower = text.lower()
        text_lower = preprocess_text_for_tts(text_lower)
        
        words = text.split()
        avg_wpm = 100
        words_per_second = avg_wpm / 60
        estimated_seconds = len(words) / words_per_second
        max_audio_length_ms = int(estimated_seconds * 1000)
        
        # Send to model queue
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
                result = model_result_queue.get(timeout=1.0)
                if result is None:
                    logger.info(f"Audio generation {this_id} - complete")
                    break
                if isinstance(result, Exception):
                    logger.error(f"Audio generation {this_id} - error: {result}")
                    raise result
                
                if chunk_counter == 0:
                    first_chunk_time = time.time() - generation_start
                    logger.info(f"Audio generation {this_id} - first chunk latency: {first_chunk_time*1000:.1f}ms")
                    
                    # Send first chunk status
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({
                            "type": "audio_status",
                            "status": "first_chunk",
                            "gen_id": this_id,
                            "session_id": session_id,
                            "user_id": user_id
                        }),
                        loop
                    )
                
                chunk_counter += 1
                audio_chunk = result
                chunk_array = audio_chunk.cpu().numpy().astype(np.float32)
                
                # Send audio chunk directly to WebSocket
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({
                        "type": "audio_chunk",
                        "audio": chunk_array.tolist(),
                        "sample_rate": generator.sample_rate,
                        "gen_id": this_id,
                        "chunk_num": chunk_counter,
                        "session_id": session_id,
                        "user_id": user_id
                    }),
                    loop
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio generation {this_id} - error: {e}")
                break
        
        # Send completion status
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({
                "type": "audio_status",
                "status": "complete",
                "gen_id": this_id,
                "session_id": session_id,
                "user_id": user_id
            }),
            loop
        )
        
        logger.info(f"Audio generation {this_id} completed successfully for user_id: {user_id}")
        
    except Exception as e:
        logger.error(f"Error in audio_generation_thread_direct: {e}")
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({
                "type": "error",
                "message": f"Audio generation failed: {str(e)}",
                "session_id": session_id,
                "user_id": user_id
            }),
            loop
        )







def process_user_input_wrapper(user_text: str, session_id: str, websocket: WebSocket):
    """
    Wrapper function to process user input with proper error handling
    """
    try:
        process_user_input(user_text, session_id)
    except Exception as e:
        logger.error(f"Error in process_user_input for session {session_id}: {e}")
        # Try to send error back to client
        try:
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "type": "error",
                    "message": f"Failed to generate response: {str(e)}",
                    "session_id": session_id
                }),
                loop
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message to session {session_id}: {send_error}")


# Make sure your process_user_input function is updated to use the message_queue properly:
def process_user_input(user_text, session_id="default", user_id=None):
    global config, is_speaking, pending_user_inputs, interrupt_flag
    
    logger.info(f"process_user_input called with: '{user_text}', session: {session_id}, user_id: {user_id}")
    
    if not user_text or user_text.strip() == "":
        logger.warning("Empty user input received, ignoring")
        return
        
    logger.info(f"Current state - is_speaking: {is_speaking}, pending_inputs: {len(pending_user_inputs)}")
    
    interrupt_flag.clear()
    is_speaking = False
    
    if is_speaking:
        logger.info(f"AI is currently speaking, adding input to pending queue: '{user_text}'")
        with user_input_lock:
            pending_user_inputs = [(user_text, session_id, user_id)]  # Include user_id in pending inputs
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
    
    # Get conversation context
    context = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in conversation_history[-5:]])
    rag_context = rag.query(user_text) if rag else ""
    
    system_prompt = config.system_prompt if config else "You are a helpful AI assistant."
    if rag_context:
        system_prompt += f"\n\nRelevant context:\n{rag_context}"
        
    logger.info(f"System prompt prepared, context length: {len(context)}")
    
    # Send thinking status
    asyncio.run_coroutine_threadsafe(
        message_queue.put({
            "type": "status", 
            "message": "Thinking...",
            "session_id": session_id
        }),
        loop
    )
    
    try:
        logger.info("Generating AI response...")
        with llm_lock:
            ai_response = llm.generate_response(system_prompt, user_text, context)
        logger.info(f"AI response generated: '{ai_response[:100]}...'")
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        conversation_history.append({
            "timestamp": timestamp,
            "user": user_text,
            "ai": ai_response
        })
        
        # Save to database WITH USER ID
        try:
            db = SessionLocal()
            conv = Conversation(
                user_id=user_id,  # This can be None for anonymous users
                session_id=session_id,
                timestamp=timestamp,
                user_message=user_text,
                ai_message=ai_response,
                audio_path=""
            )
            db.add(conv)
            db.commit()
            
            index = speaker_counters[0]
            # Include user_id in filename for better organization
            if user_id:
                output_file = f"audio/ai/user_{user_id}/{session_id}_response_{index}.wav"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            else:
                output_file = f"audio/ai/{session_id}_response_{index}.wav"
                
            speaker_counters[0] += 1
            
            conv.audio_path = output_file
            db.commit()
            db.close()
            
            if user_id:
                logger.info(f"Conversation saved to database for user {user_id}, audio path: {output_file}")
            else:
                logger.info(f"Conversation saved to database (anonymous user), audio path: {output_file}")
                
        except Exception as e:
            logger.error(f"Database error: {e}")
            
        # Add to RAG system
        if rag:
            threading.Thread(target=lambda: rag.add_conversation(user_text, ai_response), daemon=True).start()
            
        # Send response to client via message_queue
        asyncio.run_coroutine_threadsafe(
            message_queue.put({
                "type": "response", 
                "text": ai_response,
                "session_id": session_id
            }),
            loop
        )
        
        logger.info("Response sent to message queue, starting audio generation...")
        
        time.sleep(0.5)
        
        if is_speaking:
            logger.warning("Still speaking when trying to start new audio - forcing interrupt")
            interrupt_flag.set()
            is_speaking = False
            time.sleep(0.5)
            
        interrupt_flag.clear()
        is_speaking = False
        
        # Start audio generation
        threading.Thread(
            target=lambda: audio_generation_thread(ai_response, output_file, session_id), 
            daemon=True
        ).start()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({
                "type": "error", 
                "message": "Failed to generate response",
                "session_id": session_id
            }),
            loop
        )


# Also update the process_pending_inputs function to handle user_id
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
        user_text, session_id, user_id = latest_input  # Unpack user_id
        process_user_input(user_text, session_id, user_id)  # Pass user_id

# Update audio_generation_thread to include session_id
def audio_generation_thread(text, output_file, session_id="default"):
    global is_speaking, interrupt_flag, audio_queue, model_thread_running, current_generation_id, speaking_start_time, generator
    current_generation_id += 1
    this_id = current_generation_id
    interrupt_flag.clear()
    logger.info(f"Starting audio generation for ID: {this_id}, session: {session_id}")
    
    if not audio_gen_lock.acquire(blocking=False):
        logger.warning(f"Audio generation {this_id} - lock acquisition failed, another generation is in progress")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({
                "type": "error",
                "message": "Audio generation busy, skipping synthesis",
                "gen_id": this_id,
                "session_id": session_id
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
                "gen_id": this_id,
                "session_id": session_id
            }),
            loop
        )
        
        time.sleep(0.2)
        logger.info(f"Sending generating status with ID {this_id}")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({
                "type": "audio_status",
                "status": "generating",
                "gen_id": this_id,
                "session_id": session_id
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
                            "gen_id": this_id,
                            "session_id": session_id
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
                        "chunk_num": chunk_counter,
                        "session_id": session_id
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
                save_audio_and_trim(output_file, session_id, config.voice_speaker_id, complete_audio, generator.sample_rate)
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
                    "gen_id": this_id,
                    "session_id": session_id
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



# Inside your Python code, update the migrate_database function
# Ensure the migrate_database function includes creating the feedback table
def migrate_database():
    """Add user_id and starred columns to conversations table if they don't exist, and create feedback table"""
    try:
        conn = sqlite3.connect("companion.db")
        cursor = conn.cursor()

        # Check for user_id column
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'user_id' not in columns:
            print("Adding user_id column to conversations table...")
            cursor.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER")
            conn.commit()
            print("user_id column added successfully")

        # Check for starred column
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'starred' not in columns:
            print("Adding starred column to conversations table...")
            cursor.execute("ALTER TABLE conversations ADD COLUMN starred INTEGER DEFAULT 0")
            conn.commit()
            print("starred column added successfully")
        else:
            print("starred column already exists")

        # Check if feedback table exists, create if not
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback';")
        if not cursor.fetchone():
            print("Creating feedback table...")
            cursor.execute('''
                CREATE TABLE feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    email TEXT,
                    feedback_type TEXT,
                    message TEXT,
                    timestamp TEXT DEFAULT (datetime('now'))
                )
            ''')
            conn.commit()
            print("feedback table created successfully")
        else:
            print("feedback table already exists")

        conn.close()
    except Exception as e:
        print(f"Migration error: {e}")


# Ensure migrate_database is called during startup (it already is in your code)
# @app.on_event("startup")
# async def startup_event():
#     migrate_database()  # This line should already be present
#     # ... rest of your startup code ...



@app.get("/api/debug/models")
async def debug_models():
    return {
        "models_loaded": models_loaded,
        "whisper_loaded": whisper_model is not None,
        "llm_loaded": llm is not None,
        "rag_loaded": rag is not None,
        "config_loaded": config is not None
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

# Update your routes to remove authentication:
@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})




# History and user endpoints
@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    current_user = await get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login")
    
    # Get all users for the dropdown (admin feature)
    db = SessionLocal()
    try:
        all_users = db.query(User).order_by(User.email).all()
        users_list = [{"id": user.id, "email": user.email} for user in all_users]
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        users_list = []
    finally:
        db.close()
    
    return templates.TemplateResponse("history.html", {
        "request": request,
        "user_email": current_user.email,  # Access as object attribute
        "user_id": current_user.id,        # Access as object attribute
        "created_at": current_user.created_at,  # Access as object attribute
        "all_users": users_list
    })




@app.get("/api/user/conversations")
async def get_user_conversations(request: Request, user_id: int = None):
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # If no user_id provided, use current user's ID
    target_user_id = user_id if user_id else current_user.id

    db = SessionLocal()
    try:
        # Fetch conversations including the 'starred' field
        conversations = db.query(Conversation).filter(
            Conversation.user_id == target_user_id
        ).order_by(Conversation.timestamp.desc()).limit(50).all()

        return [{
            "id": conv.id,
            "timestamp": conv.timestamp,
            "user_message": conv.user_message,
            "ai_message": conv.ai_message,
            "audio_path": conv.audio_path,
            "starred": bool(conv.starred) # Include the starred status in the response
        } for conv in conversations]
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail="Error fetching conversations")
    finally:
        db.close()




@app.put("/api/user/conversations/{conv_id}/star")
async def toggle_conversation_star(conv_id: int, request: Request, data: dict): # Accept the star status from the request body
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    starred_value = data.get("starred") # Expecting True/False or 1/0 from the frontend
    if starred_value is None:
        raise HTTPException(status_code=400, detail="Starred status not provided")

    # Convert boolean to integer if necessary (0 or 1)
    if isinstance(starred_value, bool):
        starred_int = 1 if starred_value else 0
    elif isinstance(starred_value, int) and starred_value in [0, 1]:
        starred_int = starred_value
    else:
        raise HTTPException(status_code=400, detail="Invalid starred status value. Must be true/false or 0/1.")

    db = SessionLocal()
    try:
        # Find the specific conversation for the current user
        conversation = db.query(Conversation).filter(
            Conversation.id == conv_id,
            Conversation.user_id == current_user.id # Ensure user can only star their own conversations
        ).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found or does not belong to user")

        # Update the starred status
        conversation.starred = starred_int
        db.commit()
        db.refresh(conversation) # Refresh the object to get the updated values

        return {
            "id": conversation.id,
            "starred": bool(conversation.starred) # Return boolean for clarity
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating conversation star status: {e}")
        raise HTTPException(status_code=500, detail="Error updating conversation star status")
    finally:
        db.close()



# Inside your Python code (e.g., main.py), add this route

@app.get("/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request):
    current_user = await get_current_user(request)
    if current_user:
        user_email = current_user.email
        user_id = current_user.id
    else:
        # Decide how to handle anonymous feedback. For now, we'll pass None.
        # Or redirect to login if required.
        user_email = None
        user_id = None
        # Uncomment the next line if you want to require login for feedback:
        # return RedirectResponse(url="/login")

    # Pass the user email (and potentially user_id) to the template context
    return templates.TemplateResponse("feedback.html", {
        "request": request,
        "user_email": user_email,
        "user_id": user_id # Pass user_id if needed for form submission
    })



# Also add the POST route to handle the submission
@app.post("/feedback")
async def submit_feedback(request: Request, feedback_data: dict = None): # Accept JSON data
    current_user = await get_current_user(request)
    if not current_user:
        # Optionally, allow anonymous feedback or require login
        # For this example, we'll require login
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    # If feedback_data is not provided via JSON body, try form data
    if not feedback_data:
        try:
            feedback_data = await request.json()
        except:
            try:
                feedback_data = await request.form()
                feedback_data = dict(feedback_data) # Convert FormData to dict
            except:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid feedback data format")

    feedback_type = feedback_data.get("feedbackType")
    feedback_message = feedback_data.get("feedbackMessage")
    # Optional: Get email from form if not relying on session (less secure)
    # feedback_email = feedback_data.get("feedbackEmail")

    if not feedback_type or not feedback_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Feedback type and message are required")

    db = SessionLocal()
    try:
        feedback_entry = Feedback(
            user_id=current_user.id, # Use the authenticated user's ID
            email=current_user.email, # Use the authenticated user's email
            feedback_type=feedback_type,
            message=feedback_message
        )
        db.add(feedback_entry)
        db.commit()
        db.refresh(feedback_entry) # Optional: Get the ID of the new entry
        logger.info(f"Feedback submitted by user {current_user.email} (ID: {current_user.id}): {feedback_type}")
        return JSONResponse(content={"message": "Feedback submitted successfully"}, status_code=200)
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Error submitting feedback")
    finally:
        db.close()





@app.get("/api/user/feedback")
async def get_user_feedback(request: Request, user_id: int = None):
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # If no user_id provided, use current user's ID (for security, only allow viewing own feedback unless admin)
    # For this example, we'll allow viewing feedback for the user specified by user_id if it matches the current user or if admin logic is added later.
    # For now, restrict to current user or the specified user ID (assuming it's the same as current or checked elsewhere if admin).
    # The original conversation route already filters by user_id, so this follows the same pattern.
    # Be careful with authorization here if you allow one user to see another's feedback without being admin.
    # For now, let's assume the client-side logic correctly passes the intended user_id, and the backend just fetches it.
    # The original /api/user/conversations already does this, so we follow suit.
    # The calling user (current_user) should ideally be authorized to view the requested user_id's data.
    # If user_id is not provided or is the current user's ID, allow it.
    # If user_id is different, you might need admin checks here.
    # For this implementation, we'll proceed with the filter as requested by the client-side logic,
    # but ensure the requesting user is authenticated.

    target_user_id = user_id if user_id else current_user.id

    # Optional: Add admin check here if needed
    # if target_user_id != current_user.id:
    #     # Check if current_user is admin
    #     if not current_user.is_admin: # Assuming you have an is_admin field
    #         raise HTTPException(status_code=403, detail="Access forbidden")

    db = SessionLocal()
    try:
        # Fetch feedback entries for the specified user, ordered by timestamp descending
        feedback_entries = db.query(Feedback).filter(
            Feedback.user_id == target_user_id
        ).order_by(Feedback.timestamp.desc()).all()

        return [{
            "id": fb.id,
            "feedback_type": fb.feedback_type,
            "message": fb.message,
            "timestamp": fb.timestamp
        } for fb in feedback_entries]
    except Exception as e:
        logger.error(f"Error fetching feedback for user {target_user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching feedback")
    finally:
        db.close()



@app.get("/api/user/profile")
async def get_user_profile(request: Request):
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {
        "email": current_user.email,      # Access as object attribute
        "user_id": current_user.id,       # Access as object attribute
        "created_at": current_user.created_at  # Access as object attribute
    }      



@app.delete("/api/user/conversations")
async def delete_user_conversations(request: Request):
    current_user = await get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    db = SessionLocal()
    try:
        # Delete user's conversations
        deleted_count = db.query(Conversation).filter(
            Conversation.user_id == current_user.id  # Access as object attribute
        ).delete()
        
        db.commit()
        
        return {
            "message": f"Deleted {deleted_count} conversations",
            "deleted_count": deleted_count
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting conversations: {e}")
        raise HTTPException(status_code=500, detail="Error deleting conversations")
    finally:
        db.close()


@app.get("/")
async def root():
    # Force clear any cookies and redirect to login
    response = RedirectResponse(url="/login")
    return response





@app.post("/register")
async def register_user(user_data: UserCreate, response: Response):
    db = SessionLocal()
    
    try:
        # Check if user already exists
        existing_user = get_user_by_email(db, user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="User with this email already exists"
            )

        # Create new user
        user = create_user(db, user_data.email, user_data.password)
        
        # Create session - this should work now since user is still attached
        session_token = create_user_session(db, user.id)
        
        # Set session cookie
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            max_age=7*24*60*60,  # 7 days
            samesite="lax"
        )
        
        return {
            "message": "Registration successful", 
            "email": user.email,  # This should work now
            "user_id": user.id    # This should work now
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during registration: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
    finally:
        db.close()



@app.post("/login")
async def login_user(user_data: UserLogin, response: Response):
    db = SessionLocal()
    
    try:
        user = authenticate_user(db, user_data.email, user_data.password)
        if not user:
            raise HTTPException(
                status_code=400,
                detail="Invalid email or password"
            )
        
        # Create new session
        session_token = create_user_session(db, user.id)
        
        # Set session cookie
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            max_age=7*24*60*60,  # 7 days
            samesite="lax"
        )
        
        return {
            "message": "Login successful", 
            "email": user.email,  # This should work now
            "user_id": user.id    # This should work now
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(status_code=500, detail="Login failed")
    finally:
        db.close()


@app.post("/logout")
async def logout_user(response: Response):
    # Clear the session cookie
    response.delete_cookie("session_token")
    return {"message": "Logout successful"}






async def get_current_user(request: Request):
    db = SessionLocal()
    
    # Get session token from cookie
    session_token = request.cookies.get("session_token")
    
    if not session_token:
        db.close()
        return None
    
    user = get_user_from_session(db, session_token)
    db.close()
    return user

# Protected endpoint example
@app.get("/api/user/conversations_")
async def get_user_conversations(current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    db = SessionLocal()
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.timestamp.desc()).all()
    
    db.close()
    
    return [{
        "id": conv.id,
        "timestamp": conv.timestamp,
        "user_message": conv.user_message,
        "ai_message": conv.ai_message,
        "audio_path": conv.audio_path
    } for conv in conversations]




@app.on_event("startup")
async def startup_event():
    migrate_database()  # Add this line
    """Initialize all models and resources before the application starts"""
    logger.info("Starting application initialization...")
    
    # Run database migrations
    migrate_database()
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    # Create necessary directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("audio/user", exist_ok=True)
    os.makedirs("audio/ai", exist_ok=True)
    os.makedirs("embeddings_cache", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    # Load core models sequentially
    try:
        # 1. Load Whisper model first
        load_whisper_model()
        
        # 2. Load other models from saved config if available
        saved_config = config_manager.load_config()
        if saved_config:
            logger.info("Found saved configuration, initializing models...")
            try:
                config_data = CompanionConfig(**saved_config)
                initialize_models(config_data)
                logger.info("All models initialized from saved configuration")
            except Exception as e:
                logger.warning(f"Failed to initialize models from saved config: {e}")
                logger.info("Application will start with Whisper model only")
        else:
            logger.info("No saved configuration found. Application will start with Whisper model only")
            
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}")
        # Don't raise the exception - let the application start with limited functionality
        logger.warning("Application starting with limited functionality due to model loading errors")

    # Preload VAD model in background
    try:
        torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    except Exception as e:
        logger.warning(f"Could not preload VAD model: {e}")

    # Start message queue processing
    asyncio.create_task(process_message_queue())
    
    logger.info("Application startup completed")



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

@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    return templates.TemplateResponse("setup.html", {"request": request})




@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server shutting down...")
    model_thread_running.clear()





@app.get("/api/status")
async def get_system_status():
    return {
        "models_loaded": models_loaded,
        "whisper_loaded": whisper_model is not None,
        "llm_loaded": llm is not None,
        "rag_loaded": rag is not None,
        "vad_loaded": vad_processor is not None,
        "config_loaded": config is not None
    }



@app.get("/conversations", response_class=HTMLResponse)
async def conversations_page(request: Request):
    current_user = await get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("conversations.html", {
        "request": request,
        "user_email": current_user.email,
        "user_id": current_user.id
    })



@app.get("/test")
async def test_endpoint():
    return {"status": "Server is running", "timestamp": datetime.now().isoformat()}



@app.post("/test-text")
async def test_text_input(data: dict):
    user_text = data.get("text", "")
    if user_text:
        # Process directly without WebSocket
        threading.Thread(target=lambda: process_user_input(user_text, "test"), daemon=True).start()
        return {"status": "processing", "text": user_text}
    return {"status": "error", "message": "No text provided"}



@app.get("/logout")
async def logout_page(response: Response):
    # Clear the cookie and redirect to login
    response = RedirectResponse(url="/login")
    return response
    

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
    
    # Start the server only after models are loaded
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)