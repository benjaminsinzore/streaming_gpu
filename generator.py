from dataclasses import dataclass
import math
import os
from typing import List, Tuple, Generator as PyGenerator, Optional, Callable
import time
import queue
import threading
import platform
from typing_extensions import OrderedDict
import wave
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

@dataclass
class Segment:
    speaker: int
    text: str
    sample_rate = 24_000
    audio: torch.Tensor

def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )
    return tokenizer

class Generator:
    def __init__(self, model: Model):
        self._model = model
        self._model.setup_caches(1)
        self._text_tokenizer = load_llama3_tokenizer()
        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        num_codebooks = model.config.audio_num_codebooks
        mimi.set_num_codebooks(num_codebooks)
        self._num_codebooks = num_codebooks
        self._audio_tokenizer = mimi
        self.sample_rate = mimi.sample_rate
        self.device = device
        self._stream_buffer_size = 20
        self.max_seq_len = 2048
        self._cache = OrderedDict()
        self._text_token_cache = {}
        torch.set_num_threads(16)

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = f"{speaker}:{text}"
        if not hasattr(self, '_text_token_cache'):
            self._text_token_cache = {}
        if cache_key in self._text_token_cache:
            return self._text_token_cache[cache_key]
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), self._num_codebooks+1, dtype=torch.long, device=self.device)
        text_frame_mask = torch.zeros(len(text_tokens), self._num_codebooks+1, dtype=torch.bool, device=self.device)
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True
        frame_tokens = [text_frame]
        frame_masks = [text_frame_mask]
        result = (torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0))
        self._text_token_cache[cache_key] = result
        return result

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        audio_tokens = audio_tokens[:self._num_codebooks, :]
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
        audio_frame = torch.zeros(audio_tokens.size(1), self._num_codebooks+1).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self._num_codebooks+1).bool().to(self.device)
        audio_frame[:, :self._num_codebooks] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :self._num_codebooks] = True
        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        total_len = text_tokens.size(0) + audio_tokens.size(0)
        if total_len > self.max_seq_len:
            overflow = total_len - self.max_seq_len
            if text_tokens.size(0) > overflow:
                text_tokens = text_tokens[overflow:]
                text_masks = text_masks[overflow:]
            else:
                audio_overflow = overflow - text_tokens.size(0)
                text_tokens = text_tokens[0:0]
                text_masks = text_masks[0:0]
                audio_tokens = audio_tokens[audio_overflow:]
                audio_masks = audio_masks[audio_overflow:]
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def _decode_frames(self, frames):
        if not frames:
            return torch.tensor([])
        frames_reduced = [frame[:, :self._num_codebooks//2] for frame in frames]
        audio = self._audio_tokenizer.decode(torch.stack(frames_reduced).permute(1, 2, 0)).squeeze(0).squeeze(0)
        return audio

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 30,
        on_chunk_generated: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        self._model.reset_caches()
        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        initial_batch_size = 20
        normal_batch_size = 20
        initial_buffer_size = 20
        normal_buffer_size = 20
        batch_size = initial_batch_size
        buffer_size = initial_buffer_size
        first_chunk_delivered = False
        context_start = time.time()
        if context:
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        max_seq_len = 2048
        if prompt_tokens.size(0) > max_seq_len:
            prompt_tokens = prompt_tokens[-max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-max_seq_len:]
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        expected_frame_count = buffer_size
        frame_buffer = []
        zeros_1_1 = torch.zeros(1, 1).long().to(self.device)
        zeros_mask_1_1 = torch.zeros(1, 1).bool().to(self.device)

        def update_tokens(sample):
            nonlocal curr_tokens, curr_tokens_mask, curr_pos
            ones = torch.ones_like(sample).bool()
            curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones, zeros_mask_1_1], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        with self._audio_tokenizer.streaming(1):
            i = 0
            generation_start = time.time()
            while i < max_generation_len:
                batch_end = min(i + batch_size, max_generation_len)
                batch_size_actual = batch_end - i
                batch_samples = []
                for _ in range(batch_size_actual):
                    with torch.autocast(device_type=self.device.type, dtype=torch.float32):  # Use float32 for CPU
                        sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                        if sample.numel() == 0 or torch.isnan(sample).any():
                            print("Warning: Generated empty or NaN sample, stopping generation")
                            break
                    if torch.all(sample == 0):
                        break
                    batch_samples.append(sample)
                    update_tokens(sample)
                if not batch_samples:
                    break
                frame_buffer.extend(batch_samples)
                i += len(batch_samples)
                if len(frame_buffer) >= buffer_size:
                    frames_to_process = frame_buffer[:expected_frame_count]
                    if len(frames_to_process) < expected_frame_count:
                        padding_frames = [
                            torch.zeros_like(frames_to_process[0])
                            for _ in range(expected_frame_count - len(frames_to_process))
                        ]
                        frames_to_process = frames_to_process + padding_frames
                    frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                    audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                    frame_buffer = frame_buffer[expected_frame_count:]
                    cpu_chunk = audio_chunk.cpu()
                    if on_chunk_generated:
                        on_chunk_generated(cpu_chunk)
                    if not first_chunk_delivered:
                        batch_size = normal_batch_size
                        buffer_size = normal_buffer_size
                        expected_frame_count = buffer_size
                        first_chunk_delivered = True
                    yield cpu_chunk
                    if i >= 100 and (i % 100 == 0):
                        print(f"Generated {i} frames ({i * 0.08:.2f}s of audio)")
            if frame_buffer:
                if len(frame_buffer) < expected_frame_count:
                    padding_frames = [
                        torch.zeros_like(frame_buffer[0])
                        for _ in range(expected_frame_count - len(frame_buffer))
                    ]
                    frames_to_process = frame_buffer + padding_frames
                else:
                    frames_multiple = (len(frame_buffer) // expected_frame_count) * expected_frame_count
                    frames_to_process = frame_buffer[:frames_multiple]
                frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                actual_frames_percentage = min(len(frame_buffer), expected_frame_count) / expected_frame_count
                actual_samples = int(audio_chunk.shape[0] * actual_frames_percentage)
                if len(frame_buffer) < expected_frame_count:
                    audio_chunk = audio_chunk[:actual_samples]
                cpu_chunk = audio_chunk.cpu()
                if on_chunk_generated:
                    on_chunk_generated(cpu_chunk)
                yield cpu_chunk
            total_time = time.time() - generation_start
            frames_generated = i
            audio_seconds = frames_generated * 0.08
            rtf = total_time / audio_seconds if audio_seconds > 0 else float('inf')
            print(f"Total time: {total_time:.2f}s")
            print(f"Generated {frames_generated} frames ({audio_seconds:.2f}s of audio)")
            print(f"Real-time factor: {rtf:.3f}x (target: <1.0)")

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.8,
        topk: int = 40,
        stream: bool = False,
        output_file: Optional[str] = None,
    ):
        if stream:
            if output_file:
                write_chunk, close_wav = stream_audio_to_wav(output_file, self.sample_rate)
                audio_chunks = []
                t1 = time.time()
                for i, chunk in enumerate(self.generate_stream(
                    text, speaker, context, max_audio_length_ms, temperature, topk
                )):
                    write_chunk(chunk)
                    audio_chunks.append(chunk)
                    if i % 5 == 0:
                        print(f"Part {i+1} available after {time.time() - t1:.4f}s")
                        t1 = time.time()
                close_wav()
                print(f"Streaming complete, WAV file saved to {output_file}")
            else:
                audio_chunks = []
                for chunk in self.generate_stream(text, speaker, context, max_audio_length_ms, temperature, topk):
                    audio_chunks.append(chunk)
            if not audio_chunks:
                return torch.tensor([])
            return torch.cat(audio_chunks)
        self._model.reset_caches()
        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        max_seq_len = 2048
        if prompt_tokens.size(0) > max_seq_len:
            prompt_tokens = prompt_tokens[-max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-max_seq_len:]
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        samples = []
        with self._audio_tokenizer.streaming(1):
            for _ in range(max_generation_len):
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0):
                    break
                samples.append(sample)
                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1
        if not samples:
            return torch.tensor([])
        return self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

class AudioStreamWriter:
    def __init__(self, filename, sample_rate):
        self.filename = filename
        self.sample_rate = sample_rate
        self.audio_chunks = []
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.running = True
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

    def _writer_worker(self):
        buffer_chunks = []
        last_flush_time = time.time()
        while self.running or not self.queue.empty():
            try:
                chunk = self.queue.get(timeout=0.2)
                buffer_chunks.append(chunk)
                current_time = time.time()
                if len(buffer_chunks) >= 10 or (current_time - last_flush_time > 2.0 and buffer_chunks):
                    with self.lock:
                        self.audio_chunks.extend(buffer_chunks)
                    buffer_chunks = []
                    last_flush_time = current_time
            except queue.Empty:
                if buffer_chunks:
                    with self.lock:
                        self.audio_chunks.extend(buffer_chunks)
                    buffer_chunks = []
                    last_flush_time = time.time()
        if buffer_chunks:
            with self.lock:
                self.audio_chunks.extend(buffer_chunks)

    def add_chunk(self, chunk):
        try:
            self.queue.put(chunk, timeout=0.1)
        except queue.Full:
            with self.lock:
                self.audio_chunks.append(chunk)

    def write_file(self):
        self.running = False
        self.writer_thread.join(timeout=3.0)
        with self.lock:
            if not self.audio_chunks:
                return
            audio = torch.cat(self.audio_chunks)
            torchaudio.save(self.filename, audio.unsqueeze(0).cpu(), self.sample_rate)

def load_csm_1b_local(model_path: str, device: str = "cpu", audio_num_codebooks: int = 32):
    from functools import lru_cache
    from generator import Generator, Model, ModelArgs
    print(f"Loading CSM-1B model from local checkpoint '{model_path}'...")
    config = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=audio_num_codebooks,
    )
    model = Model.from_pretrained(model_path)
    model.eval()
    dtype = torch.float32  # Use float32 for CPU
    try:
        model.backbone = torch.compile(model.backbone, mode='reduce-overhead', fullgraph=True, backend='inductor')
        model.decoder = torch.compile(model.decoder, mode='reduce-overhead', fullgraph=True, backend='inductor')
    except Exception as e:
        print(f"Warning: Torch compilation failed: {e}. Proceeding without compilation.")
    model.to(device=device, dtype=dtype)
    print("Model compilation complete. Creating generator...")
    generator = Generator(model)
    generator._stream_buffer_size = 20
    generator._tokenization_cache = {}
    original_tokenize_text = generator._tokenize_text_segment

    @lru_cache(maxsize=2048)
    def cached_tokenize_text_segment(text_str, speaker_int):
        return original_tokenize_text(text_str, speaker_int)

    generator._tokenize_text_segment = lambda text, speaker: cached_tokenize_text_segment(text, speaker)
    warmup_generator(generator)
    return generator

def warmup_generator(gen: Generator, warmup_text: str = "Hello, this is a comprehensive warmup text that will exercise the model's generation capabilities.", speaker_id: int = 0):
    print("Starting warmup sequence...")
    if hasattr(gen._model, 'backbone') and hasattr(gen._model.backbone, 'positional_embedding'):
        with torch.inference_mode():
            positions = torch.arange(0, 2048).to(gen.device)
            _ = gen._model.backbone.positional_embedding(positions)
    print("Creating diverse audio contexts...")
    audio_segments = []
    for i in range(3):
        length = 24000 * (i + 1)
        audio = torch.zeros(length).to(gen.device)
        if i == 0:
            t = torch.linspace(0, 8 * math.pi, length).to(gen.device)
            audio = torch.sin(t) * 0.1
        elif i == 1:
            audio = torch.randn(length).to(gen.device) * 0.05
        else:
            audio[::800] = 0.2
            audio[::801] = -0.2
        segment = Segment(
            speaker=speaker_id,
            text=f"Warmup segment {i+1} with {length/24000:.1f}s of audio.",
            audio=audio
        )
        audio_segments.append(segment)
    print("Forcing compilation of critical components...")
    with torch.inference_mode():
        for segment in audio_segments:
            gen._tokenize_segment(segment)
        dummy_tokens = torch.ones(1, 10, gen._num_codebooks+1).long().to(gen.device)
        dummy_mask = torch.ones(1, 10, gen._num_codebooks+1).bool().to(gen.device)
        dummy_pos = torch.arange(0, 10).unsqueeze(0).to(gen.device)
        for temp in [0.6, 0.7, 0.8]:
            for topk in [20, 30, 40]:
                _ = gen._model.generate_frame(dummy_tokens, dummy_mask, dummy_pos, temp, topk)
    gen._text_token_cache.clear()
    print("Running final generation with exact same setup as a real request...")
    final_text = "This is the final warmup that exactly matches a real generation request."
    gen._tokenize_text_segment(final_text, speaker_id)
    try:
        generate_streaming_audio(
            generator=gen,
            text=final_text,
            speaker=speaker_id,
            context=[audio_segments[0]],
            output_file="warmup_final.wav",
            max_audio_length_ms=6000,
            temperature=0.7,
            topk=30,
            play_audio=False
        )
    except Exception as e:
        print(f"Final warmup run exception (ignorable): {e}")
    print("Warmup complete.")

def load_csm_1b(device: str = "cpu") -> Generator:
    print("Loading CSM-1B model...")
    model = Model.from_pretrained("sesame/csm-1b")
    model.eval()
    dtype = torch.float32  # Use float32 for CPU
    try:
        model.backbone = torch.compile(model.backbone, mode='reduce-overhead', fullgraph=True, backend='inductor')
        model.decoder = torch.compile(model.decoder, mode='reduce-overhead', fullgraph=True, backend='inductor')
    except Exception as e:
        print(f"Warning: Torch compilation failed: {e}. Proceeding without compilation.")
    model.to(device=device, dtype=dtype)
    print("Model compilation complete. Creating generator...")
    generator = Generator(model)
    generator._stream_buffer_size = 20
    generator._tokenization_cache = {}
    from functools import lru_cache
    original_tokenize_text = generator._tokenize_text_segment

    @lru_cache(maxsize=2048)
    def cached_tokenize_text_segment(text_str, speaker_int):
        return original_tokenize_text(text_str, speaker_int)

    generator._tokenize_text_segment = lambda text, speaker: cached_tokenize_text_segment(text, speaker)
    warmup_generator(generator)
    return generator

def stream_audio_to_wav(filename, sample_rate):
    wav_file = wave.open(filename, 'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)

    def write_chunk(audio_chunk):
        if isinstance(audio_chunk, torch.Tensor):
            audio_np = audio_chunk.detach().cpu().numpy()
        else:
            audio_np = audio_chunk
        if audio_np.max() <= 1.0 and audio_np.min() >= -1.0:
            audio_int = (audio_np * 32767).astype(np.int16)
        else:
            audio_int = audio_np.astype(np.int16)
        wav_file.writeframes(audio_int.tobytes())

    def close():
        wav_file.close()

    return write_chunk, close

def generate_streaming_audio(
    generator: Generator,
    text: str,
    speaker: int,
    context: List[Segment],
    output_file: str,
    max_audio_length_ms: float = 90_000,
    temperature: float = 1.0,
    topk: int = 50,
    play_audio: bool = False,
):
    write_chunk, close_wav = stream_audio_to_wav(output_file, generator.sample_rate)
    audio_queue = queue.Queue(maxsize=100) if play_audio else None
    stop_event = threading.Event()
    if play_audio:
        try:
            import sounddevice as sd
            device_info = sd.query_devices(kind='output')
            supported_rate = device_info.get('default_samplerate', 44100)
            need_resampling = abs(supported_rate - generator.sample_rate) > 100
            if need_resampling:
                try:
                    import librosa
                    print(f"Resampling from {generator.sample_rate}Hz to {int(supported_rate)}Hz for playback")
                    def audio_playback_worker():
                        while not stop_event.is_set() or not audio_queue.empty():
                            try:
                                chunk = audio_queue.get(timeout=0.5)
                                if isinstance(chunk, torch.Tensor) and chunk.numel() == 0:
                                    audio_queue.task_done()
                                    continue
                                audio_np = chunk.numpy() if isinstance(chunk, torch.Tensor) else chunk
                                if len(audio_np) < 100:
                                    audio_queue.task_done()
                                    continue
                                resampled = librosa.resample(
                                    audio_np,
                                    orig_sr=generator.sample_rate,
                                    target_sr=int(supported_rate)
                                )
                                sd.play(resampled, supported_rate, blocking=True)
                                time.sleep(0.05)
                                audio_queue.task_done()
                            except queue.Empty:
                                if not stop_event.is_set():
                                    continue
                                else:
                                    break
                            except Exception as e:
                                print(f"Playback error: {e}")
                                audio_queue.task_done()
                except ImportError:
                    print("Librosa not found. Using direct playback.")
                    need_resampling = False
            if not need_resampling:
                def audio_playback_worker():
                    while not stop_event.is_set() or not audio_queue.empty():
                        try:
                            chunk = audio_queue.get(timeout=0.5)
                            if isinstance(chunk, torch.Tensor) and chunk.numel() == 0:
                                audio_queue.task_done()
                                continue
                            audio_np = chunk.numpy() if isinstance(chunk, torch.Tensor) else chunk
                            if len(audio_np) < 100:
                                audio_queue.task_done()
                                continue
                            sd.play(audio_np, generator.sample_rate, blocking=True)
                            time.sleep(0.05)
                            audio_queue.task_done()
                        except queue.Empty:
                            if not stop_event.is_set():
                                continue
                            else:
                                break
                        except Exception as e:
                            print(f"Playback error: {e}")
                            audio_queue.task_done()
            playback_thread = threading.Thread(target=audio_playback_worker, daemon=False)
            playback_thread.start()
        except ImportError:
            print("sounddevice library not found. Install with 'pip install sounddevice' for real-time playback.")
            play_audio = False
    chunk_times = []
    latency_to_first_chunk = None
    total_audio_duration = 0
    chunk_count = 0

    def on_chunk_generated(chunk):
        nonlocal chunk_count, latency_to_first_chunk, total_audio_duration
        current_time = time.time()
        if chunk_count == 0:
            latency_to_first_chunk = current_time - start_time
            print(f"First chunk latency: {latency_to_first_chunk*1000:.1f}ms")
        write_chunk(chunk)
        chunk_count += 1
        chunk_duration = len(chunk) / generator.sample_rate
        total_audio_duration += chunk_duration
        chunk_times.append(current_time)
        if play_audio and audio_queue is not None:
            try:
                audio_queue.put(chunk, timeout=1.0)
            except queue.Full:
                pass

    print(f"Starting audio generation for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    start_time = time.time()
    frame_count = 0
    audio_chunks = []
    try:
        for audio_chunk in generator.generate_stream(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            on_chunk_generated=on_chunk_generated
        ):
            frame_count += 1
            audio_chunks.append(audio_chunk)
            if frame_count % 10 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                if total_audio_duration > 0:
                    rtf = elapsed / total_audio_duration
                    remaining_time = (max_audio_length_ms/1000 - total_audio_duration) * rtf
                    print(f"Chunk {chunk_count}: {total_audio_duration:.1f}s audio in {elapsed:.1f}s "
                          f"(RTF: {rtf:.2f}x, Est. remaining: {remaining_time:.1f}s)")
    except Exception as e:
        print(f"Error during audio generation: {e}")
        import traceback
        traceback.print_exc()
    time.sleep(0.5)
    stop_event.set()
    close_wav()
    if play_audio and 'playback_thread' in locals():
        print("Waiting for audio playback to complete...")
        try:
            timeout_start = time.time()
            while not audio_queue.empty() and time.time() - timeout_start < 5.0:
                time.sleep(0.1)
        except:
            pass
        if hasattr(sd, 'wait'):
            try:
                sd.wait()
            except:
                pass
        playback_thread.join(timeout=5.0)
        try:
            sd.stop()
        except:
            pass
    end_time = time.time()
    total_elapsed = end_time - start_time
    if len(chunk_times) > 1:
        inter_chunk_latencies = [chunk_times[i] - chunk_times[i-1] for i in range(1, len(chunk_times))]
        avg_inter_chunk_latency = sum(inter_chunk_latencies) / len(inter_chunk_latencies)
        max_inter_chunk_latency = max(inter_chunk_latencies) if inter_chunk_latencies else 0
        min_inter_chunk_latency = min(inter_chunk_latencies) if inter_chunk_latencies else 0
    else:
        avg_inter_chunk_latency = max_inter_chunk_latency = min_inter_chunk_latency = 0
    rtf = total_elapsed / total_audio_duration if total_audio_duration > 0 else float('inf')
    print("\n" + "="*50)
    print("AUDIO GENERATION PERFORMANCE METRICS")
    print("="*50)
    print(f"First chunk latency: {latency_to_first_chunk*1000:.1f}ms")
    print(f"Total generation time: {total_elapsed:.2f}s")
    print(f"Audio duration: {total_audio_duration:.2f}s")
    print(f"Real-time factor (RTF): {rtf:.3f}x (target: <1.0)")
    print(f"Number of chunks: {chunk_count}")
    print(f"Average chunk size: {(total_audio_duration/chunk_count)*1000:.1f}ms") if chunk_count > 0 else None
    print(f"Average inter-chunk latency: {avg_inter_chunk_latency*1000:.1f}ms")
    print(f"Min/Max inter-chunk latency: {min_inter_chunk_latency*1000:.1f}ms / {max_inter_chunk_latency*1000:.1f}ms")
    print(f"Chunks per second: {chunk_count/total_elapsed:.2f}")
    print(f"Output file: {output_file}")
    print("="*50)