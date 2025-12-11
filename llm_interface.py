import re
from typing import List, Dict, Any, Optional
from llama_cpp import Llama

class LLMInterface:
    def __init__(self, model_path: str, max_tokens: int = 8192, n_threads: int = 8, gpu_layers: int = 1):
        """Initialize the LLM interface using llama.cpp.
        
        Args:
            model_path (str): Path to the GGUF model file
            max_tokens (int): Maximum context length
            n_threads (int): Number of CPU threads
            gpu_layers (int): Number of layers to offload to Metal GPU (Apple Silicon)
        """
        self.model = Llama(
            model_path=model_path,
            n_ctx=max_tokens,
            n_threads=n_threads,
            n_gpu_layers=gpu_layers,  # Metal acceleration for Apple Silicon
            verbose=False
        )
        
        self.config = {
            "model_path": model_path,
            "max_tokens": max_tokens,
            "n_threads": n_threads,
            "gpu_layers": gpu_layers
        }
        
    def trim_to_last_sentence(self, text: str) -> str:
        """
        Improved sentence boundary detection with:
        - Support for multiple terminators (., !, ?)
        - Handling of nested quotes/brackets
        - Preservation of trailing non-terminator content
        """
        # Try to find the last proper sentence boundary
        match = re.search(r'(.*?[.!?][\'"\)\]]*)(?:\s+[^\w\s]|$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: Return entire text if no punctuation found
        return text.strip()

    def _format_prompt(self, system_prompt: str, user_message: str, 
                      conversation_history: str = "") -> str:
        """Consistent prompt formatting following chat template"""
        return f"""<|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
{conversation_history}
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    def generate_response(self, system_prompt: str, user_message: str, 
                         conversation_history: str = "") -> str:
        prompt = self._format_prompt(system_prompt, user_message, conversation_history)
        
        # Enhanced generation parameters
        output = self.model.create_completion(
            prompt=prompt,
            temperature=1.0,
            top_p=0.95,
            max_tokens=100,
            repeat_penalty=1.2,
            stop=[
                "</s>", "<|eot_id|>", "<|im_end|>", 
                "user:", "User:", "assistant:", "Assistant:",
                "\n\n"
            ]
        )
        
        return self.trim_to_last_sentence(output["choices"][0]["text"])

    def tokenize(self, text: str) -> List[int]:
        return self.model.tokenize(text.encode("utf-8"))

    def get_token_count(self, text: str) -> int:
        return len(self.tokenize(text))

    def batch_generate(self, prompts: List[Dict[str, str]], 
                      max_tokens: int = 512, 
                      temperature: float = 0.7) -> List[str]:
        formatted_prompts = [
            self._format_prompt(
                p.get("system", ""),
                p.get("user", ""),
                p.get("history", "")
            ) 
            for p in prompts
        ]
        
        results = []
        for prompt in formatted_prompts:
            output = self.model.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
                repeat_penalty=1.2,
                stop=[
                    "</s>", "<|eot_id|>", "<|im_end|>", 
                    "user:", "User:", "assistant:", "Assistant:",
                    "\n\n"
                ]
            )
            results.append(self.trim_to_last_sentence(output["choices"][0]["text"]))
            
        return results