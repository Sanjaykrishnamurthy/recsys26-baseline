from .llama import LLAMA_MODEL
from .ollama_lm import OLLAMA_MODEL

def load_lm_module(lm_type, device, attn_implementation, dtype):
    """Factory that loads any HuggingFace causal LM via LLAMA_MODEL.

    LLAMA_MODEL uses AutoModelForCausalLM + AutoTokenizer under the hood, so
    it works with any instruction-tuned model (Llama, Qwen, Mistral, etc.)
    as long as the model ID is a valid Hugging Face identifier.

    Args:
        lm_type: Hugging Face model ID (e.g. "Qwen/Qwen3-4B",
                 "meta-llama/Llama-3.2-1B-Instruct").
        device: Torch device string ("cuda", "mps", "cpu").
        attn_implementation: Attention backend ("eager" or "flash_attention_2").
        dtype: Torch dtype for model weights (e.g. torch.bfloat16).

    Returns:
        An initialised LLAMA_MODEL instance wrapping the requested model.
    """
    # Ollama models are prefixed with "ollama:" e.g. "ollama:llama3.2:1b"
    if lm_type.startswith("ollama:"):
        ollama_model_name = lm_type[len("ollama:"):]
        return OLLAMA_MODEL(model_name=ollama_model_name)

    # All other values are treated as HuggingFace model IDs
    return LLAMA_MODEL(
        model_name=lm_type,
        device=device,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
