# scripts/utils.py
"""Minimal metrics functions for thesis experiments"""

import torch
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def load_model(model_name, quant_type=None):
    """
    Load model with optional quantization.
    
    quant_type options:
        None   -> FP16 (full precision)
        "8bit" -> 8-bit quantization
        "4bit" -> 4-bit quantization (NF4)
    """
    print(f"\nLoading: {model_name}")
    
    if quant_type == "4bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quant_type == "8bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16 if quant_config is None else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Print model size (helpful for thesis)
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"   Parameters: {param_count:.1f}B")
    
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"   GPU Memory: {mem:.2f} GB")
    
    return model, tokenizer


def measure_latency(model, tokenizer, prompt="The capital of France is", max_new_tokens=128):
    """Returns latency in ms/token and throughput in tokens/sec"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Warmup
    for _ in range(2):
        _ = model.generate(**inputs, max_new_tokens=32)
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    latency_ms = (elapsed / gen_tokens) * 1000
    
    return {
        'latency_ms_per_token': round(latency_ms, 2),
        'throughput_tokens_per_sec': round(1000 / latency_ms, 2),
        'tokens_generated': gen_tokens,
    }


def measure_memory():
    """Returns peak GPU memory in GB"""
    if torch.cuda.is_available():
        return {'peak_gpu_memory_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    return {'peak_gpu_memory_gb': 0}


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Lower is better. Returns perplexity score."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for text in texts:
        if len(text.strip()) < 10:
            continue
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to("cuda")
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.shape[1]
            total_tokens += input_ids.shape[1]
        
        # Clear memory between samples
        torch.cuda.empty_cache()
    
    if total_tokens == 0:
        return float('inf')
    
    return round(torch.exp(torch.tensor(total_loss / total_tokens)).item(), 2)


def get_wikitext_samples(num_samples=100, split="test"):
    """Get N samples from WikiText-103 for evaluation."""
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    
    texts = []
    for item in dataset:
        text = item["text"].strip()
        if len(text) > 50:  # Only substantial texts
            texts.append(text)
            if len(texts) >= num_samples:
                break
    
    print(f"Loaded {len(texts)} samples from WikiText")
    return texts


def save_result(result_dict):
    """Append one experiment result to CSV file"""
    import os
    os.makedirs("results", exist_ok=True)
    
    df = pd.DataFrame([result_dict])
    if os.path.exists("results/metrics.csv"):
        existing = pd.read_csv("results/metrics.csv")
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv("results/metrics.csv", index=False)
    print(f"Result saved to results/metrics.csv")