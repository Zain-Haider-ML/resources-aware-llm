# scripts/baseline.py
"""Run FP16 baseline evaluation"""

!git clone https://github.com/Zain-Haider-ML/resources-aware-llm.git
%cd resources-aware-llm
%cd scripts


import sys
sys.path.append('.')
from config import STUDENT_MODEL, MAX_NEW_TOKENS, TEST_SAMPLES
from utils import load_model, measure_latency, measure_memory, compute_perplexity, get_wikitext_samples, save_result

def run_baseline():
    print("\n" + "="*60)
    print("BASELINE: FP16 (Full Precision)")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model(STUDENT_MODEL, quant_type=None)
    
    # Get test data
    texts = get_wikitext_samples(TEST_SAMPLES)
    
    # Measure
    print("\n📊 Measuring perplexity...")
    ppl = compute_perplexity(model, tokenizer, texts)
    
    print("\n⏱️ Measuring latency...")
    torch.cuda.reset_peak_memory_stats()
    latency = measure_latency(model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)
    memory = measure_memory()
    
    # Save
    result = {
        'model': STUDENT_MODEL.split('/')[-1],
        'method': 'FP16_baseline',
        'quantization': 'none',
        'perplexity': ppl,
        'latency_ms_per_token': latency['latency_ms_per_token'],
        'throughput_tokens_per_sec': latency['throughput_tokens_per_sec'],
        'peak_memory_gb': memory['peak_gpu_memory_gb'],
    }
    
    save_result(result)
    
    print("\n" + "="*60)
    print(f"RESULTS:")
    print(f"  Perplexity: {ppl}")
    print(f"  Latency: {latency['latency_ms_per_token']} ms/token")
    print(f"  Throughput: {latency['throughput_tokens_per_sec']} tokens/sec")
    print(f"  Memory: {memory['peak_gpu_memory_gb']} GB")
    print("="*60)
    
    return result

if __name__ == "__main__":
    run_baseline()