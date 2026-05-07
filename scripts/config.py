# scripts/config.py
"""All configuration in one place - change models here"""

# Models (change these if you don't have LLaMA access)
# STUDENT_MODEL = "meta-llama/Llama-2-7b-hf"      # 7B for baseline & quantization
# TEACHER_MODEL = "meta-llama/Llama-2-13b-hf"    # 13B for distillation

# Fallback models if LLaMA access delayed (uncomment to use)
STUDENT_MODEL = "mistralai/Mistral-7B-v0.1"
TEACHER_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # For quick testing

# Experiment settings
MAX_NEW_TOKENS = 128
MAX_SEQ_LEN = 2048
TEST_SAMPLES = 100  # Number of WikiText samples to evaluate

# Output
RESULTS_FILE = "results/metrics.csv"