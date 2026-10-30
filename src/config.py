"""
Competition configuration and constants for 'Name That LLM'.
"""

from pathlib import Path

# Paths inside Kaggle environment
KAGGLE_INPUT_DIR = Path("/kaggle/input/competitions/name-that-llm")
TRAIN_CSV = KAGGLE_INPUT_DIR / "train.csv"
TEST_CSV = KAGGLE_INPUT_DIR / "test.csv"
SAMPLE_SUBMISSION = KAGGLE_INPUT_DIR / "sample_submission.parquet"

# All 18 valid candidate models
VALID_18_MODELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]
N_CLASSES = len(VALID_18_MODELS)

MODEL_TO_IDX = {model: i for i, model in enumerate(VALID_18_MODELS)}
IDX_TO_MODEL = {i: model for i, model in enumerate(VALID_18_MODELS)}

# Lab mapping for coarse analysis
MODEL_TO_LAB = {
    "claude-fable-5": "anthropic",
    "claude-opus-4-6": "anthropic",
    "claude-opus-4-8": "anthropic",
    "claude-opus-5": "anthropic",
    "claude-sonnet-5": "anthropic",
    "gemini-3.1-pro-preview": "google",
    "gemini-3.6-flash": "google",
    "gemini-3.7-flash": "google",
    "gemini-3.8-flash": "google",
    "glm-5.3": "zai",
    "glm-5.3-flash": "zai",
    "gpt-5.5": "openai",
    "gpt-5.6-luna": "openai",
    "gpt-5.6-sol": "openai",
    "gpt-5.6-terra": "openai",
    "gpt-6-astra": "openai",
    "kimi-k3": "moonshot",
    "qwen3.8-max": "qwen",
}

RANDOM_SEED = 42
N_SPLITS = 5
MAX_SEQ_LEN = 384
