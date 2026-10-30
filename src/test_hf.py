from transformers import AutoTokenizer

print("Testing Hugging Face model access...")
try:
    tok_deb = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    print(f"[OK] microsoft/deberta-v3-base tokenizer loaded: {len(tok_deb)} tokens")
except Exception as e:
    print(f"[ERR] deberta: {e}")

try:
    tok_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    print(f"[OK] Qwen/Qwen2.5-1.5B tokenizer loaded: {len(tok_qwen)} tokens")
except Exception as e:
    print(f"[ERR] qwen 1.5B: {e}")

try:
    tok_qwen7 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    print(f"[OK] Qwen/Qwen2.5-7B tokenizer loaded: {len(tok_qwen7)} tokens")
except Exception as e:
    print(f"[ERR] qwen 7B: {e}")
