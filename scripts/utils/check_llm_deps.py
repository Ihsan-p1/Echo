try:
    import torch
    import transformers
    import peft
    import bitsandbytes
    import datasets
    print("SUCCESS: All LLM training dependencies (torch, transformers, peft, bitsandbytes, datasets) are available.")
except ImportError as e:
    print(f"FAILED: {e}")
except Exception as e:
    print(f"ERROR: {e}")
