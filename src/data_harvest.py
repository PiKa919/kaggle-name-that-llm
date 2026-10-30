"""
Data harvesting and validation pipeline for external multi-model response corpora.
Ensures external passages strictly conform to competition invariants:
- Single paragraph (no double newlines)
- Authentic completion length (40 to 250 words)
- Valid model string belonging to the 18 candidate classes
- Preserves all casing, quotes, dashes, and typography
"""

import re
from typing import Optional
import pandas as pd
from src.config import VALID_18_MODELS


def clean_passage(text: str) -> str:
    """Strip superficial markdown wrappers while keeping all internal tokens intact."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Strip whole-passage markdown bold or italic wrappers if present
    if (text.startswith("**") and text.endswith("**")) or (text.startswith("__") and text.endswith("__")):
        text = text[2:-2].strip()
    elif (text.startswith("*") and text.endswith("*")) or (text.startswith("_") and text.endswith("_")):
        text = text[1:-1].strip()
    return text


def validate_external_passage(text: str, model: str) -> bool:
    """Validate whether an external candidate passage satisfies competition specs."""
    if model not in VALID_18_MODELS:
        return False
    if not isinstance(text, str):
        return False
    
    text = clean_passage(text)
    
    # Must be a single paragraph (no blank line breaks)
    if "\n\n" in text or "\r\n\r\n" in text:
        return False
    
    # Check word count between 40 and 250 words
    words = text.split()
    if len(words) < 40 or len(words) > 250:
        return False
        
    return True


def filter_and_standardize_corpus(df: pd.DataFrame, 
                                  text_col: str = "text", 
                                  model_col: str = "model", 
                                  prompt_col: Optional[str] = "prompt_id") -> pd.DataFrame:
    """
    Standardize column names, clean passages, and filter out non-conforming rows.
    """
    records = []
    for idx, row in df.iterrows():
        t = row[text_col]
        m = str(row[model_col]).strip()
        if validate_external_passage(t, m):
            rec = {
                "text": clean_passage(t),
                "model": m,
            }
            if prompt_col and prompt_col in row and pd.notna(row[prompt_col]):
                rec["prompt_id"] = str(row[prompt_col])
            else:
                rec["prompt_id"] = f"ext_{idx}"
            records.append(rec)
            
    out_df = pd.DataFrame(records)
    return out_df
