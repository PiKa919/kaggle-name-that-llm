import os
import pandas as pd
import pytest
from src.config import VALID_18_MODELS

def test_submission_parquet_schema(tmp_path):
    sub_path = tmp_path / "submission.parquet"
    
    # Generate valid dummy submission
    df = pd.DataFrame({
        "id": ["test_001", "test_002", "test_003"],
        "model": ["gpt-5.5", "claude-opus-5", "gemini-3.6-flash"]
    })
    df.to_parquet(sub_path, index=False)
    
    # Verification assertions
    sub = pd.read_parquet(sub_path)
    assert list(sub.columns) == ["id", "model"]
    assert len(sub) == 3
    assert not sub.isnull().any().any()
    assert sub["model"].isin(VALID_18_MODELS).all()
