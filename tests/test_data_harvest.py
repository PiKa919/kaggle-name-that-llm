import pytest
from src.config import VALID_18_MODELS

def test_passage_filter_invariants():
    from src.data_harvest import validate_external_passage
    
    # Reject multi-paragraph text
    assert not validate_external_passage("Paragraph 1.\n\nParagraph 2.", "gpt-5.5")
    
    # Reject outside length range (<40 words)
    assert not validate_external_passage("Too short words only ten here in this tiny passage.", "gpt-5.5")
    
    # Reject outside length range (>250 words)
    long_text = " ".join(["word"] * 300)
    assert not validate_external_passage(long_text, "gpt-5.5")
    
    # Reject invalid model name
    valid_text = " ".join(["word"] * 120)
    assert not validate_external_passage(valid_text, "invalid-model-name")
    
    # Accept valid passage
    assert validate_external_passage(valid_text, "gpt-5.5")
    assert validate_external_passage(valid_text, "claude-opus-5")
