import pytest
from fine_tuning.model import LLMModel

@pytest.fixture(scope="module")
def model():
    return LLMModel("config.yaml")

def test_generate(model):
    prompt = "What is a bond yield?"
    output = model.generate(prompt, max_length=50)
    # Check if the output is a non-empty string
    assert isinstance(output, str)
    assert len(output.strip()) > 0
