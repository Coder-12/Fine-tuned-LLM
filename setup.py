from setuptools import setup, find_packages

setup(
    name="LLM-FineTuning",
    version="0.1",
    description="Fine-Tuning an LLM for Industry Specific Applications using LoRA",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.18.0",
        "peft>=0.3.0",
        "torch>=1.12.0",
        "fastapi>=0.78.0",
        "uvicorn>=0.18.0",
        "pyyaml>=6.0"
    ],
    entry_points={
        "console_scripts": [
            "llm-finetuning=fine_tuning.trainer:main",
        ],
    },
)
