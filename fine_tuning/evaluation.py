import yaml
from fine_tuning.model import LLMModel


def evaluate():
    # Load configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load the fine-tuned model
    finetuned_model = LLMModel("config.yaml")

    # Example evaluation prompts
    prompts = [
        "Explain the impact of interest rate changes on bond prices.",
        "What are the main risks in option trading?"
    ]

    for prompt in prompts:
        output = finetuned_model.generate(prompt, max_length=100)
        print(f"Prompt: {prompt}")
        print(f"Generated Output: {output}\n")

    # In production, you could compute additional metrics like BLEU, ROUGE, or perplexity.


if __name__ == "__main__":
    evaluate()
