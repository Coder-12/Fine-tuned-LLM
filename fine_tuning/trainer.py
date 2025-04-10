import os
import torch
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import yaml
from pathlib import Path
from .data_loader import load_data
from .model import LLMModel


class FinanceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Combine prompt and response for training (this is a simplified example)
        return {"text": item["prompt"] + "\n" + item["response"]}


def collate_fn(batch, tokenizer, max_length=512):
    texts = [item["text"] for item in batch]
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")


def train():
    # Load configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    training_data = load_data(config['data']['training_data_path'])
    dataset = FinanceDataset(training_data)

    # Initialize the model and tokenizer
    llm = LLMModel("config.yaml")
    tokenizer = llm.tokenizer
    model = llm.model

    dataloader = DataLoader(dataset, batch_size=config['model']['batch_size'], shuffle=True,
                            collate_fn=lambda x: collate_fn(x, tokenizer))

    optimizer = AdamW(model.parameters(), lr=config['model']['learning_rate'])
    num_epochs = config['model']['epochs']
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(num_epochs):
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(llm.device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Save model after training
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")


def main():
    train()


if __name__ == "__main__":
    main()
