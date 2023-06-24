from tokenization import Tokenizer
import torch
import torch.nn as nn
from sublayers import GPT
from config import GPTConfig
from torch import Tensor
from torch.nn import functional as F


class GPTTrainer(nn.Module):
    def __init__(self, config=GPTConfig(), n_classes=10) -> None:
        super(GPTTrainer, self).__init__()

        self.config = config
        self.tokenizer = Tokenizer(model_path=config.model_path)
        self.model = GPT(config=config, n_classes=n_classes)

    def forward(self, input):
        return self.model(input)

    def generate(
        self,
        prompt: str,
        max_len: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> str:
        tokens = self.tokenizer.encode(prompt)
        tokens = tokens.unsqueeze(0)

        for i in range(max_len):
            logits = self.forward(tokens)
            logits = logits[:, -1, :] / temperature

            filtered_logits = self.top_k_top_p_filtering(
                logits, top_k=top_k, top_p=top_p
            )

            probabilities = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probabilities, num_samples=1).squeeze(1)

            tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1)

            if next_token == self.tokenizer.eos_id:
                break

        return self.tokenizer.decode(tokens.squeeze(0).tolist())

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    @staticmethod
    def top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        filter_value: float = -float("Inf"),
    ) -> torch.Tensor:
        assert logits.dim() == 1
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p

            if 0 in sorted_indices_to_remove:
                sorted_indices_to_remove = sorted_indices_to_remove[0]
            else:
                sorted_indices_to_remove = sorted_indices_to_remove[-1]

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def train(
        self,
        train_loader,
        val_loader,
        epochs,
        device,
        lr=1e-4,
        warmup_steps=10000,
        weight_decay=0.01,
        accumulation_steps=16,
        print_every=100,
    ):
        self.to(device)
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
        )
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}")
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                with torch.cuda.amp.autocast():
                    outputs = self(input_ids)
                    logits = outputs.logits
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if i % print_every == 0:
                    print(f"Step: {i} Loss: {loss.item()}")
            self.save("gpt.pth")
            self.eval()
            total = 0
            correct = 0
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                with torch.cuda.amp.autocast():
                    outputs = self(input_ids)
                    logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
            print(f"Validation Accuracy: {(correct / total) * 100}%")
            self.train()


if __name__ == "__main__":
    #train_loader, val_loader = get_loaders()
    #model = GPTTrainer()
    #model.train(train_loader, val_loader, epochs=5, device="cuda")