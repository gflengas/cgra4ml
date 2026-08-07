import os

import torch
import torch.nn as nn

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xor.pt")


class XOR(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_1 = nn.Linear(2, 3, bias=False)
        self.hidden_2 = nn.Linear(3, 3, bias=False)
        self.out = nn.Linear(3, 2, bias=False)
        
        self.silu = nn.SiLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.silu(x)
        x = self.hidden_2(x)
        x = self.silu(x)
        x = self.out(x)
        x = self.softmax(x)
        return x

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([0, 1, 1, 0])  # class index: x1 XOR x2


def train(model, x=X, y=Y, epochs=20000, lr=0.001, seed=0, patience=500):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    best_loss = float("inf")
    epochs_without_improvement = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(torch.log(out.clamp_min(1e-9)), y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch + 1:5d}  loss {loss.item():.4f}")

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"epoch {epoch + 1:5d}  loss {loss.item():.4f}  "
                      f"(early stop: no improvement for {patience} epochs)")
                break

    return model


model = XOR()

if __name__ == "__main__":
    train(model)

    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
    print("targets:    ", Y.tolist())
    print("predictions:", preds.tolist())

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")
