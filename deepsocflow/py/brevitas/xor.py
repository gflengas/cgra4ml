import os

import torch
import torch.nn as nn

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xor.pt")
QONNX_PATH = os.path.join(MODEL_DIR, "xor.onnx")
GRAPH_JSON_PATH = os.path.join(MODEL_DIR, "xor_graph.json")


class XOR(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_1 = nn.Linear(2, 3, bias=True)
        self.relu_1 = nn.ReLU()
        self.hidden_2 = nn.Linear(3, 3, bias=True)
        self.relu_2 = nn.ReLU()
        self.out = nn.Linear(3, 2, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.hidden_1(x)
        x = self.relu_1(x)
        x = self.hidden_2(x)
        x = self.relu_2(x)
        x = self.out(x)
        x = self.softmax(x)
        return x

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([0, 1, 1, 0])  # class index: x1 XOR x2

# Swept seeds 0-79 (each trained to convergence, then quantized through
# quantized_model) and picked the one with the largest post-quantization
# margin - the softmax probability assigned to the correct class stays ~0.9995
# above the decision boundary on all 4 rows, vs. many seeds landing in a
# dead-ReLU local optimum ([0,0,0,0]-style degenerate output). Must be set
# before XOR() is constructed: nn.Linear draws its initial weights at
# construction time, not inside train().
DEFAULT_SEED = 30


def train(model, x=X, y=Y, epochs=20000, lr=0.001, seed=DEFAULT_SEED, patience=500):
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


def load(model, path=MODEL_PATH):
    model.load_state_dict(torch.load(path))
    return model

if __name__ == "__main__":
    from deepsocflow.py.brevitas.quantization.ptq import quantized_model

    torch.manual_seed(DEFAULT_SEED)  # must run before XOR() draws its initial weights
    model = XOR()

    train(model)

    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
    print("targets:    ", Y.tolist())
    print("predictions:", preds.tolist())

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")

    model = load(model, path=MODEL_PATH)
    
    layer_bits = {
        'hidden_1': {'weight_bits': 8, 'bias_bits': 16},
        'hidden_2': {'weight_bits': 8, 'bias_bits': 16},
        'out': {'weight_bits': 8, 'bias_bits': 16},
    }
    
    qm = quantized_model(model, layer_bits=layer_bits)
    qm.quantization(X)

    qm.eval()
    with torch.no_grad():
        qm_out = qm(X)
        qm_preds = qm_out.argmax(dim=-1)
    print("qm predictions:", qm_preds.tolist())
    print("qm output:", qm_out.tolist())

    qm.export(X, QONNX_PATH)
    print(f"exported qonnx to {QONNX_PATH}")

    qm.export_graph_json(X, GRAPH_JSON_PATH)
    print(f"exported model graph json to {GRAPH_JSON_PATH}")
