import os

import torch

from deepsocflow.py.brevitas.utils import Model

if __name__ == '__main__':
    json_path = os.path.join(os.path.dirname(__file__), 'models', 'xor', 'xor.json')

    model = Model(json_path=json_path)
    model.eval()
    print(f"Built model '{json_path}' with {len(model.layers)} layers")

    x = torch.randn(4, 2)
    with torch.no_grad():
        out = model(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Output row sums (softmax check, should be ~1.0): {out.sum(dim=-1).tolist()}")

    print()
    print("Model graph:")
    model.print_graph()
