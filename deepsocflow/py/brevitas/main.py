import os

from deepsocflow.py.brevitas.sim import FixedPointModel
from deepsocflow.py.brevitas.xor import X, Y

if __name__ == '__main__':
    json_path = os.path.join(os.path.dirname(__file__), 'model', 'xor_graph.json')

    # 1. build the model's structure from the already-quantized graph JSON
    #    (shapes, activations, topology - weight/bias arrays not populated yet)
    model = FixedPointModel(json_path)
    print(f"Built model '{json_path}' with {len(model.bundle_order)} bundles")

    # 2. only now pull the quantized int weight/bias values in, from the same JSON
    model.load_int_weights(json_path)
    print("Loaded int weights")

    x_int = model.quantize_input(X)
    logits_int = model.forward(x_int)  # pure int64 arithmetic from here on
    preds = logits_int.argmax(axis=-1)  # softmax is monotonic - argmax unaffected

    print()
    print("x_int:", x_int.tolist())
    print("logits_int:", logits_int.tolist())
    print("targets:    ", Y.tolist())
    print("predictions:", preds.tolist())

    print()
    print("Model graph:")
    model.print_graph()
