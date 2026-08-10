import os

import numpy as np


def clog2(x):
    return int(np.ceil(np.log2(x)))


def check_hardware(model, hw):
    """Asserts every bundle's tensors FIT the hardware's configured bit-widths
    (<=, not ==): a bundle can legitimately declare fewer bits than hw.X_BITS
    (e.g. a ReLU activation narrowed to bits-1 so its non-negative output still
    fits the signed datapath - see ptq.py's _quantize_activation) without that
    being a hardware mismatch. Mirrors deepsocflow/py/xmodel.py:55-57 and
    xbundle.py:148-161. Must be called after model.forward() has populated
    model.trace, since the activation-range check inspects real computed
    values, not just declared bits."""
    for name in model.bundle_order:
        bundle = model.bundles[name]

        assert bundle['input_bits'] <= hw.X_BITS, (
            f"bundle '{name}': input_bits={bundle['input_bits']} > hw.X_BITS={hw.X_BITS}")
        assert bundle['act_bits'] <= hw.X_BITS, (
            f"bundle '{name}': act_bits={bundle['act_bits']} > hw.X_BITS={hw.X_BITS}")

        # ACC_WIDTH bound - Phase 1 uses the bundle's real in_features (CI) as the
        # channel count, unlike the legacy backend's RAM_WEIGHTS_DEPTH-derived r.CM
        # (which pads to the hardware's max channel depth, not the model's actual
        # shape) - that padding only matters once engine-layout export (Phase 2,
        # not in this plan) is wired in.
        acc_width = hw.K_BITS + hw.X_BITS + clog2(bundle['in_features'])
        assert acc_width <= hw.Y_BITS, (
            f"bundle '{name}': ACC_WIDTH={acc_width} > hw.Y_BITS={hw.Y_BITS}")

        if name in getattr(model, 'trace', {}):
            out = model.trace[name]['out']
            if bundle['act_signed']:
                lo, hi = -2 ** (hw.X_BITS - 1), 2 ** (hw.X_BITS - 1) - 1
            else:
                lo, hi = 0, 2 ** hw.X_BITS - 1
            assert out.min() >= lo and out.max() <= hi, (
                f"bundle '{name}': activation values [{out.min()},{out.max()}] "
                f"outside signed={bundle['act_signed']} hw.X_BITS={hw.X_BITS} range [{lo},{hi}]")


def _savetxt_int(path, arr):
    np.savetxt(path, np.asarray(arr).flatten(), fmt='%d')


def _savetxt_float(path, arr):
    np.savetxt(path, np.asarray(arr).flatten(), fmt='%f')


def _to_nhwc(out_2d):
    """(XN, CO) -> (1, XN, 1, CO), matching the dense-as-1x1-conv reshape at
    deepsocflow/py/xbundle.py:125-128."""
    xn, co = out_2d.shape
    return out_2d.reshape(1, xn, 1, co)


def export_inference(model, hw, x_float, data_dir=None, clean=True, batch_size=1):
    """Runs model.forward() on x_float[:batch_size] and writes the
    layout-independent golden-reference files legacy's xmodel.py::export_inference
    produces (y_exp.txt, {ib}_y_nhwc_exp.txt) into hw.DATA_DIR - same filenames,
    same np.savetxt formats, same flatten order, so a future RTL step can
    consume them unchanged. Engine-layout files ({ib}_{ip}_{it}_*, .bin blobs)
    are not produced here - see the design spec's Phase 2."""
    data_dir = data_dir or hw.DATA_DIR

    if clean:
        os.makedirs(data_dir, exist_ok=True)
        for entry in os.scandir(data_dir):
            os.remove(entry.path)
    else:
        os.makedirs(data_dir, exist_ok=True)

    x_batch = np.asarray(x_float)[:batch_size]
    x_int = model.quantize_input(x_batch)
    model.forward(x_int)

    check_hardware(model, hw)

    files = []

    last_name = model.bundle_order[-1]
    last_bundle = model.bundles[last_name]
    if last_bundle['softmax']:
        y_exp = model.softmax_out
        y_exp_path = os.path.join(data_dir, "y_exp.txt")
        _savetxt_float(y_exp_path, y_exp)
    else:
        y_exp = model.outputs[last_name]
        y_exp_path = os.path.join(data_dir, "y_exp.txt")
        _savetxt_int(y_exp_path, y_exp)
    files.append(y_exp_path)

    flat = y_exp.flatten()
    n = len(flat)
    for i in range(n):
        if i < 20 or n - i <= 20:
            print(f"y_exp {i}: {flat[i]}")

    for ib, name in enumerate(model.bundle_order):
        out = model.trace[name]['out']
        nhwc = _to_nhwc(out)
        path = os.path.join(data_dir, f"{ib}_y_nhwc_exp.txt")
        _savetxt_int(path, nhwc)
        files.append(path)

    print(f"Weights, inputs, outputs saved to {data_dir}/")

    return {
        'files': files,
        'y_exp': y_exp,
        'softmax_frac': model.softmax_frac,
        'softmax_max_i': model.softmax_max_i,
    }
