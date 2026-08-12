import os

import numpy as np


def clog2(x):
    return int(np.ceil(np.log2(x)))


def check_hardware(model, hw):
    """Asserts every bundle's tensors FIT the hardware's configured bit-widths.
    input_bits/act_bits use <=, not ==: a bundle can legitimately declare fewer
    bits than hw.X_BITS (e.g. a ReLU activation narrowed to bits-1 so its
    non-negative output still fits the signed datapath - see ptq.py's
    _quantize_activation) without that being a hardware mismatch.
    weight.bits/bias.bits validate against hw.K_BITS/hw.B_BITS too:
    adapter.py's build_bundles labels every weight/bias tensor with
    bits=hw.K_BITS/hw.B_BITS regardless of the JSON's real bit-width (it has
    no other source for the hardware's configured width), so a mismatch there
    doesn't raise on its own - it silently mislabels or truncates the legacy
    XTensor/export path's engine-layout `.bin` blobs. weight.bits must equal
    hw.K_BITS (weight bit-width is a single hardware-wide packing width, not
    per-layer - mirrors deepsocflow/py/xmodel.py:56's `hw.K_BITS ==
    sys_bits.k`); bias.bits only needs to fit within hw.B_BITS (mirrors
    xmodel.py:57's `hw.B_BITS >= sys_bits.b`) since bias storage just needs
    enough headroom, not an exact width. Mirrors deepsocflow/py/xmodel.py:55-57
    and xbundle.py:148-161. Must be called after model.forward() has populated
    model.trace, since the activation-range check inspects real computed
    values, not just declared bits."""
    for name in model.bundle_order:
        bundle = model.bundles[name]

        assert bundle['input_bits'] <= hw.X_BITS, (
            f"bundle '{name}': input_bits={bundle['input_bits']} > hw.X_BITS={hw.X_BITS}")
        assert bundle['act_bits'] <= hw.X_BITS, (
            f"bundle '{name}': act_bits={bundle['act_bits']} > hw.X_BITS={hw.X_BITS}")
        assert bundle['weight_bits'] == hw.K_BITS, (
            f"bundle '{name}': weight.bits={bundle['weight_bits']} != hw.K_BITS={hw.K_BITS}")
        if bundle['bias_bits'] is not None:
            assert bundle['bias_bits'] <= hw.B_BITS, (
                f"bundle '{name}': bias.bits={bundle['bias_bits']} > hw.B_BITS={hw.B_BITS}")

        # ACC_WIDTH bound - Phase 1 uses the bundle's real in_features (CI) as the
        # channel count, unlike the legacy backend's RAM_WEIGHTS_DEPTH-derived r.CM
        # (which pads to the hardware's max channel depth, not the model's actual
        # shape) - that padding only matters once engine-layout export (Phase 2,
        # not in this plan) is wired in.
        acc_width = hw.K_BITS + hw.X_BITS + clog2(bundle['in_features'])
        assert acc_width <= hw.Y_BITS, (
            f"bundle '{name}': ACC_WIDTH={acc_width} > hw.Y_BITS={hw.Y_BITS}")

        # LUT activations (deepsocflow/py/brevitas/lut.py). Same reasoning as the
        # weight/bias checks above: adapter.py hands the table to the legacy
        # exporter, which packs its entries as activation words - so a table whose
        # output is wider than the hardware's activation width silently truncates
        # in the .bin blob instead of raising.
        lut = bundle.get('lut')
        if lut is not None:
            assert lut.out_bits <= hw.X_BITS, (
                f"bundle '{name}': LUT out_bits={lut.out_bits} > hw.X_BITS={hw.X_BITS} "
                f"- table entries are stored as packed activation words")
            # A wider index is not a hardware limit but a sanity bound: 2**16
            # entries is 64 KB per activation, far past anything intended to ship,
            # and almost certainly means act_input_bits was set wrong.
            assert lut.in_bits <= 16, (
                f"bundle '{name}': LUT in_bits={lut.in_bits} needs a "
                f"{2 ** lut.in_bits} entry table ({lut.nbytes} B) - check act_input_bits")
            # Raised inside ActLut too, but repeated here so the failure names the
            # bundle rather than just the fracs.
            acc_frac = bundle['input_frac'] + bundle['weight_frac']
            assert acc_frac >= lut.in_frac, (
                f"bundle '{name}': acc_frac={acc_frac} < LUT in_frac={lut.in_frac} "
                f"- the index rescale must be a right shift, never a multiply")

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


def export_rtl(model, hw, x_float, batch_size=4):
    """Exports everything the RTL testbench consumes - engine-layout text files,
    packed .bin blobs, and config_fw.h - by adapting this model's bundles onto
    the legacy XBundle surface and reusing the legacy export path.

    Unlike export_inference (which writes layout-independent golden reference
    text and is left untouched), this drives the real hardware file format.

    config_fw.h is written to the CURRENT WORKING DIRECTORY by the legacy
    exporter (xmodel.py), not into hw.DATA_DIR - run this from the directory
    where the firmware build expects it."""
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.rtl_export import _export_bundles

    x_int = model.quantize_input(np.asarray(x_float)[:batch_size])
    model.forward(x_int)

    check_hardware(model, hw)
    build_bundles(model, hw)

    os.makedirs(hw.DATA_DIR, exist_ok=True)
    for entry in os.scandir(hw.DATA_DIR):
        os.remove(entry.path)

    _export_bundles(hw, None)  # x=None: the adapter's call_int is a no-op

    files = sorted(entry.path for entry in os.scandir(hw.DATA_DIR))
    return {'files': files}
