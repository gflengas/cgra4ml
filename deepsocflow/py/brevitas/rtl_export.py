"""Numpy-only port of the legacy engine-layout exporter (deepsocflow/py/xbundle.py
::XBundle.export and deepsocflow/py/xmodel.py::_export_bundles/verify_inference),
for the brevitas backend.

Why a fork instead of reuse: deepsocflow/py/brevitas/adapter.py used to call the
legacy functions directly (BrevitasBundle.export() called XBundle.export(self, ...)
as an unbound method, and export.py::export_rtl imported xmodel.py::_export_bundles
directly) - deliberately, to reuse RTL-verified bit-packing logic rather than
re-derive it (see CLAUDE.md's 2026-08-11 "brevitas -> RTL end-to-end" entry). But
importing xbundle.py/xmodel.py at all requires tensorflow/qkeras (module-level
imports), even though the code paths brevitas actually uses (this file's three
functions) touch TF in exactly two spots: one tf.keras.backend.conv2d call
(replaced below by _conv2d_same) and a handful of `.numpy()` calls on XTensor
fields (dropped - deepsocflow/py/brevitas/xtensor.py's XTensor already holds
numpy arrays). Everything else - all the reorder_*_q2e_conv/get_runtime_params/
create_headers hardware-layout math in deepsocflow/py/dataflow.py - was already
pure numpy; only its import chain went through the TF-heavy utils.py, fixed
separately (deepsocflow/py/dataflow.py now imports deepsocflow/py/numeric.py).

Ported logic must stay bit-for-bit identical to the legacy originals - this
writes the exact file format the RTL testbench and PYNQ driver consume."""
import json

import numpy as np

from deepsocflow.py.numeric import BUNDLES, clog2
from deepsocflow.py.dataflow import (
    get_runtime_params, create_headers, check_sparsity,
    reorder_b_q2e_conv, reorder_w_q2e_conv, reorder_x_q2e_conv, reorder_y_q2e_conv,
    pack_words_into_bytes, predict_model_performance,
)


def _conv2d_same(x, w):
    """(N,H,W,Ci) x (KH,KW,Ci,Co) -> (N,H,W,Co), stride (1,1), matching
    tf.keras.backend.conv2d(x, w, padding='same') for stride (1,1) - the only
    stride xbundle.py's export() ever calls it with (dense-as-1x1-conv here,
    and 'same'-padding conv striding is handled separately, before this call,
    by dataflow.py's own CSH/CSW logic - this only ever sees the un-strided
    per-pass slices).

    TF's SAME padding for stride 1: pad_total = kernel_size - 1, split with
    the extra pixel (if odd) going to the bottom/right - matches np.pad below.
    At the 1x1 kernel size the brevitas adapter actually produces today
    (dense-only, see CLAUDE.md's Known Issues on conv support), this reduces
    to a per-pixel channel matmul with no padding at all."""
    N, H, W, Ci = x.shape
    KH, KW, _, Co = w.shape
    pad_h, pad_w = KH - 1, KW - 1
    pad_top, pad_left = pad_h // 2, pad_w // 2
    pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left
    xp = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
    out = np.zeros((N, H, W, Co), dtype=np.float32)
    for kh in range(KH):
        for kw in range(KW):
            out += np.einsum('nhwc,cd->nhwd', xp[:, kh:kh+H, kw:kw+W, :], w[kh, kw])
    return out


def export_bundle(bundle, hw, is_last):  # hw: Hardware config, is_last: True if this is the final bundle in the network
    b = bundle

    if not b.core.type == 'conv':
        print('Conv -> Dense Reshape')
        CI,CO = b.core.w.itensor.shape  # input/output channels (dense treated as a 1x1 conv)
        XN, _ = b.core.x.itensor.shape  # input batch size
        w_int = b.core.w.itensor.reshape(1,1,CI,CO) # (CI,CO) -> (KH,KW,CI,CO)
        x_int = b.core.x.itensor.reshape(1,XN,1,CI) # (XN,CI) -> (XN, XH, XW, CI)
        y_int = b.core.y.itensor.reshape(1,XN,1,CO) # (XN,CI) -> (XN, XH, XW, CI)
        o_int = (b.pre_softmax if b.softmax else b.out).itensor.reshape(1,XN,1,CO)
    else:
        w_int = b.core.w.itensor
        x_int = b.core.x.itensor
        y_int = b.core.y.itensor
        o_int = (b.pre_softmax if b.softmax else b.out).itensor

    b_int = b.core.b.itensor if b.core.b else None
    # w/x/y/b/o _int: integer (quantized) tensors for weights, input, conv-sum, bias, and (bundle) output
    r = get_runtime_params(
        hw=hw,
        w_shape=w_int.shape,
        x_shape=x_int.shape,
        o_shape=b.out.ftensor.shape,
        core=b.core,
        pool=b.pool,
        flatten = b.flatten,
        )
    r = create_headers(hw, r)

    assert r.KH <= hw.KH_MAX
    assert r.KW <= hw.KW_MAX
    assert r.CM <= hw.CI_MAX
    assert r.XH <= hw.XH_MAX
    assert r.XW <= hw.XW_MAX
    assert r.XN <= hw.XN_MAX

    cm_max = r.CM_0 if r.CP==1 else r.CM
    EDGES = cm_max * r.XW #* int(np.ceil(r.XH/hw.ROWS)-1)
    assert EDGES <= hw.RAM_EDGES_DEPTH or r.KH == 1, f"Edges: {EDGES} < {hw.RAM_EDGES_DEPTH}"

    assert r.XW >= r.KH//2
    ACC_WIDTH = hw.K_BITS + hw.X_BITS + clog2(r.KH*r.KW*r.CM)
    assert ACC_WIDTH <= hw.Y_BITS, f"ACC_WIDTH:{ACC_WIDTH} > Y_BITS{hw.Y_BITS}"

    print(r)
    check_sparsity(w_int, x_int)

    # b/w/x/y "e" suffix: tensor reordered into hardware engine layout (see reorder_*_q2e_conv)
    b.be = reorder_b_q2e_conv(b_int, hw, r) if b.core.b else None
    b.we = reorder_w_q2e_conv(w_int, hw, r)
    b.ye_exp_shape = (r.IT, r.XN, r.XL, r.XW*r.CO_PRL, hw.ROWS)
    b.ye_hw = np.zeros(b.ye_exp_shape)

    b.xe = reorder_x_q2e_conv(x_int, hw, r)
    b.ye_exp = reorder_y_q2e_conv(y_int, hw, r)  # expected engine-layout conv-sum
    b.o_int = o_int
    b.oe_sum_exp = o_int if is_last else reorder_y_q2e_conv(y_int, hw, r)  # expected summed output (engine layout)
    b.oe_exp_nhwc = o_int  # expected output in N,H,W,C layout
    print(f"x reshape: [int]:{b.core.x.itensor.shape}, int:{x_int.shape}. xe:{b.xe[0].shape}")

    '''
    Prepare expected outputs for each pass
    '''
    b.ye_exp_p = []  # ye_exp per pass (p)
    ic_left = ic_right = 0
    for ip in range(r.CP):
        CM_p = r.CM_0 if ip==0 else r.CM
        ic_right += CM_p

        wp = w_int[:,:, ic_left:ic_right, :]  # weight slice (w) for this pass (p)
        xp = x_int[:,:,:, ic_left:ic_right ]  # input slice (x) for this pass (p)
        yp = _conv2d_same(xp.astype(np.float32), wp.astype(np.float32)).astype(np.int32)  # conv-sum (y) for this pass (p)
        b.ye_exp_p += [reorder_y_q2e_conv(yp, hw, r)]
        ic_left = ic_right
    b.hw, b.r = hw, r


def _export_bundles(hw, x):
    """Bundle loop shared by both backends. Assumes BUNDLES is already populated
    and, for the brevitas backend, that each bundle's integer tensors are already
    computed (its call_int is a no-op). `x` is the input XTensor consumed by
    bundle 0's call_int; the brevitas backend passes None."""
    add_buffer_map = []
    out_buffer_map = []

    for ib, b in enumerate(BUNDLES):
        print(f'-----------------ib:{ib}-----------------------')
        b.call_int(x if ib==0 else None, hw)
        b.export(hw, False)

        '''
        OUTPUT BUFFER ALLOCATION
        '''
        print(f'input_out_map:{out_buffer_map}')

        '''Find and assign a free buffer. If not, add new buffer'''
        b.out_buffer_idx = -1
        next_ibs = sorted(list(b.next_ibs))  # next_ibs: bundle indices (ib) that consume this bundle's output
        if len(next_ibs) != 0:
            for im in range(len(out_buffer_map)):  # im: index of a buffer slot in the map
                if out_buffer_map[im] is None:
                    out_buffer_map[im] = {'in':b.ib, 'out':next_ibs}
                    b.out_buffer_idx = im
                    break
            else: #m if break is not hit
                b.out_buffer_idx = len(out_buffer_map)
                out_buffer_map += [{'in':b.ib, 'out':next_ibs}]

        print('out_buffer_idx:', b.out_buffer_idx)

        '''Free the buffers whose last destination is current bundle'''
        for im in range(len(out_buffer_map)):
            buf = out_buffer_map[im]
            if buf is not None:
                if buf['out'][-1] == b.ib:
                    out_buffer_map[im] = None

        print(f'out_buffer_map:{out_buffer_map}')



        '''
        ADD BUFFER ALLOCATION
        '''
        print(f'input_add_map:{add_buffer_map}')

        '''Find and assign a free buffer. If not, add new buffer'''
        b.add_out_buffer_idx = -1
        # sorted() for the same reason next_ibs is sorted above: the free step
        # below reads buf['out'][-1] as "the last consumer", so the order has to
        # be ascending regardless of what the caller handed us.
        next_add_ibs = sorted(b.next_add_ibs)
        if len(next_add_ibs) != 0:
            for im in range(len(add_buffer_map)):
                if add_buffer_map[im] is None:
                    add_buffer_map[im] = {'in':b.ib, 'out':next_add_ibs}
                    b.add_out_buffer_idx = im
                    break
            else: #m if break is not hit
                b.add_out_buffer_idx = len(add_buffer_map)
                add_buffer_map += [{'in':b.ib, 'out':next_add_ibs}]

        print('add_out_buffer_idx:', b.add_out_buffer_idx)

        '''Free the buffers whose last destination is current bundle'''
        for im in range(len(add_buffer_map)):
            buf = add_buffer_map[im]
            if buf is not None:
                if buf['out'][-1] == b.ib:
                    add_buffer_map[im] = None

        print(f'add_buffer_map:{add_buffer_map}')


    d_perf = predict_model_performance(hw=hw)
    print(f"Predicted performance: {d_perf}")

    '''
    Write Runtime Headers
    '''
    # Built alongside config_fw.h below, from the exact same per-bundle values -
    # config.json is the PYNQ driver's JSON mirror of config_fw.h's #define
    # macros ("defines") and Bundle_t struct array ("bundles"), field-for-field
    # (see deepsocflow/c/runtime.h's Bundle_t). Building it in the same loop
    # (rather than a second pass over BUNDLES) avoids the two ever drifting
    # apart - it reads the same locals the ch.write(...) calls below use.
    bundles_json = []

    # Activation LUTs (deepsocflow/py/brevitas/lut.py). Collected before the
    # bundle loop because the table array has to be emitted ahead of the
    # Bundle_t initializers that index into it. Identical tables are shared: two
    # bundles with the same activation on the same grid produce the same bytes,
    # and duplicating them would waste the config_fw.h space this design is
    # chosen for. Bundles on the quant_lrelu path get index -1.
    lut_tables = []          # unique tables, in emission order
    lut_meta = []            # (in_bits,) per unique table
    lut_idx_of_ib = {}
    for b in BUNDLES:
        lut = getattr(b.core.act, 'lut', None)
        if lut is None:
            lut_idx_of_ib[b.ib] = (-1, 0)
            continue
        key = (lut.activation, lut.in_bits, lut.in_frac, lut.out_bits, lut.out_frac,
               lut.out_signed, tuple(int(v) for v in lut.table))
        for i, (existing_key, _) in enumerate(lut_tables):
            if existing_key == key:
                lut_idx_of_ib[b.ib] = (i, lut.in_bits)
                break
        else:
            lut_tables.append((key, lut))
            lut_meta.append(lut.in_bits)
            lut_idx_of_ib[b.ib] = (len(lut_tables) - 1, lut.in_bits)

    # Tables of different widths are padded to a common stride so LUTS stays a
    # plain 2-D array in C. ca_lut_bits tells the firmware how much of each row
    # is real, so the padding is never addressed.
    lut_entries = max((2 ** bits for bits in lut_meta), default=0)

    x_bytes_all = x_bytes = w_bytes = b_words = x_bytes_max = nhwc_words_max = o_bytes_max = o_words_max = 0
    with open (f'./config_fw.h', 'w') as ch:

        ch.write(f"#define N_BUNDLES {len(BUNDLES)}\n")
        ch.write(f"#define N_LUTS      {len(lut_tables)}\n")
        ch.write(f"#define LUT_ENTRIES {lut_entries}\n")
        if lut_tables:
            ch.write(f"static const i8 LUTS [N_LUTS][LUT_ENTRIES] = {{\n")
            for _, lut in lut_tables:
                padded = list(int(v) for v in lut.table) + [0] * (lut_entries - lut.table.size)
                body = ','.join(f"{v:>4}" for v in padded)
                ch.write(f"  /* {lut.activation} {lut.in_bits}b/frac{lut.in_frac} -> "
                         f"{lut.out_bits}b/frac{lut.out_frac} */\n  {{{body}}},\n")
            ch.write("};\n")
        ch.write("\n")
        ch.write(f"Bundle_t bundles [N_BUNDLES] = {{\n")

        # Naming below: _bpt = bytes per transfer, _b suffix = value for the current bundle,
        # ca_/aa_/pa_ prefixes = core/residual-add/pool activation params (nzero/shift/pl_scale, see XActivation)
        for ib, b in enumerate(BUNDLES):
            assert ib == b.ib

            w_bpt    = (hw.K_BITS*b.we[-1][0].size)//8  # weight bytes-per-transfer (last pass)
            w_bpt_p0 = (hw.K_BITS*b.we[0][0].size)//8    # weight bytes-per-transfer (pass 0)
            x_bpt    = (hw.X_BITS*b.xe[-1].size)//8      # input bytes-per-transfer (last pass)
            x_bpt_p0 = (hw.X_BITS*b.xe[0].size )//8      # input bytes-per-transfer (pass 0)

            if ib == len(BUNDLES)-1:
                o_words_b = b.o_int.size
                o_bytes_b = o_words_b*4 # int or float
                o_words = o_words_b
            else:
                b_next    = BUNDLES[ib+1]
                o_wpt     = b_next.xe[-1].size    # output words-per-transfer, i.e. next bundle's input (last pass)
                o_wpt_p0  = b_next.xe[0].size     # output words-per-transfer, i.e. next bundle's input (pass 0)
                o_words_b = o_wpt_p0 + (b_next.r.CP-1)*o_wpt

                o_bpt = (hw.X_BITS*b_next.xe[-1].size)//8    # output bytes-per-transfer (last pass)
                o_bpt_p0 = (hw.X_BITS*b_next.xe[0].size)//8  # output bytes-per-transfer (pass 0)
                o_bytes_b = o_bpt_p0 + (b_next.r.CP-1)*o_bpt

            xp_words  = b.r.XN * b.r.XL * b.r.XW * (hw.ROWS+b.r.X_PAD)  # input words per pass (p)

            w_bytes_b = (w_bpt_p0 + (b.r.CP-1)*w_bpt)*b.r.IT
            x_bytes_b = (x_bpt_p0 + (b.r.CP-1)*x_bpt)
            nhwc_words_b = b.r.XN * b.r.XH * b.r.XW * b.r.CO  # output words in N,H,W,C layout

            x_bytes_max = max(x_bytes_max, x_bytes_b)
            nhwc_words_max = max(nhwc_words_max, nhwc_words_b)
            o_bytes_max = max(o_bytes_max, o_bytes_b)
            o_words_max = max(o_words_max, o_words_b)
            w_bytes += w_bytes_b
            x_bytes_all += x_bytes_b

            ib_out = -1 if len(b.next_ibs) == 0 else sorted(b.next_ibs)[0]  # bundle index (ib) of consumer, or -1 if none

            if ib == 0:
                x_bytes = (x_bpt_p0 + (b.r.CP-1)*x_bpt)

            y_coe = b.r.CO_PRL  # output channels processed in parallel per iteration
            y_coe_tl = b.r.CO_PRL if (b.r.CO==b.r.IT*b.r.CO_PRL) else b.r.CO%b.r.IT  # coe count in the tail (last) iteration
            y_r_ll = hw.ROWS if b.r.XH==b.r.XL*hw.ROWS else  b.r.XH % hw.ROWS        # row count in the last (ll) row-block

            ca_nzero, ca_shift, ca_pl_scale = b.core.act.non_zero, b.core.act.shift_bits, b.core.act.plog_slope  # core (conv/dense) activation params
            # On a LUT bundle ca_shift lands the accumulator on the TABLE'S INDEX
            # grid, not on the activation's output grid (adapter.py sets it that
            # way); ca_nzero/ca_pl_scale are unused there.
            ca_lut_idx, ca_lut_bits = lut_idx_of_ib[b.ib]

            (aa_nzero, aa_shift, aa_pl_scale) = (b.add .act.non_zero, b.add .act.shift_bits, b.add .act.plog_slope)if b.add  is not None else (0,0,0)  # residual-add activation params
            (pa_nzero, pa_shift, pa_pl_scale) = (b.pool.act.non_zero, b.pool.act.shift_bits, b.pool.act.plog_slope)if b.pool is not None else (0,0,0)  # pool activation params

            add_out_buffer_idx = b.add_out_buffer_idx
            add_in_buffer_idx = BUNDLES[b.add.source_ib].add_out_buffer_idx if b.add is not None else -1  # buffer holding this bundle's residual/skip input
            in_buffer_idx = BUNDLES[b.prev_ib].out_buffer_idx if b.prev_ib is not None else -1

            if b.pool is None:
                pool_type = 'POOL_NONE'
            elif b.pool.type == 'max':
                pool_type = 'POOL_MAX'
            elif b.pool.type == 'avg':
                pool_type = 'POOL_AVG'

            out_type = 'float' if (ib == len(BUNDLES)-1 and b.softmax) else 'int32_t'

            ch.write(f"   {{.n={b.r.XN:<3}, .l={b.r.XL:<3}, .kw={b.r.KW:<3}, .coe={y_coe:<3}, .h={b.r.XH:<3}, .w={b.r.XW:<3}, .ci={b.r.CI:<4}, .co={b.r.CO:<4}, .w_kw2={b.r.XW-b.r.KW//2:<3}, .t={b.r.IT:<3}, .p={b.r.CP:<3}, .cm={b.r.CM:<3}, .cm_p0={b.r.CM_0:<3}, .on={b.r.ON:<3}, .oh={b.r.OH:<3}, .ow={b.r.OW:<3}, .oc={b.r.OC:<4}, .ch={b.r.CYH:<3}, .ph={b.r.PYH:<3}, .cw={b.r.CYW:<3}, .pw={b.r.PYW:<3}, .pkh={b.r.PKH:<3}, .psh={b.r.PSH:<3}, .pkw={b.r.PKW:<3}, .psw={b.r.PSW:<3}, ")
            ch.write(     f".xp_words={xp_words:<6}, .b_offset={b_words:<5}, .w_bpt={w_bpt:<5}, .w_bpt_p0={w_bpt_p0:<5}, .x_bpt={x_bpt:<8}, .x_bpt_p0={x_bpt_p0:<8}, .o_words={o_words_b:<8}, .o_bytes={o_bytes_b:<8}, ")
            ch.write(     f".ib_out={ib_out:<4}, .in_buffer_idx={in_buffer_idx:<3}, .out_buffer_idx={b.out_buffer_idx:<3}, .add_out_buffer_idx={add_out_buffer_idx:<2}, .add_in_buffer_idx={add_in_buffer_idx:<2}, ")
            ch.write(     f".is_bias={1*(b.core.b is not None):<3}, .is_flatten={1*(b.flatten is not None):<3}, .is_softmax={1*(b.softmax is not None):<3}, ")
            ch.write(     f".x_pad={b.r.X_PAD:<3}, .b_val_shift={b.core.bias_val_shift:<3}, .b_bias_shift={b.core.bias_b_shift:<3}, .ca_nzero={ca_nzero:<3}, .ca_shift={ca_shift:<3}, .ca_pl_scale={ca_pl_scale:<3}, .aa_nzero={aa_nzero:<3}, .aa_shift={aa_shift:<3}, .aa_pl_scale={aa_pl_scale:<3}, .pa_nzero={pa_nzero:<3}, .pa_shift={pa_shift:<3}, .pa_pl_scale={pa_pl_scale:<3}, .ca_lut_idx={ca_lut_idx:<3}, .ca_lut_bits={ca_lut_bits:<3}, .softmax_frac={b.softmax_frac:<3}, ")
            ch.write(     f".csh={b.r.CSH:<3}, .csh_shift={b.r.CSH_SHIFT:<3}, .psh_shift={b.r.PSH_SHIFT:<3}, .csw={b.r.CSW:<3}, .csw_shift={b.r.CSW_SHIFT:<3}, .psw_shift={b.r.PSW_SHIFT:<3}, .pool={pool_type:<10}, ")
            ch.write(     f".softmax_max_i={b.softmax_max_i:<15}, ")
            ch.write(     f".header={b.r.header:>23}u, ")
            ch.write(     f".debug_nhwc_words={b.oe_exp_nhwc.size:<9} }}")

            bundles_json.append({
                'n': int(b.r.XN), 'l': int(b.r.XL), 'kw': int(b.r.KW), 'coe': int(y_coe),
                'h': int(b.r.XH), 'w': int(b.r.XW), 'ci': int(b.r.CI), 'co': int(b.r.CO),
                'w_kw2': int(b.r.XW-b.r.KW//2), 't': int(b.r.IT), 'p': int(b.r.CP),
                'cm': int(b.r.CM), 'cm_p0': int(b.r.CM_0),
                'on': int(b.r.ON), 'oh': int(b.r.OH), 'ow': int(b.r.OW), 'oc': int(b.r.OC),
                'ch': int(b.r.CYH), 'ph': int(b.r.PYH), 'cw': int(b.r.CYW), 'pw': int(b.r.PYW),
                'pkh': int(b.r.PKH), 'psh': int(b.r.PSH), 'pkw': int(b.r.PKW), 'psw': int(b.r.PSW),
                'xp_words': int(xp_words), 'b_offset': int(b_words),
                'w_bpt': int(w_bpt), 'w_bpt_p0': int(w_bpt_p0),
                'x_bpt': int(x_bpt), 'x_bpt_p0': int(x_bpt_p0),
                'o_words': int(o_words_b), 'o_bytes': int(o_bytes_b),
                'ib_out': int(ib_out), 'in_buffer_idx': int(in_buffer_idx),
                'out_buffer_idx': int(b.out_buffer_idx), 'add_out_buffer_idx': int(add_out_buffer_idx),
                'add_in_buffer_idx': int(add_in_buffer_idx),
                'is_bias': 1*(b.core.b is not None), 'is_flatten': 1*(b.flatten is not None),
                'is_softmax': 1*(b.softmax is not None),
                'x_pad': int(b.r.X_PAD), 'b_val_shift': int(b.core.bias_val_shift),
                'b_bias_shift': int(b.core.bias_b_shift),
                'ca_nzero': int(ca_nzero), 'ca_shift': int(ca_shift), 'ca_pl_scale': int(ca_pl_scale),
                'aa_nzero': int(aa_nzero), 'aa_shift': int(aa_shift), 'aa_pl_scale': int(aa_pl_scale),
                'pa_nzero': int(pa_nzero), 'pa_shift': int(pa_shift), 'pa_pl_scale': int(pa_pl_scale),
                'ca_lut_idx': int(ca_lut_idx), 'ca_lut_bits': int(ca_lut_bits),
                'softmax_frac': int(b.softmax_frac),
                'csh': int(b.r.CSH), 'csh_shift': int(b.r.CSH_SHIFT), 'psh_shift': int(b.r.PSH_SHIFT),
                'csw': int(b.r.CSW), 'csw_shift': int(b.r.CSW_SHIFT), 'psw_shift': int(b.r.PSW_SHIFT),
                'pool': pool_type,
                # config_fw.h's .softmax_max_i is the hardware's 2**17-scaled fixed-point
                # form (deepsocflow/c/runtime.h:400 divides by 1<<17 at read time) - the
                # JSON gives the PYNQ driver the already-divided plain float directly.
                'softmax_max_f': b.softmax_max_i / 2**17,
                'header': int(b.r.header),
                'debug_nhwc_words': int(b.oe_exp_nhwc.size),
            })

            b_words += b.be.size if b.core.b else 0
            if b.ib != len(BUNDLES)-1:
                ch.write(',\n')


        ch.write(f"\n}};\n\n")
        ch.write(f"#define X_BITS_L2   {int(np.log2(hw.X_BITS))}\n")
        ch.write(f"#define W_BITS_L2   {int(np.log2(hw.K_BITS))}\n")
        ch.write(f"#define KH_MAX      {hw.KH_MAX}\n")
        ch.write(f"#define PE_ROWS     {hw.ROWS}\n")
        ch.write(f"#define PE_COLS     {hw.COLS}\n\n")

        ch.write(f"#define N_OUT_BUF   {max(len(out_buffer_map),1)}\n")
        ch.write(f"#define N_ADD_BUF   {len(add_buffer_map) if len(add_buffer_map) > 0 else ''}\n")
        ch.write(f"#define WB_BYTES    {w_bytes + (b_words*hw.B_BITS)//8}\n")
        ch.write(f"#define W_BYTES     {w_bytes}\n")
        ch.write(f"#define X_BYTES     {x_bytes}\n")
        ch.write(f"#define O_WORDS     {o_words}\n")
        ch.write(f"#define O_WORDS_MAX {o_words_max}\n")
        ch.write(f"#define O_BYTES_MAX {o_bytes_max}\n")
        ch.write(f"#define X_BYTES_ALL {x_bytes_all}\n")
        ch.write(f"#define NHWC_WORDS  {nhwc_words_max}\n")
        ch.write(f"#define Y_TYPE      int{hw.Y_OUT_BITS}_t\n")
        ch.write(f"#define B_TYPE      int{hw.B_BITS}_t\n")
        ch.write(f"#define O_TYPE      {out_type}\n")
        ch.write(f"#define B_WORDS     {b_words}\n")
        ch.write(f"#define AXI_WIDTH   {hw.AXI_WIDTH}\n")
        ch.write(f"#define CONFIG_BASEADDR 0x{hw.CONFIG_BASEADDR}\n")
        ch.write(f'#define DATA_DIR   "../{hw.DATA_DIR}"\n\n')

        defines_json = {
            'N_BUNDLES': len(BUNDLES),
            'N_LUTS': len(lut_tables),
            'LUT_ENTRIES': lut_entries,
            'X_BITS_L2': int(np.log2(hw.X_BITS)),
            'W_BITS_L2': int(np.log2(hw.K_BITS)),
            'KH_MAX': hw.KH_MAX,
            'PE_ROWS': hw.ROWS,
            'PE_COLS': hw.COLS,
            'N_OUT_BUF': max(len(out_buffer_map), 1),
            'N_ADD_BUF': len(add_buffer_map),
            'WB_BYTES': w_bytes + (b_words*hw.B_BITS)//8,
            'W_BYTES': w_bytes,
            'X_BYTES': x_bytes,
            'O_WORDS': o_words,
            'O_WORDS_MAX': o_words_max,
            'O_BYTES_MAX': o_bytes_max,
            'X_BYTES_ALL': x_bytes_all,
            'NHWC_WORDS': nhwc_words_max,
            'Y_TYPE_str': f'int{hw.Y_OUT_BITS}',
            'B_TYPE_str': f'int{hw.B_BITS}',
            'O_TYPE_str': 'float32' if out_type == 'float' else 'int32',
            'B_WORDS': b_words,
            'AXI_WIDTH': hw.AXI_WIDTH,
            'CONFIG_BASEADDR': str(hw.CONFIG_BASEADDR),
            'DATA_DIR': hw.DATA_DIR,
        }
        # 'luts' mirrors config_fw.h's LUTS array exactly - same rows, same
        # padding, same order - so pynq_driver.py indexes it with the very
        # ca_lut_idx the firmware uses. Built from the same lut_tables list the
        # header was written from, for the same no-drift reason as 'bundles'.
        luts_json = [
            {'activation': lut.activation,
             'in_bits': int(lut.in_bits), 'in_frac': int(lut.in_frac),
             'out_bits': int(lut.out_bits), 'out_frac': int(lut.out_frac),
             'out_signed': bool(lut.out_signed),
             'table': [int(v) for v in lut.table] + [0] * (lut_entries - lut.table.size)}
            for _, lut in lut_tables
        ]
        with open('./config.json', 'w') as cj:
            json.dump({'defines': defines_json, 'bundles': bundles_json,
                       'luts': luts_json}, cj, indent=4)

        mask_nums = [(2**hw.X_BITS-1) << (p*hw.X_BITS)  for p in range(8//hw.X_BITS)]
        mask_nums = ~np.array(mask_nums, dtype=np.uint8)
        ch.write(f"static const uint8_t X_POSITION_INVERTED_MASKS [] = {{ {', '.join([str(n) for n in mask_nums])} }};\n")

        '''
        Write Binary Files
        '''
        type_d = { 'np': {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64} }

        w_bitstring = b''
        x_bitstring = b''
        b_bitstring = b''
        x_bitstring_0 = b''

        for ib, b in enumerate(BUNDLES):
            assert ib == b.ib
            x_bitstring_b = b''
            if b.core.b:
                b_bitstring += b.be.astype(type_d['np'][hw.B_BITS]).tobytes()
            for ip in range(b.r.CP):  # ip: pass index (0..CP-1)
                xe = pack_words_into_bytes(arr=b.xe[ip].flatten(), bits=hw.X_BITS)
                x_bitstring_b += xe.tobytes()

                for it in range(b.r.IT):  # it: iteration index (0..IT-1)
                    we = pack_words_into_bytes(arr=b.we[ip][it].flatten(), bits=hw.K_BITS)
                    w_bitstring += we.tobytes()
            x_bitstring += x_bitstring_b
            with open(f"{hw.DATA_DIR}/{ib}_x_sim.bin", 'wb') as f:
                f.write(x_bitstring_b)
            if ib==0:
                x_bitstring_0 = x_bitstring_b
        with open(f"{hw.DATA_DIR}/x.bin", 'wb') as f:
            f.write(x_bitstring_0)

        with open(f"{hw.DATA_DIR}/wb.bin", 'wb') as f:
            f.write(w_bitstring + b_bitstring)

        with open(f"{hw.DATA_DIR}/wbx.bin", 'wb') as f:
            f.write(w_bitstring + b_bitstring + x_bitstring_0)

        with open(f"{hw.DATA_DIR}/x_all.bin", 'wb') as f:
            f.write(x_bitstring)


        '''
        Write Text files of vectors
        '''
        for ib, b in enumerate(BUNDLES):
            assert ib == b.ib
            np.savetxt(f"{hw.DATA_DIR}/{b.ib}_y_nhwc_exp.txt", b.oe_exp_nhwc.flatten(), fmt='%d')
            np.savetxt(f"{hw.DATA_DIR}/{b.ib}_xe.txt", np.concatenate([a.flatten() for a in b.xe]), fmt='%d')
            for ip in range(b.r.CP):
                CM_p = b.r.CM_0 if ip==0 else b.r.CM

                xp = b.xe[ip].flatten()
                np.savetxt(f"{hw.DATA_DIR}/{b.ib}_{ip}_x.txt", xp, fmt='%d')

                for it in range(b.r.IT):
                    wp = b.we[ip][it].flatten()
                    assert wp.shape == ((CM_p*b.r.KH+hw.CONFIG_BEATS)*hw.COLS,), f"{wp.shape} != {(CM_p*b.r.KH+hw.CONFIG_BEATS)*hw.COLS}"
                    np.savetxt(f"{hw.DATA_DIR}/{b.ib}_{ip}_{it}_w.txt", wp, fmt='%d')
                    np.savetxt(f"{hw.DATA_DIR}/{b.ib}_{ip}_{it}_y_exp.txt", b.ye_exp_p[ip][it].flatten(), fmt='%d')

        y_exp = (b.out.ftensor if b.softmax else b.o_int).flatten()
        np.savetxt(f"{hw.DATA_DIR}/y_exp.txt", y_exp, fmt= '%f' if b.softmax else '%d')
        for i in range(len(y_exp)):
            if (i < 20 or len(y_exp)-i < 20):
                print(f"y_exp {i}: {y_exp[i]}")

        print(f'Weights, inputs, outputs saved to {hw.DATA_DIR}/ib_ip_it_*.txt')


def verify_inference(model, hw, SIM, SIM_PATH='', TRACE=False):
    # SIM: simulator name ('verilator'/'icarus'/'xsim'), SIM_PATH: dir containing the simulator binary, TRACE: enable waveform dump
    # Below: _exp = expected value (computed in Python), _sim = value read back from the RTL simulation output
    '''
    RUN SIMULATION
    '''
    hw.simulate(SIM=SIM, SIM_PATH=SIM_PATH, TRACE=TRACE)


    '''
    CHECK ERROR
    '''
    for ib, b in enumerate(BUNDLES):
        assert ib == b.ib

        ''' Verify raw output '''
        for ip in range(b.r.CP):      # ip: pass index
            for it in range(b.r.IT):  # it: iteration index
                y_raw_exp = b.ye_exp_p[ip][it]
                y_raw_sim = np.loadtxt(f"{hw.DATA_DIR}/{b.ib}_{ip}_{it}_y_raw_sim.txt", np.int32)[:y_raw_exp.size].reshape(y_raw_exp.shape)
                error = np.sum(np.abs(y_raw_exp-y_raw_sim))
                assert error == 0, f"Error={error}, for y_raw_sim at {b.ib=}_{ip=}_{it=}"

        ''' Verify sum output '''
        y_sum_exp = b.oe_sum_exp
        y_sum_sim = np.loadtxt(f"{hw.DATA_DIR}/{b.ib}_y_sum_sim.txt", np.int32)[:y_sum_exp.size].reshape(y_sum_exp.shape)
        error = np.sum(np.abs(y_sum_exp-y_sum_sim))
        assert error == 0, f"Error={error}, for y_sum_sim at {b.ib=}"

        ''' Verify processed output HWC'''
        if not (ib == len(BUNDLES)-1 and b.softmax):
            y_nhwc_sim = np.loadtxt(f"{hw.DATA_DIR}/{b.ib}_y_nhwc_sim.txt",np.int32).reshape(b.oe_exp_nhwc.shape)
            error = np.sum(np.abs(y_nhwc_sim - b.oe_exp_nhwc))
            assert error == 0, f"sim:\n{y_nhwc_sim[0,:,:,0]}\n exp:\n{b.oe_exp_nhwc[0,:,:,0]}\n input:\n{b.pool.x.itensor[0,:,:,0] if b.pool else None}"


        ''' Verify tiled output'''
        if (ib == len(BUNDLES)-1):
            if b.softmax:
                y_tiled_exp = b.out.ftensor.reshape(1,b.r.XH,1,b.r.CO)
                y_tiled_sim = np.loadtxt(f"{hw.DATA_DIR}/{b.ib}_y_tiled_sim.txt", np.float32).reshape(y_tiled_exp.shape)/2**17
                error = np.max(np.abs(y_tiled_sim-y_tiled_exp))
                assert np.allclose(y_tiled_sim, y_tiled_exp, atol=0.5), f"Error={error}, \nsub:\n{y_tiled_sim-y_tiled_exp} for y_tiled_sim at {b.ib=}. \n y_tiled_sim=\n{y_tiled_sim} \n y_tiled_exp=\n{y_tiled_exp}\n \npre_softmax=\n{b.pre_softmax}"
            else:
                y_tiled_exp = b.o_int
                y_tiled_sim = np.loadtxt(f"{hw.DATA_DIR}/{b.ib}_y_tiled_sim.txt", np.float32).reshape(y_tiled_exp.shape)/2**17
                error = np.sum(np.abs(y_tiled_sim-y_tiled_exp))
                assert error == 0, f"Error={error}, for y_tiled_sim at {b.ib=}"
        else:
            y_tiled_exp = np.concatenate([a.flatten() for a in BUNDLES[ib+1].xe])
            y_tiled_sim = np.loadtxt(f"{hw.DATA_DIR}/{b.ib}_y_tiled_sim.txt", np.float32).reshape(y_tiled_exp.shape)
            error = np.sum(np.abs(y_tiled_sim-y_tiled_exp))
            assert error == 0, f"Error={error}, for y_tiled_sim at {b.ib=}"

        ''' Verify packed output'''
        if ib != len(BUNDLES)-1 and len(b.next_ibs) != 0:
            with open(f'{hw.DATA_DIR}/{ib}_y_packed_sim.bin', 'rb') as f_sim, open(f'{hw.DATA_DIR}/{ib+1}_x_sim.bin', 'rb') as f_exp:
                y_packed_sim = np.frombuffer(f_sim.read(), dtype=np.uint8)
                y_packed_exp = np.frombuffer(f_exp.read(), dtype=np.uint8)
            diff  = y_packed_sim-y_packed_exp
            error = np.sum(np.abs(diff))
            assert error == 0, f"Error={error}, for y_packed_sim at {b.ib=}, y_packed_sim=\n{y_packed_sim[:100]} \n y_packed_exp=\n{y_packed_exp[:100]}\n, diff=\n{diff.tolist()}\n  y_packed_sim=\n{y_packed_sim.tolist()} \n y_packed_exp=\n{y_packed_exp.tolist()}\n"

        print(f"Bundle {b.ib}, Error: {error}. Passed")
