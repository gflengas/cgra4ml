# What ResNet still needs — design and feasibility

## Verdict

**ResNet18 is feasible.** Of the three remaining gaps, two are Python-only and one needs a firmware change to `runtime.h` — but **not an RTL change and therefore no re-synthesis and no new bitstream**, because the piece that has to change (output tiling) runs on the CPU.

| Gap | Feasible | Where the work is |
|---|---|---|
| Strided conv on an even input | **Yes — verified numerically** | Python only (`ptq.py`) |
| `'same'` pooling + `AdaptiveAvgPool2d` | Yes | Python only (`ptq.py`, `sim.py`) |
| Branching main path (downsample blocks) | Yes, but | `runtime.h` + `Bundle_t` + `pynq_driver.py`. CPU-side; no fabric change |

Residual add and BatchNorm are already done (2026-08-13/14).

---

## 1. Strided conv on an even input — solved, and the fix is small

### The problem

The engine computes a **stride-1, symmetrically-padded** convolution and then keeps pixels from `CSH_SHIFT` with stride `s` (`dataflow.py:44-46`). For a 3x3 stride-2 conv over an 8-wide axis that offset is **1**, so it keeps pixels 1,3,5,7. `torch.nn.Conv2d(padding=1, stride=2)` keeps 0,2,4,6. Same shape, plausible values, every feature map off by one pixel — currently refused by `ptq.py::_assert_stride_matches_engine`, which is why every strided conv in ResNet18 is rejected.

### The fix

The two agree once the float model uses TF's **asymmetric** padding explicitly:

```
pad_total = max((ceil(n/s) - 1) * s + k - n, 0)
pad_lo    = pad_total // 2          # the smaller half goes top/left
nn.ZeroPad2d((pad_lo, pad_total - pad_lo, pad_lo, pad_total - pad_lo))
nn.Conv2d(..., padding=0, stride=s)
```

**Verified numerically against the engine's own path** (`conv2d_same_int` + `_apply_conv_stride`) on `n=8/16/224/56/7` with `k=3` and `k=7`: exact match on every one, including ResNet18's real geometries (224/7/2 for the stem, 56/3/2 and 28/3/2 and 14/3/2 for the stage transitions).

### Work

- `ptq.py`: recognize an `nn.ZeroPad2d` immediately before a compute layer, absorb its padding into the geometry, and validate that `zeropad + conv.padding` equals TF's `pad_total` for that input size. Keep `_assert_stride_matches_engine` for the case where no explicit pad is present — the current message should then point at this pattern.
- A helper (`tf_same_padding(n, k, s)`) so model authors do not hand-compute it, plus a `StageI` using it.

---

## 2. `'same'` pooling and `AdaptiveAvgPool2d` — straightforward

`AdaptiveAvgPool2d((1,1))` is global average pooling: with the input spatial size known at export time it is exactly `AvgPool2d(kernel_size=(H,W), stride=(H,W))`, whose window never clips, so `count` stays constant and it reuses the average pooling that already exists. Resolve the kernel from the traced input shape and map it; a few lines.

`'same'` pooling is the fiddlier half. `dataflow.py:60-72` already computes the geometry, and `runtime.h:534-535` derives `count` from the **clipped** window (`(ph_end-ph_beg)*(pw_end-pw_beg)`), so the divisor shrinks at the borders. `sim.py`'s `avgpool2d_valid_int`/`maxpool2d_valid_int` assume a constant window and must grow an edge-aware version. Mechanical, but it is exactly the kind of index arithmetic that is easy to get subtly wrong — it should be pinned against the C the way `div_round` was (`deepsocflow/test/c/div_round_dump.c`), not against a re-reading of `runtime.h`.

---

## 3. Branching main path — the real constraint, and it is not where it looks

### What ResNet needs

```
x -> conv3x3 -> bn -> relu -> conv3x3 -> bn ->(+)-> relu
 \                                          /
  \-> downsample 1x1 -> bn -----------------
```

`x` feeds **two bundles as main input**. The buffer allocator already looks like it handles this: `next_ibs` is a sorted list and the out-buffer is freed only after the last consumer (`rtl_export.py:158-175`).

### Why it does not work today

`runtime.h:249-252`:

```c
if (pb->ib_out == -1) return;
else pb_out = &bundles[pb->ib_out];
```

The output is written **in exactly one consumer's engine tiling** — everything after this uses `pb_out->cm_p0`, `pb_out->cm`, `pb_out->x_pad`, `pb_out->w`, `pb_out->l`, `pb_out->p`, `pb_out->xp_words`. The exporter picks `ib_out = sorted(b.next_ibs)[0]` (`rtl_export.py:313`) — the *first* consumer.

Two consumers with different kernel sizes need different tilings of the same data. Measured for a 64-channel tensor on `processing_elements=(8,24)`:

| consumer | KH | CM | X_PAD |
|---|---|---|---|
| 1x1 downsample | 1 | 512 | 0 |
| 3x3 main path | 3 | 170 | 6 |

So the second consumer reads a buffer laid out for the first. **This is why `run/example.py` works and a ResNet downsample block would not**: example.py's fan-out is to a *residual add* (`next_add_ibs`, a separate untiled `add_buffers` write at `runtime.h:244-245`), never two main inputs. `run/resnet18.py` does have two main consumers (`x_skip = self.b5(x_skip); x = self.b6(x)` off the same `x`) — which suggests that path was never validated end to end. Worth confirming before trusting it as a reference.

### The fix

Write the output once per distinct consumer tiling. This is **CPU-side only** — `tile_write`/`write_x` are firmware, the fabric is not involved — so it costs a firmware rebuild, not a bitstream.

- `Bundle_t` grows a small fan-out list instead of a scalar `ib_out` (plus the matching `out_buffer_idx` per entry).
- `process_and_store_output` loops over it, calling `tile_write` once per consumer.
- `rtl_export.py` emits the list; the allocator already tracks the consumers.
- `pynq_driver.py` mirrors it.

**One trap, already paid for once:** C designated initializers default missing fields to **0**, and a missing `ib_out` entry defaulting to 0 reads as "bundle 0" rather than "none". The LUT work hit exactly this with `ca_lut_idx` and solved it with a `#if defined(N_LUTS) && N_LUTS > 0` guard so the whole path compiles out when the exporter does not emit it. Use the same pattern, or have the exporter always emit an explicit count.

**Cheaper interim option:** if both consumers happen to share a kernel size, they share a tiling and it works today unchanged. That is not ResNet's shape (1x1 vs 3x3), but it is worth an assert so the unsupported case fails loudly instead of reading a mis-tiled buffer.

---

## Practical limits, separate from correctness

- `max_image_size` defaults to 32; 224x224 needs 512, which widens `BITS_COLS_MAX`/`L_MAX` and the header packing. `create_headers` has a hard assert on total header width (`dataflow.py:110`) that should be checked before assuming it fits.
- ResNet18 is ~11 M parameters; at 8-bit weights that is ~11 MB of `wbx.bin` plus activations, against whatever contiguous DMA memory `pynq.allocate` can obtain. The repo's `run/resnet18.py` is CIFAR-sized (32x32) and far smaller — a much more realistic first target than 224x224 ImageNet.
- Per-tensor weight quantization after BN folding is still unmeasured on a *trained* network (see the 2026-08-14 Progress Log entry). ResNet is where it would first matter.

## Recommended order

1. **Strided conv** — Python only, verified, and unblocks the stem plus every stage transition.
2. **`AdaptiveAvgPool2d`** — reuses existing average pooling; completes the classifier head.
3. **`'same'` pooling** — pin the edge-clipped `count` against the C.
4. **Branching main path** — last, because it is the only one that touches firmware and needs `pynq_driver.py` kept in step.

After 1-3 a ResNet *without* downsample blocks (all stages at constant width) runs. Step 4 is what makes a real ResNet18 possible.
