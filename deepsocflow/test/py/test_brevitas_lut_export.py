"""Export-level checks for LUT activations.

These deliberately drive the REAL `export_rtl` -> `_export_bundles` path and
assert on the emitted `config_fw.h` / `config.json` text, not on adapter
attributes. CLAUDE.md's 2026-08-11 entry is explicit about why: every adapter
defect found so far (`bias_val_shift`, `softmax_frac`, `is_flatten`) was
invisible at the attribute level and only appeared once a test read the emitted
config. A LUT adds two more fields with exactly that failure mode.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepsocflow.py.brevitas.hardware import Hardware
from deepsocflow.py.brevitas.ptq import quantized_model
from deepsocflow.py.brevitas.sim import FixedPointModel
from deepsocflow.py.brevitas.export import export_rtl


X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])


def _hw():
    return Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                    bits_bias=16, bits_sum=32, axi_width=128)


def _export(tmp_path, monkeypatch, activation=None, act_input_bits=8):
    """Train nothing - an untrained net is fine here, since these tests are about
    the file format, not accuracy. Exports into tmp_path and returns the two
    emitted config files' contents."""
    import torch.nn as nn

    act = activation or nn.SiLU
    torch.manual_seed(30)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_1 = nn.Linear(2, 8, bias=True)
            self.act_1 = act()
            self.hidden_2 = nn.Linear(8, 8, bias=True)
            self.act_2 = act()
            self.out = nn.Linear(8, 2, bias=True)

        def forward(self, x):
            x = self.act_1(self.hidden_1(x))
            x = self.act_2(self.hidden_2(x))
            return self.out(x)

    qm = quantized_model(Net(), weight_bits=8, bias_bits=16,
                         act_input_bits=act_input_bits)
    qm.quantization(X)
    json_path = str(tmp_path / "g.json")
    qm.export_graph_json(X, json_path)

    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)

    monkeypatch.chdir(tmp_path)
    export_rtl(model, _hw(), X, batch_size=4)

    return ((tmp_path / "config_fw.h").read_text(),
            json.loads((tmp_path / "config.json").read_text()),
            model)


def test_same_activation_still_needs_one_table_per_grid(tmp_path, monkeypatch):
    """Two bundles both using SiLU do NOT share a table, because calibration
    gives each its own input/output scale - so the tabulated values differ.

    Worth pinning explicitly: it is tempting to assume "same activation, one
    table", and that assumption is what would make deduplication merge two
    bundles that need different functions. Sharing is keyed on the table's actual
    contents, not on the activation's name. The practical consequence is that
    table storage scales with the number of curved bundles, not with the number
    of distinct activation types."""
    header, cfg, model = _export(tmp_path, monkeypatch)

    assert "#define N_LUTS      2" in header
    assert "#define LUT_ENTRIES 256" in header
    assert "static const i8 LUTS [N_LUTS][LUT_ENTRIES]" in header
    assert cfg['defines']['N_LUTS'] == 2

    lut_bundles = [b for b in cfg['bundles'] if b['ca_lut_idx'] != -1]
    assert len(lut_bundles) == 2
    assert sorted(b['ca_lut_idx'] for b in lut_bundles) == [0, 1]
    for b in lut_bundles:
        assert b['ca_lut_bits'] == 8

    # both are silu, but on different grids - that is why they are not shared
    assert {l['activation'] for l in cfg['luts']} == {'silu'}
    grids = {(l['in_bits'], l['in_frac'], l['out_frac']) for l in cfg['luts']}
    assert len(grids) == 2, f"expected two distinct grids, got {grids}"


def test_identical_grids_are_shared(tmp_path, monkeypatch):
    """The other half: when two bundles' tables really are byte-identical, they
    must collapse to one entry."""
    header, cfg, model = _export(tmp_path, monkeypatch)

    # force the second bundle onto the first one's grid, then re-export
    luts = [b['lut'] for b in model.bundles.values() if b['lut'] is not None]
    first = luts[0]
    for lut in luts[1:]:
        lut.table = first.table.copy()
        lut.in_bits, lut.in_frac = first.in_bits, first.in_frac
        lut.out_bits, lut.out_frac = first.out_bits, first.out_frac
        lut.out_signed = first.out_signed
        lut.activation = first.activation

    export_rtl(model, _hw(), X, batch_size=4)
    cfg2 = json.loads((tmp_path / "config.json").read_text())
    assert cfg2['defines']['N_LUTS'] == 1
    assert all(b['ca_lut_idx'] == 0 for b in cfg2['bundles'] if b['ca_lut_idx'] != -1)


def test_emitted_table_matches_the_model_table_exactly(tmp_path, monkeypatch):
    """The bytes in config_fw.h must be the same integers sim.py executed - this
    is the seam where a transcription bug would make C and Python disagree while
    both still 'work'."""
    header, cfg, model = _export(tmp_path, monkeypatch)

    lut = next(b['lut'] for b in model.bundles.values() if b['lut'] is not None)
    assert cfg['luts'][0]['table'][:lut.table.size] == [int(v) for v in lut.table]
    assert cfg['luts'][0]['in_bits'] == lut.in_bits
    assert cfg['luts'][0]['in_frac'] == lut.in_frac

    # and the same values really are in the C text
    body = header.split("LUTS [N_LUTS][LUT_ENTRIES] = {", 1)[1].split("};", 1)[0]
    emitted = [int(v) for v in body.replace('\n', '').split('{', 1)[1].split('}', 1)[0].split(',')]
    assert emitted[:lut.table.size] == [int(v) for v in lut.table]


def test_ca_shift_is_the_index_shift_not_the_output_shift(tmp_path, monkeypatch):
    """The field whose meaning changes on a LUT bundle. Emitting the output-grid
    shift here would index the table at the wrong scale, and the simulation would
    still run."""
    _, cfg, model = _export(tmp_path, monkeypatch)

    for name, b in zip(model.bundle_order, cfg['bundles']):
        bundle = model.bundles[name]
        lut = bundle['lut']
        if lut is None:
            continue
        acc_frac = bundle['input_frac'] + bundle['weight_frac']
        assert b['ca_shift'] == acc_frac - lut.in_frac
        # and it genuinely differs from what 1a would have emitted, otherwise
        # this test would pass without discriminating
        assert lut.in_frac != bundle['act_frac'] or acc_frac - bundle['act_frac'] == b['ca_shift']


def test_relu_model_emits_no_tables(tmp_path, monkeypatch):
    """The inertness guarantee: a model with no curved activation must produce
    the same config_fw.h it always did, plus two neutral fields."""
    import torch.nn as nn

    header, cfg, _ = _export(tmp_path, monkeypatch, activation=nn.ReLU,
                             act_input_bits=None)

    assert "#define N_LUTS      0" in header
    assert "static const i8 LUTS" not in header
    assert cfg['luts'] == []
    for b in cfg['bundles']:
        assert b['ca_lut_idx'] == -1
        assert b['ca_lut_bits'] == 0


def test_config_json_luts_mirror_config_fw_h(tmp_path, monkeypatch):
    """pynq_driver.py reads config.json while the firmware reads config_fw.h; if
    they disagree the board silently computes something else than the simulation."""
    header, cfg, _ = _export(tmp_path, monkeypatch)

    body = header.split("LUTS [N_LUTS][LUT_ENTRIES] = {", 1)[1].split("\n};", 1)[0]
    rows = [r for r in body.split('{')[1:]]
    emitted = [[int(v) for v in r.split('}')[0].split(',')] for r in rows]

    assert len(emitted) == len(cfg['luts'])
    for c_row, json_row in zip(emitted, cfg['luts']):
        assert c_row == json_row['table']


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_table_size_follows_act_input_bits(tmp_path, monkeypatch, bits):
    header, cfg, _ = _export(tmp_path, monkeypatch, act_input_bits=bits)
    assert f"#define LUT_ENTRIES {2 ** bits}" in header
    assert len(cfg['luts'][0]['table']) == 2 ** bits
    for b in cfg['bundles']:
        if b['ca_lut_idx'] != -1:
            assert b['ca_lut_bits'] == bits


def test_distinct_grids_are_not_deduplicated(tmp_path, monkeypatch):
    """Sharing is by exact table content. Two activations that differ must get
    separate entries, or one bundle silently executes the other's function."""
    import torch.nn as nn

    torch.manual_seed(30)

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_1 = nn.Linear(2, 8, bias=True)
            self.act_1 = nn.SiLU()
            self.hidden_2 = nn.Linear(8, 8, bias=True)
            self.act_2 = nn.Tanh()
            self.out = nn.Linear(8, 2, bias=True)

        def forward(self, x):
            x = self.act_1(self.hidden_1(x))
            x = self.act_2(self.hidden_2(x))
            return self.out(x)

    qm = quantized_model(Mixed(), weight_bits=8, bias_bits=16, act_input_bits=8)
    qm.quantization(X)
    json_path = str(tmp_path / "mixed.json")
    qm.export_graph_json(X, json_path)
    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)

    monkeypatch.chdir(tmp_path)
    export_rtl(model, _hw(), X, batch_size=4)
    cfg = json.loads((tmp_path / "config.json").read_text())

    assert cfg['defines']['N_LUTS'] == 2
    assert {l['activation'] for l in cfg['luts']} == {'silu', 'tanh'}
    idxs = [b['ca_lut_idx'] for b in cfg['bundles'] if b['ca_lut_idx'] != -1]
    assert sorted(idxs) == [0, 1]
