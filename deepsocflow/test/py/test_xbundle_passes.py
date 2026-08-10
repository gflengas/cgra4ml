"""Guards the per-pass input-channel slicing in XBundle.export. Commit d3091e2
dropped `ic_right += CM_p` from that loop while adding comments, making every
pass slice [0:0]; nothing caught it because the arithmetic was inlined in a loop
that needs a fully built model to reach."""
from collections import namedtuple

import pytest


def _runtime(CP, CM_0, CM, CI):
    return namedtuple('R', ['CP', 'CM_0', 'CM', 'CI'])(CP=CP, CM_0=CM_0, CM=CM, CI=CI)


def test_single_pass_covers_all_channels():
    pytest.importorskip("tensorflow")
    from deepsocflow.py.xbundle import _pass_channel_slices

    assert _pass_channel_slices(_runtime(CP=1, CM_0=3, CM=72, CI=3)) == [(0, 3)]


def test_multi_pass_slices_are_contiguous_and_cover_all_channels():
    pytest.importorskip("tensorflow")
    from deepsocflow.py.xbundle import _pass_channel_slices

    # CI=200 split as CM_0=56 then two full passes of 72
    slices = _pass_channel_slices(_runtime(CP=3, CM_0=56, CM=72, CI=200))

    assert len(slices) == 3
    assert slices[0][0] == 0, "first pass must start at channel 0"
    assert slices[-1][1] == 200, "last pass must end at CI"
    for (_, prev_right), (next_left, _) in zip(slices, slices[1:]):
        assert prev_right == next_left, "slices must be contiguous, no gaps"


def test_no_slice_is_empty():
    """The actual d3091e2 regression: every slice was [0:0], which TF's conv2d
    rejects with 'filter depth must be strictly positive, got 0'."""
    pytest.importorskip("tensorflow")
    from deepsocflow.py.xbundle import _pass_channel_slices

    for left, right in _pass_channel_slices(_runtime(CP=3, CM_0=56, CM=72, CI=200)):
        assert right > left, f"empty channel slice [{left}:{right}]"


def test_first_pass_uses_cm_0_and_rest_use_cm():
    pytest.importorskip("tensorflow")
    from deepsocflow.py.xbundle import _pass_channel_slices

    slices = _pass_channel_slices(_runtime(CP=3, CM_0=56, CM=72, CI=200))
    widths = [right - left for left, right in slices]
    assert widths == [56, 72, 72]
