import torch.nn as nn

from deepsocflow.py.brevitas.xlayer.quantOperation import QuantResidualAdd

# Every XBundle instance registers itself here (mirrors deepsocflow/py/utils.py's
# BUNDLES) so bundle-to-bundle graph edges (ib/prev_ib/next_ibs/next_add_ibs) can be
# tracked across a forward pass.
BUNDLES = []


def reset_bundles():
    BUNDLES.clear()


class XBundle(nn.Module):

    def __init__(self, core, pool=None, add_act=None, flatten=False, softmax=False,
                 pre_pad=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = core
        self.pool = pool
        # An explicit pad applied before the core, used only to express TF-'same'
        # asymmetric padding for a STRIDED conv - torch's Conv2d(padding=) can
        # only pad symmetrically, and for an even input size that lands the first
        # window one pixel away from where the engine puts it. It exists purely so
        # the float model matches the hardware; the engine still receives the
        # UNPADDED tensor and does its own padding internally.
        self.pre_pad = pre_pad

        self.add = QuantResidualAdd(act=add_act) if add_act else None
        self.flatten = nn.Flatten() if flatten else None
        self.softmax = nn.Softmax(dim=-1) if softmax else None

        self.ib = None            # this bundle's index in the global BUNDLES list
        self.prev_ib = None       # ib of the bundle whose output feeds this bundle's main input
        self.next_ibs = []        # ibs of bundles that consume this bundle's main output
        self.next_add_ibs = []    # ibs of bundles that consume this bundle's output via a residual/skip add

    def call(self, x, x_add=None):  # x_add: residual/skip-connection tensor to add, if any

        self.ib = len(BUNDLES)
        BUNDLES.append(self)

        if hasattr(x, "ib"):
            self.prev_ib = x.ib
            BUNDLES[self.prev_ib].next_ibs += [self.ib]

        if self.pre_pad is not None:
            x = self.pre_pad(x)

        x = self.core(x)
        x = self.core.act(x)

        if x_add is not None:
            assert self.add is not None, "Activation function must be provided for add layer"
            self.add.source_ib = x_add.ib
            BUNDLES[x_add.ib].next_add_ibs += [self.ib]
            x = self.add(x, x_add)
        elif self.add is not None:
            raise ValueError("A Bundle initialized with add_act, should have the add tensor passed")

        if self.pool:
            x = self.pool(x)
            x = self.pool.act(x)

        if self.flatten:
            x = self.flatten(x)

        if self.softmax:
            x = self.softmax(x)

        # TODO: cache self.out = x here once an XTensor port exists, so call_int/export
        # can read BUNDLES[ib].out the way the Keras original does (xbundle.py:73-75).
        x.ib = self.ib
        return x

    forward = call

    def call_int(self, x, hw):  # x: XTensor input (only used for the first/ib==0 bundle), hw: Hardware config
        raise NotImplementedError(
            "XBundle.call_int requires an XTensor + hardware.py port for the brevitas backend; "
            "it will run the fixed-point simulation and assert parity against call()"
        )

    def export(self, hw, is_last):  # hw: Hardware config, is_last: True if this is the final bundle in the network
        raise NotImplementedError(
            "XBundle.export requires the hardware-engine-layout reorder helpers "
            "(reorder_*_q2e_conv) ported to the brevitas backend"
        )
