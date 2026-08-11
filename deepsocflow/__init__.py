# The legacy qkeras backend (utils/dataflow/xbundle/xmodel/xlayers/hardware)
# needs tensorflow. The brevitas backend (deepsocflow.py.brevitas.*,
# deepsocflow.py.numeric) does not - but Python always runs this __init__.py
# first for any deepsocflow.* import, so a hard failure here would block the
# brevitas backend too. Only skip the legacy imports when tensorflow itself
# is unavailable; any other ImportError still raises normally.
try:
    import tensorflow as _tensorflow
except ImportError:
    _tensorflow = None

if _tensorflow is not None:
    from deepsocflow.py.utils import *
    from deepsocflow.py.dataflow import *
    from deepsocflow.py.xbundle import *
    from deepsocflow.py.xmodel import *
    from deepsocflow.py.xlayers import *
    from deepsocflow.py.hardware import *