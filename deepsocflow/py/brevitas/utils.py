# Every XBundle instance registers itself here (mirrors deepsocflow/py/utils.py's
# BUNDLES) so bundle-to-bundle graph edges (ib/prev_ib/next_ibs/next_add_ibs) can be
# tracked across a forward pass.
BUNDLES = []


def reset_bundles():
    BUNDLES.clear()
