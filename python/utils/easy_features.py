"""Thin wiring around easy_local_features, mirroring how HLoc wires it in
HLoc/hloc/extractors/easy-features.py (an easy_local_features model is built
with `getMethod(elf_model, elf_conf)` and moved to the best available
device); see also the 'easy-features' entries in HLoc/hloc/extract_features.py.
"""
import torch
from easy_local_features import getMethod


def get_extractor(elf_model, elf_conf=None):
    """Instantiate an easy_local_features extractor and move it to the best
    available device, the same way hloc.extractors.EasyFeatures._init does.

    Like hloc's wiring, this accepts any name from
    easy_local_features.available_methods (getMethod itself raises a clear
    ValueError for unknown names) -- not just available_extractors, since
    methods hloc already wires this way (e.g. 'sfd2') live outside that
    narrower list.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getMethod(elf_model, elf_conf or {})
    model.to(device)
    return model, device
