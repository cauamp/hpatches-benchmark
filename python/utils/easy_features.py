"""Thin wiring around easy_local_features, mirroring how HLoc wires it in
HLoc/hloc/extractors/easy-features.py (an easy_local_features model is built
with `getMethod(elf_model, elf_conf)` and moved to the best available
device); see also the 'easy-features' entries in HLoc/hloc/extract_features.py.
"""
import types

import numpy as np
import scipy
import torch
from easy_local_features import getMethod

# easy_local_features' d2net baseline still calls the long-removed
# scipy.misc.imresize(arr, scale_factor) for images above its size cap; scipy
# dropped that module years ago. Restore just enough of it (float scale
# factor only, matching how baseline_d2net.py calls it) rather than patching
# the vendored package.
if not hasattr(scipy, 'misc'):
    def _imresize_compat(arr, size, interp='bilinear'):
        from PIL import Image
        im = Image.fromarray(np.asarray(arr).astype(np.uint8))
        resample = {'nearest': Image.NEAREST, 'bilinear': Image.BILINEAR,
                    'bicubic': Image.BICUBIC}.get(interp, Image.BILINEAR)
        new_size = (int(round(im.width * size)), int(round(im.height * size)))
        return np.array(im.resize(new_size, resample))

    scipy.misc = types.SimpleNamespace(imresize=_imresize_compat)


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
