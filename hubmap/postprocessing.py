
import numpy as np
from skimage.morphology import convex_hull_image
from scipy import ndimage


def make_cc_convex(seg):
    seg_out = np.zeros_like(seg)
    seg_comp, num_comp = ndimage.label(seg)
    for c in range(1, num_comp + 1):
        seg_c = (seg_comp == c)
        seg_c = convex_hull_image(seg_c)
        seg_out[seg_c] = 1
    return seg_out
