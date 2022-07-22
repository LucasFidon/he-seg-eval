import numpy as np
import cv2
from skimage import morphology


def gen_bg_mask(orig_img):
    """
    Tile background segmentation method.
    This is adapted from the thresholding method used for the WSSS4LUAD challenge.
    See:
    https://github.com/ChuHan89/WSSS-Tissue/blob/52fe45efcc70c41e2d2e0722cd9b961ccd2d3a75/tool/infer_utils.py#L512
    """
    img_array = np.array(orig_img).astype(np.uint8)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    binary = np.uint8(binary)
    dst = morphology.remove_small_objects(binary == 255, min_size=50, connectivity=1)
    bg_mask = np.zeros(orig_img.shape[:2])
    bg_mask[dst == True] = 1.  # 1 == background
    return bg_mask