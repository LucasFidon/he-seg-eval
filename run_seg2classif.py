"""
Convert segmentations into classification labels using the rule
class i proba = num pixel pred i  / num pixels not background
for i not background.
Here background = white voxels

Only for WSSS4LUAD.
"""

import os
import pickle
from argparse import ArgumentParser
import numpy as np
from skimage import io
import nibabel as nib
from definitions import *
from utils.WSSS4LUAD import *

parser = ArgumentParser()
parser.add_argument('--seg_folder', required=True)
parser.add_argument(
    '--img_folder',
    default='/Users/lfidon/data/WSSS4LUAD/1.training',
)


def main(args):
    case_names = [
        n.replace('.nii.gz', '')
        for n in os.listdir(args.seg_folder)
        if '.nii.gz' in n
    ]
    pred_proba = {}
    for case_n in case_names:
        print('\n***', case_n)
        pred_proba[case_n] = {}
        # Load segmentation
        seg_path = os.path.join(
            args.seg_folder, '%s.nii.gz' % case_n)
        seg_nii = nib.load(seg_path)
        seg_np = seg_nii.get_fdata().astype(np.uint8)

        # Load image
        img_name = None
        for img_n in os.listdir(args.img_folder):
            if case_n in img_n:
                img_name = img_n
                break
        img_path = os.path.join(
            args.img_folder, img_name)
        img = io.imread(img_path)

        # Get true label
        label_code = img_name.replace('.png', '').split('-')[-1]
        label = None
        if label_code == '[1, 0, 0]':
            label = 2
        elif label_code == '[0, 1, 0]':
            label = 1
        elif label_code == '[0, 0, 1]':
            label = 0
        elif label_code == '[1, 1, 0]':
            label = 3
        pred_proba[case_n]['true_label'] = label
        print('true label:', label)

        # Get the background mask
        bg_mask = gen_bg_mask(img)  # 1 is background
        num_fg = np.sum(bg_mask == 0)
        proba = []
        for c in range(1, 4):
            p_c = np.sum(seg_np == c) / num_fg
            proba.append(p_c)
        pred_proba[case_n]['pred'] = proba
        print('predicted proba:', proba)

    save_path = os.path.join(
        args.seg_folder, 'classification.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(pred_proba, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
