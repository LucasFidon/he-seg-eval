import os
import numpy as np
import nibabel as nib
import pickle
from evaluation_metrics.segmentation_metrics import *
from definitions import *

METRIC_NAMES = ['dice', 'hausdorff']


def compute_evaluation_metrics(pred_seg_path, gt_seg_path, roi_list, roi_to_labels):
    def load_np(seg_path):
        seg = nib.load(seg_path).get_fdata().astype(np.uint8)
        seg = np.squeeze(seg)  # Remove axes of length one
        return seg
    pred_seg_folder, pred_seg_name = os.path.split(pred_seg_path)
    pred_seg = load_np(pred_seg_path)
    gt_seg = load_np(gt_seg_path)

    # Check which ROIs are present
    is_present = {}
    labels_present = np.unique(gt_seg).tolist()
    for roi in roi_list:
        present = False
        for label in roi_to_labels[roi]:
            if label in labels_present:
                present = True
        is_present[roi] = present

    # Compute the metrics
    dice_values = {}
    haus_values = {}
    for roi in roi_list:
        if is_present[roi]:
            dice_values[roi] = 100 * dice_score(
                pred_seg,
                gt_seg,
                fg_class=roi_to_labels[roi],
            )
            haus_values[roi] = min(
                MAX_HD,
                haussdorff_distance(
                    pred_seg,
                    gt_seg,
                    fg_class=roi_to_labels[roi],
                    percentile=95,
                )
            )
        else:
            dice_values[roi] = None
            haus_values[roi] = None
    print('\n\033[92mEvaluation for %s\033[0m' % pred_seg_name)
    print('Dice scores:')
    print(dice_values)
    print('Hausdorff95 distances:')
    print(haus_values)
    return dice_values, haus_values


def print_results(metrics, roi_list=ROI, save_path=None):
    print('\nGlobal statistics for the metrics')
    for roi in roi_list:
        print('\033[92m%s\033[0m' % roi)
        for metric in METRIC_NAMES:
            key = '%s_%s' % (metric, roi)
            n_cases = len(metrics[key])
            if n_cases == 0:
                print('No data for %s' % key)
                continue
            mean = np.mean(metrics[key])
            std = np.std(metrics[key])
            q3 = np.percentile(metrics[key], 75)
            p95 = np.percentile(metrics[key], 95)
            q1 = np.percentile(metrics[key], 25)
            p5 = np.percentile(metrics[key], 5)
            print(key)
            print('%d cases' % n_cases)
            if metric == 'dice':
                print('mean=%.1f, std=%.1f, q1=%.1f, p5=%.1f' % (mean, std, q1, p5))
            else:
                print('mean=%.1f, std=%.1f, q3=%.1f, p95=%.1f' % (mean, std, q3, p95))
        print('-----------')
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)
