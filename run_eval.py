import os
from argparse import ArgumentParser
from definitions import *
from evaluation_metrics.eval import compute_evaluation_metrics, METRIC_NAMES, print_results

parser = ArgumentParser()
parser.add_argument('--testing_set_folder', required=True)
parser.add_argument('--predictions_folder', required=True)
parser.add_argument('--task', default=TASKS[0], help='Any values in %s' % str(TASKS))


def main(args):
    case_names = [
        n.replace('.nii.gz', '')
        for n in os.listdir(args.testing_set_folder)
        if '.nii.gz' in n
    ]
    metrics = {
        '%s_%s' % (metric, roi): [] for roi in ROI_EVAL for metric in METRIC_NAMES
    }

    # Evaluation
    for case_n in case_names:
        pred_seg_path = os.path.join(args.predictions_folder, '%s.nii.gz' % case_n)
        gt_seg_path = os.path.join(args.testing_set_folder, '%s.nii.gz' % case_n)
        dice, haus = compute_evaluation_metrics(
            pred_seg_path,
            gt_seg_path,
            roi_list=ROI_EVAL,
            roi_to_labels=LABELS[args.task],
        )
        for roi in ROI_EVAL:
            if dice[roi] is not None:
                metrics['dice_%s' % roi].append(dice[roi])
            if haus[roi] is not None:
                metrics['hausdorff_%s' % roi].append(haus[roi])

    # Print the results
    metrics_save_path = os.path.join(args.predictions_folder, 'metrics.pkl')
    print_results(metrics, save_path=metrics_save_path, roi_list=ROI_EVAL)
    print('Metrics have been saved in %s' % metrics_save_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
