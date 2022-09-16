import os
import cv2
import numpy as np
from skimage import io
import pandas as pd
from argparse import ArgumentParser
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.preprocessing.preprocessing import resample_patient
from nnunet.paths import network_training_output_dir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

DEFAULT_FOLDER_TO_INFER = '/storage/hpa_hubmap_kidney/validation_PAS/images_3000_3000_shift_V1_kidney'
OUT_FOLDER = '/workspace'
CSV = '/storage/hpa_hubmap_kidney/validation_PAS/val_kidney.csv'
DEFAULT_TASK = 162
DEFAULT_TRAINER = 'nnUNetTrainerV2'
PLAN = 'nnUNetPlansv2.1_HPA_HuBMAP'
MODEL_MPP = 1.
SCALES_TTA = [0.75, 1.25]

TRAINERS_AVAILABLE = ['nnUNetTrainerV2', 'coatTrainer']

parser = ArgumentParser()
parser.add_argument('--input', nargs='+', default=DEFAULT_FOLDER_TO_INFER)
parser.add_argument('--trainer', type=str, default=DEFAULT_TRAINER,
                    help='try values in %s' % str(TRAINERS_AVAILABLE))
parser.add_argument('--task', type=int, default=DEFAULT_TASK)
parser.add_argument('--folds', nargs='+', type=int, default=0)
parser.add_argument('--tta', action='store_true')


def load_image(tiff_path: str):
    img = io.imread(tiff_path)
    return img


def preprocess(img, mpp, model_mpp):
    ori_spacing = (999, mpp, mpp)
    target_spacing = (999, model_mpp, model_mpp)
    res = img.transpose((2, 0, 1))
    # Add z dimension
    res = res[:, None, :, :]
    res = res.astype(float)
    res, _ = resample_patient(
        data=res, seg=None, original_spacing=ori_spacing,
        target_spacing=target_spacing, order_data=1, force_separate_z=True,
    )
    mean = res.mean(axis=(1, 2, 3))
    std = res.std(axis=(1, 2, 3))
    res = (res - mean[:, None, None, None]) / std[:, None, None, None]
    return res


# https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
# with transposed mask
def rle_encode_less_memory(img):
    # the image should be transposed
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def main(args):
    from time import time
    t0 = time()
    df = pd.read_csv(CSV)

    names = []
    preds = []

    # Prepare model
    task_name = convert_id_to_task_name(args.task)
    model_folder = os.path.join(
        network_training_output_dir,
        '2d',
        task_name,
        args.trainer + "__" + PLAN,
    )
    folds = args.folds
    assert os.path.exists(model_folder), 'Folder %s not found' % model_folder
    trainer, params = load_model_and_checkpoint_files(
        model_folder, folds, mixed_precision=False,
        checkpoint_name='model_final_checkpoint',
    )

    # Inference
    for i, row in df.iterrows():
        idx = str(row.id)
        print('\n\033[92m(%d/%d) Image' % (i, len(df)), idx, '\033[0m')
        mpp = float(row.pixel_size)
        h = 3000
        w = 3000
        softmax = 0.
        # Allow TTA using more than one input folder
        for input_path in args.input:
            tiff_path = os.path.join(input_path, '%s.tiff' % idx)
            print('infer on', tiff_path)
            img0 = load_image(tiff_path)
            img = preprocess(img0, mpp, MODEL_MPP)
            aug_img = None
            if args.tta:
                print('Use TTA')
                aug_img = {}
                for s in SCALES_TTA:
                    aug_img[s] = preprocess(img0, mpp, s * MODEL_MPP)
            for p in params:
                trainer.load_checkpoint_ram(p, False)
                softmax += trainer.predict_preprocessed_data_return_seg_and_softmax(
                    img, do_mirroring=True, mirror_axes=trainer.data_aug_params['mirror_axes'],
                    use_sliding_window=True, step_size=0.5, use_gaussian=True, all_in_gpu=None,
                    mixed_precision=False)[1]
                if aug_img is not None:
                    for s in SCALES_TTA:
                        softmax_s = trainer.predict_preprocessed_data_return_seg_and_softmax(
                            aug_img[s], do_mirroring=True, mirror_axes=trainer.data_aug_params['mirror_axes'],
                            use_sliding_window=True, step_size=0.5, use_gaussian=True, all_in_gpu=None,
                            mixed_precision=False)[1]
                        softmax_model_mpp, _ = resample_patient(
                            data=softmax_s, seg=None,
                            original_spacing=(999, s * MODEL_MPP, s * MODEL_MPP),
                            target_spacing=(999, MODEL_MPP, MODEL_MPP),
                            order_data=1, force_separate_z=True,
                        )
                        softmax += softmax_model_mpp
        # todo investigate why we get -0.2 mean Dice for task 162
        # try upsampling the softmax
        # try with mixed precision
        pred_seg = np.argmax(softmax, axis=0)[0]
        seg = cv2.resize(pred_seg, (h, w), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        rle = rle_encode_less_memory(seg)
        names.append(idx)
        preds.append(rle)

    df = pd.DataFrame({'id': names, 'rle': preds})
    csv_n = 'pred_task%d_tr%s_fold' % (args.task, args.trainer)
    for f in folds:
        csv_n += str(f)
    if args.tta:
        csv_n += '_tta'
    for p in args.input:
        csv_n += '_' + p.split('/')[-1]
    csv_n += '.csv'
    out_csv = os.path.join(OUT_FOLDER, csv_n)
    df.to_csv(out_csv, index=False)

    print('Total time', time() - t0, 'seconds')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
