import os
import cv2
import numpy as np
from skimage import io
import pandas as pd
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.preprocessing.preprocessing import resample_patient
from nnunet.paths import network_training_output_dir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

# DEFAULT_FOLDER_TO_INFER = '/workspace/hubmap-prostate'
DEFAULT_FOLDER_TO_INFER = '/workspace/tiles_prostate_6mpp'
OUT_FOLDER = '/workspace'
DEFAULT_TASK = 161
TRAINER = 'nnUNetTrainerV2'
PLAN = 'nnUNetPlansv2.1_HPA_HuBMAP'
MODEL_MPP = 1.
MPP = 6.263


def load_image(tiff_path: str):
    img = io.imread(tiff_path)
    return img


def preprocess(img, mpp, model_mpp):
    ori_spacing = (999, mpp, mpp)
    target_spacing = (999, model_mpp, model_mpp)
    img = img.transpose((2, 0, 1))
    # Add z dimension
    img = img[:, None, :, :]
    img = img.astype(float)
    img, _ = resample_patient(
        data=img, seg=None, original_spacing=ori_spacing,
        target_spacing=target_spacing, order_data=1, force_separate_z=True,
    )
    mean = img.mean(axis=(1, 2, 3))
    std = img.std(axis=(1, 2, 3))
    img = (img - mean[:, None, None, None]) / std[:, None, None, None]
    return img


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


def main():
    from time import time
    t0 = time()

    names = []
    preds = []

    # Prepare model
    task_name = convert_id_to_task_name(DEFAULT_TASK)
    model_folder = os.path.join(
        network_training_output_dir,
        '2d',
        task_name,
        TRAINER + "__" + PLAN,
    )
    folds = 0
    assert os.path.exists(model_folder), 'Folder %s not found' % model_folder
    trainer, params = load_model_and_checkpoint_files(
        model_folder, folds, mixed_precision=False,
        checkpoint_name='model_final_checkpoint',
    )
    trainer.load_checkpoint_ram(params[0], False)

    # Get file paths
    files = [
        os.path.join(DEFAULT_FOLDER_TO_INFER, fn)
        for fn in os.listdir(DEFAULT_FOLDER_TO_INFER)
        if fn[-4:] == '.png'
    ]

    # Inference
    # for folder_n in os.listdir(DEFAULT_FOLDER_TO_INFER):
    #     input_path = os.path.join(DEFAULT_FOLDER_TO_INFER, folder_n)
    #     for file_n in os.listdir(input_path):
    #         if file_n[-4:] != '.png':
    #             continue
    for png_path in files:
        idx = os.path.split(png_path)[1].split('.')[0]
        print('\n\033[92mImage', idx, '\033[0m')
        # h = 3000
        # w = 3000
        print('infer on', png_path)
        img = load_image(png_path)
        w, h, _ = img.shape
        # import pdb
        # pdb.set_trace()
        img = preprocess(img, MPP, MODEL_MPP)
        softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
            img, do_mirroring=True, mirror_axes=trainer.data_aug_params['mirror_axes'],
            use_sliding_window=True, step_size=0.5, use_gaussian=True, all_in_gpu=None,
            mixed_precision=False)[1]
        pred_seg = np.argmax(softmax, axis=0)[0]
        seg = cv2.resize(pred_seg, (h, w), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        rle = rle_encode_less_memory(seg)
        names.append(idx)
        preds.append(rle)

    df = pd.DataFrame({'id': names, 'rle': preds})
    csv_n = 'pred_task%d_prostate_MPP6.csv' % DEFAULT_TASK
    out_csv = os.path.join(OUT_FOLDER, csv_n)
    df.to_csv(out_csv, index=False)

    print('Total time', time() - t0, 'seconds')


if __name__ == '__main__':
    main()
