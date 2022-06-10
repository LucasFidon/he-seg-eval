
LABELS = {
    'Normal': 1,
    'Stroma': 2,
    'Tumor': 3,
    'background': 0,
}
LABELS_EVAL = {
    'Abnormal': [2, 3],
    'Normal': [1],
    'Stroma': [2],
    'Tumor': [3],
}
ROI = list(LABELS.keys())
ROI_EVAL = list(LABELS_EVAL.keys())
METRICS = ['dice', 'hausdorff']
# TODO: what is the unit
MAX_HD = 300  # in mm (default maximum value given when the predicted mask is empty)
SPACING = [1, 1, 1]  # in mm
