
TASKS = [
    'WSSS4LUAD',
    'WSSS4LUAD_2C',
]

LABELS = {
    'WSSS4LUAD': {
        'Normal': 1,
        'Stroma': 2,
        'Tumor': 3,
        'background': 0,
        'Normal/Stroma': [1, 2],
    },
    'WSSS4LUAD_2C': {
        'Normal/Stroma': 1,
        'Tumor': 2,
        'background': 0,
    },
}
ROI = list(LABELS['WSSS4LUAD'].keys())
ROI_EVAL = list(LABELS['WSSS4LUAD'].keys())
METRICS = ['dice', 'hausdorff']
# TODO: what is the unit
MAX_HD = 300  # in mm (default maximum value given when the predicted mask is empty)
SPACING = [1, 1, 1]  # in mm
