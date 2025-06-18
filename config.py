import os

DATASET_PATH = 'dataset'

DATASET_SUBFOLDER_METADATA = os.path.join(DATASET_PATH, 'Skin Cancer')

DATASET_SUBFOLDER_IMAGES = os.path.join(DATASET_SUBFOLDER_METADATA, 'Skin Cancer')

CSV_PATH = os.path.join(DATASET_PATH, 'HAM10000_metadata.csv')
IMAGE_DIR = DATASET_SUBFOLDER_IMAGES

LABEL_MAP = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}

CLASS_NAMES = list(LABEL_MAP.keys())
NUM_CLASSES = len(CLASS_NAMES)

MODEL_SAVE_PATH = 'models/'
RESULT_PATH = 'results/'

KAGGLE_DATASET_NAME = "farjanakabirsamanta/skin-cancer-dataset"