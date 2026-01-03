from data.datasets import (
    VQADataset,
    COCOCaptionsVanillaDataset,
)
from data.collators import (
    VQACollator,
    COCOCaptionsVanillaCollator,
)
from evaluation.cider_utils import (
    VanillaCOCOGenerationDataset,
    VanillaGenerationCollator,
)


TRAIN_DATASET_COLLATOR_MAP = {
    "coco_caption": (COCOCaptionsVanillaDataset, COCOCaptionsVanillaCollator),
    "ocr_vqa": (None, None),
    "vqa": (VQADataset, VQACollator),
}

GEN_DATASET_COLLATOR_MAP = {
    "coco_caption": (VanillaCOCOGenerationDataset, VanillaGenerationCollator),
    "ocr_vqa": (None, None),
    "vqa": (None, None),
}