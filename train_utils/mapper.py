from data.datasets import (
    VQADataset,
    COCOCaptionsVanillaDataset,
    OCRVQAVanillaDataset,
    VQADualDataset,
    OCRVQADualDataset,
)
from data.collators import (
    VQACollator,
    COCOCaptionsVanillaCollator,
    OCRVQAVanillaCollator,
    VQADualCollator,
)
from evaluation.coco_captions import (
    COCOCaptionsVanillaGenerationDataset,
    COCOCaptionsVanillaGenerationCollator,
    COCOCaptionsDualTowerGenerationDataset,
    COCOCaptionsDualTowerGenerationCollator,
)
from evaluation.ocr_vqa import (
    OCRVQAVanillaGenerationDataset,
    OCRVQAVanillaGenerationCollator,
    OCRVQADualTowerGenerationDataset,
    OCRVQADualTowerGenerationCollator,
)


TRAIN_DATASET_COLLATOR_MAP = {
    "coco_captions": (COCOCaptionsVanillaDataset, COCOCaptionsVanillaCollator),
    "coco_captions_dual": (VQADualDataset, VQADualCollator),
    
    "ocr_vqa": (OCRVQAVanillaDataset, OCRVQAVanillaCollator),
    "ocr_vqa_dual": (OCRVQADualDataset, VQADualCollator),
    
    "vqa": (VQADataset, VQACollator),
    "vqa_dual": (VQADualDataset, VQADualCollator),
}

GEN_DATASET_COLLATOR_MAP = {
    "coco_captions": (COCOCaptionsVanillaGenerationDataset, COCOCaptionsVanillaGenerationCollator),
    "coco_captions_dual": (COCOCaptionsDualTowerGenerationDataset, COCOCaptionsDualTowerGenerationCollator),
    
    "ocr_vqa": (OCRVQAVanillaGenerationDataset, OCRVQAVanillaGenerationCollator),
    "ocr_vqa_dual": (OCRVQADualTowerGenerationDataset, OCRVQADualTowerGenerationCollator),

    "vqa": (None, None),
    "vqa_dual": (None, None),
}