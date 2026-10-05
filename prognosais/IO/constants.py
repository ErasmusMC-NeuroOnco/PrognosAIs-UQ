# Data constants
DATA_LABELS_TRAIN = "labels_all.txt"
DATA_LABELS_TEST = "labels.txt"
DATA_NIFTI_EXTENSION = ".nii.gz"
DATA_TXT_EXTENSION = ".txt"
T1_MODALITY = "T1"
T1CE_MODALITY = "T1CE"
T2_MODALITY = "T2"
FLAIR_MODALITY = "FLAIR"
SCAN_TYPES = [T1_MODALITY, T1CE_MODALITY, T2_MODALITY, FLAIR_MODALITY]
MASK_FILE_NAME = "MASK"
CANONICAL_MODALITY_ORDER = SCAN_TYPES + [MASK_FILE_NAME]
DATA_MASK_ORIGIN = "mask_origin.txt"

SUPPORTED_MODALITIES = set(SCAN_TYPES)

CSNET_MODEL_TYPE = "csnet"
MODEL_TYPES = [CSNET_MODEL_TYPE]

# Preprocessing constants
FAILED_PATIENTS_FOLDER = "failed_patients"
DICOM_FOLDER = "DICOM"
NIFTI_FOLDER = "NIFTI"
BIASFIELD_CORRECTED_FOLDER = "BIASFIELD_CORRECTED"
REGISTERED_FOLDER = "REGISTERED"
ELASTIX_FOLDER = "ELASTIX_PARAMETERS"
PREPROCESSED_FOLDER = "PREPROCESSED"
GLIOSEG_PREPROCESSED_FOLDER = "FINAL_POSTPROCESSED"

MNI_ATLAS_FOLDER = "MNI152_atlas"
T1_MNI_ATLAS = "mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
T2_MNI_ATLAS = "mni_icbm152_t2_tal_nlin_sym_09a.nii.gz"
MNI_ATLAS_MASK = "brain_mask.nii.gz"

PARAMETER_FOLDER = "parameters"
ELASTIX_RIGID_PARAMETER_MAP_FILE = "parameter_map_rigid.txt"
ELASTIX_AFFINE_PARAMETER_MAP_FILE = "parameter_map_affine.txt"
ELASTIX_LOGFILE = "logfile_{modality}.log"
ELASTIX_PARAMETERS = "elastix_parameters_{modality}.{0}.txt"

N_JOBS_PREPROCESSING = 5
FAILURE_REPORT_FILE = "preprocessing_failure_report_{exact_time}.txt"

# Results constants
LOGFILES_DIR_NAME = "logfiles"
OUTPUT_LOGFILES_DIR_NAME = "output"
ERROR_LOGFILES_DIR_NAME = "error"
RESULTS_DIR_NAME = "results"
INFORMATION_DIR_NAME = "information"
PREDICTIONS_DIR_NAME = "predictions"
MODELS_DIR_NAME = "models"
METRICS_DIR_NAME = "metrics"
FOLDS_SUMMARY_DIR_NAME = "folds_summary"

PREDICTIONS_SUMMARY_LABELED_STEM = "preds_summary_labeled"
METRICS_STATISTICS_STEM = "metrics_statistics"
PREDICTIONS_SUMMARY_UNLABELED_STEM = "preds_summary_unlabeled"
SEGMENTATION_METRICS_STEM = "segmentation_metrics"
DICE_SCORE_SUMMARY_STEM = "dice_score_summary"
HAUSDORFF_DISTANCE_SUMMARY_STEM = "hausdorff_distance_summary"
MODEL_SUMMARY_FILE = "model_summary.txt"

PREDICTION_FILE_NAME = "PRED"
PROBABILITY_MAP_FILE_NAME = "PROB_MAP"

UQ_RESULTS_DIR_NAME = "UQ"
MC_DROPOUT_DIR_NAME = "MC_dropout"
DEEP_ENSEMBLE_DIR_NAME = "Deep_ensemble"
MC_DROPOUT_DEEP_ENSEMBLE_DIR_NAME = "MC_DE"
UQ_METHOD_MCD = "mcd"
UQ_METHOD_DE = "de"
UQ_METHOD_MCD_DE = "mcd_de"
INFERENCE_MODE_LABELED = "labeled"
INFERENCE_MODE_UNLABELED = "unlabeled"
SUPPORTED_INFERENCE_MODES = (
    INFERENCE_MODE_LABELED,
    INFERENCE_MODE_UNLABELED,
)
SUPPORTED_UQ_METHODS = (UQ_METHOD_MCD, UQ_METHOD_DE, UQ_METHOD_MCD_DE)
UQ_METHOD_RESULTS_DIR_NAMES = {
    UQ_METHOD_MCD: MC_DROPOUT_DIR_NAME,
    UQ_METHOD_DE: DEEP_ENSEMBLE_DIR_NAME,
    UQ_METHOD_MCD_DE: MC_DROPOUT_DEEP_ENSEMBLE_DIR_NAME,
}
UQ_METHODS_REQUIRING_MC_SAMPLES = (UQ_METHOD_MCD, UQ_METHOD_MCD_DE)
UQ_METHODS_REQUIRING_NUM_MODELS = (UQ_METHOD_DE, UQ_METHOD_MCD_DE)
UQ_METHODS_WITH_EMBEDDED_SEGMENTATION_LABELS = (
    UQ_METHOD_MCD,
    UQ_METHOD_MCD_DE,
)
UQ_AGGREGATION_MEMORY_CONFIG_KEY = "uq_aggregation"
DROPOUT_EXPERIMENT_DIR_PREFIX = "dropout_"
PROBABILITY_SUM_TOLERANCE = 1e-3
ENTROPY_EPSILON = 1e-10
MC_DROPOUT_RELATIVE_DIR = (
    f"{RESULTS_DIR_NAME}/{UQ_RESULTS_DIR_NAME}/{MC_DROPOUT_DIR_NAME}"
)
PROBABILITY_MAP_RELATIVE_PATH = (
    f"{RESULTS_DIR_NAME}/{PREDICTIONS_DIR_NAME}/{{case}}/"
    f"{PROBABILITY_MAP_FILE_NAME}{DATA_NIFTI_EXTENSION}"
)

ROC_CHANCE_LEVEL_LABEL = "Chance level AUC 0.5"
BOXPLOT_FILE_STEMS = {
    "dice": "boxplot_dice",
    "hd": "boxplot_hd",
}
METRIC_COLUMN_NAME = "Metric"
MIN_COLUMN_NAME = "Min"
MAX_COLUMN_NAME = "Max"
MEAN_COLUMN_NAME = "Mean"
MEDIAN_COLUMN_NAME = "Median"
PRECISION_COLUMN_NAME = "precision"
RECALL_COLUMN_NAME = "recall"
SENSITIVITY_COLUMN_NAME = "sensitivity"
SPECIFICITY_COLUMN_NAME = "specificity"
F1_SCORE_COLUMN_NAME = "f1-score"
SUPPORT_COLUMN_NAME = "support"
ACCURACY_ROW_NAME = "accuracy"
MACRO_AVG_ROW_NAME = "macro avg"
WEIGHTED_AVG_ROW_NAME = "weighted avg"
CASE_COLUMN_NAME = "Case"
CASE_ID_DATA_KEY = "case_id"
TRUE_CLASS_COLUMN_NAME = "True class"
PREDICTED_CLASS_COLUMN_NAME = "Predicted class"
CORRECTLY_CLASSIFIED_COLUMN_NAME = "Correctly classified"
CONFIDENCE_LABEL_COLUMN_TEMPLATE = "Confidence label {class_idx}"
TASK_COLUMN_NAME = "Task"
METHOD_COLUMN_NAME = "Method"
ERROR_COLUMN_NAME = "Error"
PREDICTIVE_ENTROPY_COLUMN_NAME = "Predictive entropy"
EXPECTED_ENTROPY_COLUMN_NAME = "Expected entropy"
MUTUAL_INFORMATION_COLUMN_NAME = "Mutual information"
MEAN_PROBABILITY_VECTOR_COLUMN_NAME = "Mean probability vector"
UQ_SCHEMA_VERSION = 3
SUPPORTED_IMAGE_OUTPUT_FORMATS = (".png", ".svg", ".pdf")
SUPPORTED_TABLE_OUTPUT_FORMATS = (".csv", ".xlsx")
GRADE_SHIFT_OFFSET = 2
# NUM_CLASSES_IDH = 2
# NUM_CLASSES_1P19Q = 2
# NUM_CLASSES_GRADE = 3
# NUM_CLASSES_SEG = 2
SEGMENTATION_CLASS_LABELS = ("Background", "Tumor")
NUM_SEGMENTATION_CLASSES = len(SEGMENTATION_CLASS_LABELS)
SEGMENTATION_MEAN_PREDICTIONS_KEY = "Mean predictions"
SEGMENTATION_TASK_NAME = "Tumor segmentation"
SEGMENTATION_DICE_SCORE_COLUMN = "Dice score"
SEGMENTATION_HD_COLUMN = "HD"
HAUSDORFF_DISTANCE_METRIC_NAME = "Hausdorff distance"
SEGMENTATION_TRUE_CLASS_COLUMN = TRUE_CLASS_COLUMN_NAME
SEGMENTATION_PREDICTED_CLASS_COLUMN = PREDICTED_CLASS_COLUMN_NAME
SEGMENTATION_CASE_COLUMN = CASE_COLUMN_NAME
SEGMENTATION_MUQ_PREFIX = "MUQ_seg"
SEGMENTATION_UQ_PREFIX = "UQ_seg"
SEGMENTATION_LABELS_FILENAME = "labels.pkl"
SEGMENTATION_MASKS_DIR_PREFIX = "MASKS_"
SEGMENTATION_UQ_MAPS_DIR_PREFIX = "UQ_maps_"
SEGMENTATION_GT_PREFIX = "GT_"
SEGMENTATION_PRED_PREFIX = "PRED_"
SEGMENTATION_PUM_PREFIX = "PUM_"
SEGMENTATION_AUM_PREFIX = "AUM_"
SEGMENTATION_EUM_PREFIX = "EUM_"
ROC_CHANCE_LEVEL_SHORT_LABEL = "Chance level"

HDBET_MIN_LENGTH = [110, 110, 110]
BRAIN_MASK_NAME = "mask_brain_"

# Classification task keys
TASK_IDH = "idh"
TASK_1P19Q = "onep19q"
TASK_GRADE = "grade"

DEFAULT_CLASSIFICATION_TASKS = (
    TASK_IDH,
    TASK_1P19Q,
    TASK_GRADE,
)

SUPPORTED_CLASSIFICATION_TASKS = (
    TASK_IDH,
    TASK_1P19Q,
    TASK_GRADE,
)

# Classification definition keys
NUM_CLASSES_KEY = "num_classes"
OUTPUT_KEY_KEY = "output_key"
OUTPUT_NAME_KEY = "output_name"
DISPLAY_NAME_KEY = "display_name"
PRETTY_NAME_KEY = "pretty_name"
CLASS_NAMES_KEY = "class_names"
CLASS_VALUES_KEY = "class_values"
FILENAME_STEM_KEY = "filename_stem"
REPORT_INDEX_LABELS_KEY = "report_index_labels"

# Classification task definitions
CLASSIFICATION_TASK_DEFINITIONS = {
    TASK_IDH: {
        NUM_CLASSES_KEY: 2,
        OUTPUT_KEY_KEY: TASK_IDH,
        OUTPUT_NAME_KEY: "IDH Mutation Status",
        DISPLAY_NAME_KEY: "IDH",
        PRETTY_NAME_KEY: "IDH mutation",
        CLASS_NAMES_KEY: ("Wildtype", "Mutated"),
        CLASS_VALUES_KEY: (0, 1),
        REPORT_INDEX_LABELS_KEY: (
            "0: IDH wildtype",
            "1: IDH mutated",
        ),
        FILENAME_STEM_KEY: "IDH",
    },
    TASK_1P19Q: {
        NUM_CLASSES_KEY: 2,
        OUTPUT_KEY_KEY: TASK_1P19Q,
        OUTPUT_NAME_KEY: "1p/19q Codeletion Status",
        DISPLAY_NAME_KEY: "1p19q",
        PRETTY_NAME_KEY: "1p19q codeletion",
        CLASS_NAMES_KEY: ("Intact", "Co-deleted"),
        CLASS_VALUES_KEY: (0, 1),
        REPORT_INDEX_LABELS_KEY: (
            "0: 1p19q intact",
            "1: 1p19q co-deleted",
        ),
        FILENAME_STEM_KEY: "1p19q",
    },
    TASK_GRADE: {
        NUM_CLASSES_KEY: 3,
        OUTPUT_KEY_KEY: TASK_GRADE,
        OUTPUT_NAME_KEY: "Tumor Grade",
        DISPLAY_NAME_KEY: "Grade",
        PRETTY_NAME_KEY: "Tumor grade",
        CLASS_NAMES_KEY: ("Grade 2", "Grade 3", "Grade 4"),
        CLASS_VALUES_KEY: (2, 3, 4),
        REPORT_INDEX_LABELS_KEY: (
            "0: Grade 2",
            "1: Grade 3",
            "2: Grade 4",
        ),
        FILENAME_STEM_KEY: "Grade",
    },
}

NUM_CLASSES_PER_LABEL = {
    task_name: task_definition[NUM_CLASSES_KEY]
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASS_NAMES_PER_LABEL = {
    task_name: task_definition[CLASS_NAMES_KEY]
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASS_VALUES_PER_LABEL = {
    task_name: task_definition[CLASS_VALUES_KEY]
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASSIFICATION_OUTPUT_KEYS = {
    task_name: task_definition[OUTPUT_KEY_KEY]
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASSIFICATION_OUTPUT_NAMES = {
    task_name: task_definition[OUTPUT_NAME_KEY]
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASSIFICATION_DISPLAY_NAMES = {
    task_name: task_definition[DISPLAY_NAME_KEY]
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASSIFICATION_PRETTY_NAMES = {
    task_name: task_definition[PRETTY_NAME_KEY]
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

DEFAULT_CLASS_DICT = {
    task_name: NUM_CLASSES_PER_LABEL[task_name]
    for task_name in DEFAULT_CLASSIFICATION_TASKS
}

CLASSIFICATION_REPORT_FILE_STEMS = {
    task_name: (f"cr_{task_definition[FILENAME_STEM_KEY].lower()}")
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASSIFICATION_LABEL_COLUMNS = {
    task_name: (f"Label {task_definition[DISPLAY_NAME_KEY]}")
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASSIFICATION_PROBABILITY_COLUMNS = {
    task_name: (f"Probs {task_definition[DISPLAY_NAME_KEY]}")
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASSIFICATION_PREDICTION_COLUMNS = {
    task_name: (f"Prediction {task_definition[DISPLAY_NAME_KEY]}")
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CONFUSION_MATRIX_FILE_STEMS = {
    task_name: (f"{task_definition[FILENAME_STEM_KEY]}_cm")
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

DISTRIBUTION_FILE_STEMS = {
    task_name: (f"{task_definition[FILENAME_STEM_KEY]}_distribution")
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASSIFICATION_CONFUSION_MATRIX_LABELS = {
    task_name: task_definition[CLASS_NAMES_KEY]
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}

CLASSIFICATION_REPORT_INDEX_LABELS = {
    task_name: task_definition[REPORT_INDEX_LABELS_KEY]
    for task_name, task_definition in CLASSIFICATION_TASK_DEFINITIONS.items()
}
