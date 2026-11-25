"""
Configuration file for Tyre Defect Detection pipeline.
Contains all hyperparameters, paths, and constants.
"""

import torch
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
DATASET_ROOT = Path(__file__).parent
TRAIN_DIR = DATASET_ROOT / "data" / "train" / "images"
TRAIN_LABELS = DATASET_ROOT / "data" / "train" / "labels" / "_train.csv"
VALID_DIR = DATASET_ROOT / "data" / "valid" / "images"
VALID_LABELS = DATASET_ROOT / "data" / "valid" / "labels" / "_valid.csv"
TEST_DIR = DATASET_ROOT / "data" / "test" / "images"
TEST_LABELS = DATASET_ROOT / "data" / "test" / "labels" / "_test.csv"

OUTPUT_DIR = DATASET_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# DEVICE
# ============================================================================
DEVICE = torch.device("cpu")  # Force CPU for training
print(f"Using device: {DEVICE}")

# GPU Optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Faster convolution algorithms
    torch.backends.cudnn.deterministic = False  # Slight speed improvement
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPU available. Running on CPU (slow)")


# ============================================================================
# REPRODUCIBILITY
# ============================================================================
SEED = 42

# ============================================================================
# CLASS CONFIGURATION
# ============================================================================
NUM_CLASSES = 3
CLASS_NAMES = ["Bulge", "Cracks", "Non-defective"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# CSV column names (in order: Bulge, Cracks, Non-defective)
LABEL_COLUMNS = ["Bulge", "Cracks", "Non-defective"]

# ============================================================================
# IMAGE CONFIGURATION
# ============================================================================
ORIGINAL_IMG_SIZE = 640
TARGET_IMG_SIZE = 224
NUM_CHANNELS = 3  # After converting grayscale to RGB

# ============================================================================
# NORMALIZATION (ImageNet stats)
# ============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# AUGMENTATION PARAMETERS
# ============================================================================
# These are defaults and will be overridden by Optuna tuning
TRAIN_AUG_ROTATION = 15
TRAIN_AUG_BRIGHTNESS = 0.2
TRAIN_AUG_CONTRAST = 0.2

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_NAME = "efficientnet_b2"
DROPOUT_RATE = 0.3

# ============================================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================================
OPTUNA_TRIALS = 20
OPTUNA_TRIAL_EPOCHS = 5  # Each trial trains Phase 1 for 5 epochs

# ============================================================================
# TRAINING PHASES
# ============================================================================
PHASE_1_EPOCHS = 2  # Reduced for CPU testing
PHASE_2_EPOCHS = 2  # Reduced for CPU testing
PHASE_3_EPOCHS = 2  # Reduced for CPU testing

PHASE_1_LR = 1e-3
PHASE_2_LR = 1e-4
PHASE_3_LR = 1e-5

# Batch sizes (optimized for CPU training - reduced for speed)
# Training: 8 (reduced for CPU)
# Validation/Test: 16 (reduced for CPU)
BATCH_SIZE = 8
BATCH_SIZE_VAL_TEST = 16

# Mixed precision (recommended for CUDA GPUs with low VRAM)
USE_AMP = True

# ============================================================================
# OPTIMIZER & SCHEDULER
# ============================================================================
OPTIMIZER_TYPE = "adam"  # adam or sgd
SCHEDULER_TYPE = "cosine"  # cosine annealing

# ============================================================================
# EARLY STOPPING
# ============================================================================
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "val_f1_weighted"  # Can be val_loss, val_accuracy, val_f1_weighted

# ============================================================================
# LOSS FUNCTION
# ============================================================================
LOSS_FN = "cross_entropy"  # Will use class weights for imbalance

# ============================================================================
# LOGGING & CHECKPOINTS
# ============================================================================
TRAINING_LOG_FILE = OUTPUT_DIR / "training_log.csv"
TEST_RESULTS_FILE = OUTPUT_DIR / "test_results.json"
CONFUSION_MATRIX_FILE = OUTPUT_DIR / "confusion_matrix.png"
BEST_OPTUNA_PARAMS_FILE = OUTPUT_DIR / "best_optuna_params.json"

CHECKPOINT_PHASE_1 = OUTPUT_DIR / "best_checkpoint_phase1.pt"
CHECKPOINT_PHASE_2 = OUTPUT_DIR / "best_checkpoint_phase2.pt"
CHECKPOINT_PHASE_3 = OUTPUT_DIR / "best_checkpoint_phase3.pt"

# ============================================================================
# INFERENCE
# ============================================================================
CONFIDENCE_THRESHOLD = 0.0  # No threshold, return top prediction
