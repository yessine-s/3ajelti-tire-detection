# Tyre Defect Detection - CPU Training

## Quick Start

```bash
python main.py --no-tuning
```

## Project Structure

```
tyre_model/
├── Python Source Files
│   ├── config.py                 # Configuration (Device=CPU, Batch=8/16, Epochs=2/2/2)
│   ├── dataset.py                # Data loading (num_workers=0 for Windows)
│   ├── model.py                  # EfficientNet-B2 architecture
│   ├── train.py                  # Training with epoch output (Loss, Accuracy, F1)
│   ├── evaluate.py               # Model evaluation
│   ├── inference.py              # Prediction on new images
│   ├── main.py                   # Main pipeline
│   ├── hyperparameter_tuning.py  # Optuna hyperparameter tuning
│   └── test_gpu_epoch.py         # Single epoch test
│
├── Configuration
│   └── requirements.txt           # Python dependencies
│
├── Data & Output
│   ├── data/                      # Dataset (train, valid, test)
│   └── outputs/                   # Checkpoints and results
│
└── Documentation
    └── README.md                  # This file
```

## Run Commands

### Full Pipeline (with Optuna tuning)
```bash
python main.py
```

### Skip Tuning (faster)
```bash
python main.py --no-tuning
```

### Test Single Epoch
```bash
python test_gpu_epoch.py
```

## Output Format

Each epoch shows:
```
Phase 1: Freeze Backbone Epoch 1/2
======================================================================
Learning Rate: 1.00e-03
======================================================================
TRAIN -> Loss: 0.8234 | Accuracy: 0.6532 | F1-Score: 0.6234
VAL   -> Loss: 0.7456 | Accuracy: 0.7123 | F1-Score: 0.6987
======================================================================
```

## Training Configuration (CPU)

Device: CPU
Batch Size (Train): 8
Batch Size (Val/Test): 16
Phase 1 Epochs: 2
Phase 2 Epochs: 2
Phase 3 Epochs: 2
Estimated Time: 45-60 minutes

## To Increase Training Quality

Edit `config.py`:
```python
PHASE_1_EPOCHS = 10  # Change from 2
PHASE_2_EPOCHS = 10  # Change from 2
PHASE_3_EPOCHS = 15  # Change from 2
```

This will give you full training (approximately 4-5 hours on CPU).

## Output Files

After training:
- outputs/training_log.csv - Training history
- outputs/test_results.json - Test metrics
- outputs/confusion_matrix.png - Confusion matrix
- outputs/best_checkpoint_phase*.pt - Model checkpoints

## Status

✅ Ready for CPU training
✅ Clear epoch output with metrics
✅ All optimizations applied
✅ Cache cleared
✅ Unused files removed
