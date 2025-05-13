#!/bin/bash

# Default parameters
MODEL_TYPE="improved_cnn"
BATCH_SIZE=128
EPOCHS=100
LEARNING_RATE=0.1
WEIGHT_DECAY=5e-4
SCHEDULER="cosine"
DATA_AUGMENTATION=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        --no_data_augmentation)
            DATA_AUGMENTATION=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create experiment directory
EXPERIMENT_DIR="experiments/improved_$(date +%Y%m%d_%H%M%S)_${MODEL_TYPE}"
mkdir -p "$EXPERIMENT_DIR"

# Run the training
python improved_cifar_training.py \
    --model_type "$MODEL_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --scheduler "$SCHEDULER" \
    $([ "$DATA_AUGMENTATION" = true ] && echo "--data_augmentation") \
    > "$EXPERIMENT_DIR/training.log" 2>&1

# Move the best model to the experiment directory
if [ -f "best_model_${MODEL_TYPE}.pth" ]; then
    mv "best_model_${MODEL_TYPE}.pth" "$EXPERIMENT_DIR/"
fi

echo "Experiment completed. Results saved in $EXPERIMENT_DIR" 