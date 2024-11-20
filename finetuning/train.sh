DATETIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/${DATETIME}"}
DATA_PATH=${DATA_PATH:-"./data"}
mkdir -p $OUTPUT_DIR

BASE_MODEL=${BASE_MODEL:-"meta-llama/Llama-3.1-8B"}
DATASET_NAME=${DATASET_NAME:-"ise-uiuc/Magicoder-OSS-Instruct-75K"}

export WANDB_API_KEY="96fd9604d382603e6eadd53e82d8e45bd4389bd3"
export HUGGING_FACE_HUB_TOKEN="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"

WANDB_API_KEY="96fd9604d382603e6eadd53e82d8e45bd4389bd3"
HUGGING_FACE_HUB_TOKEN="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"

torchrun --nproc_per_node=8 \
    --master_port=1234 finetune.py \
    --base_model $BASE_MODEL \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 64 \
    --micro_batch_size 2 \
    --num_epochs 2 \
    --learning_rate 3e-4 \
    --cutoff_len 2048 \
    --val_set_size 0.10 \
    --add_eos_token True \
    --seed 42 \
    --train_on_inputs True \
    --use_flash_attention True \
    --wandb_project "program-cal-diverse" 

# Copy tokenizer files to appropriate location, modify this if model is different
# if [[ $BASE_MODEL == *"7b"* ]]; then
#     cp -r ./tokenizer_files/7B/* $OUTPUT_DIR
# elif [[ $BASE_MODEL == *"13b"* ]]; then
#     cp -r ./tokenizer_files/13B/* $OUTPUT_DIR
# else
#     echo "Base model size not recognized. Tokenizer files not copied."
# fi
cp -r ./tokenizer_files/* $OUTPUT_DIR
# wee~
