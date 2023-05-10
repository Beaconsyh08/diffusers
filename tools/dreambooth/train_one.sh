accelerate config default
DATA_NAME="haomo"
export MODEL_NAME="/mnt/share_disk/lei/git/diffusers/local_models/stable-diffusion-v1-5"
# export INSTANCE_DIR="./data/train/finetune/$DATA_NAME"
export INSTANCE_DIR="/mnt/ve_share/generation/data/train/diffusions/5000/imgs"
export OUTPUT_DIR="./res/finetune/dreambooth/${DATA_NAME}_5000"
# export CLASS_DIR="./data/train/finetune/night_class"
export INSTANCE_PROMPT="/mnt/ve_share/generation/data/train/diffusions/5000/pmps"

accelerate launch ./examples/dreambooth/train_dreambooth_one.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=512 \
  --train_batch_size=10 \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=80000 \
  --checkpointing_steps=10000

# 200 for 32000  max_train_steps * train_batch_size
  # --with_prior_preservation --prior_loss_weight=1.0 \

  # --class_prompt="a photo of night traffic scene" \
#   --use_8bit_adam \