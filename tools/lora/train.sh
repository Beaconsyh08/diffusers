accelerate config default
DATA_NAME="haomo_night"
export MODEL_NAME="/mnt/share_disk/lei/git/diffusers/local_models/stable-diffusion-v1-5"
export OUTPUT_DIR="./res/finetune/lora/$DATA_NAME"
# export HUB_MODEL_ID="pokemon-lora"
# export DATASET_NAME="./data/train/finetune/pokemon/data"
export DATASET_NAME="./data/train/finetune/$DATA_NAME"
# export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16"  ./examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --seed=1337

  # --push_to_hub \
  # --hub_model_id=${HUB_MODEL_ID} \
  # --validation_prompt="A pokemon with blue eyes." \
