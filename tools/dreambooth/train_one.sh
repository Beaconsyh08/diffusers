accelerate config default
export MODEL_NAME="/mnt/ve_share/generation/models/online/diffusions/res/finetune/dreambooth/SD-HM-V0.4.0"
# export INSTANCE_DIR="./data/train/finetune/$DATA_NAME"
export INSTANCE_DIR="/mnt/ve_share/generation/data/train/GAN/all_snow/imgs"
export OUTPUT_DIR="/mnt/ve_share/generation/models/online/diffusions/res/finetune/dreambooth/SD-HM-V0.4.0.snow.4"
# export CLASS_DIR="./data/train/finetune/night_class"
export INSTANCE_PROMPT="/mnt/ve_share/generation/data/train/GAN/all_snow/pmps"

accelerate launch ./examples/dreambooth/train_dreambooth_one.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt=$INSTANCE_PROMPT \
  --resolution=512 \
  --train_batch_size=16 \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=64 \
  --checkpointing_steps=10000
  
pip install safetensors
python ./scripts/convert_diffusers_to_original_stable_diffusion.py --use_safetensors --model_path $OUTPUT_DIR --checkpoint_path $OUTPUT_DIR/model.safetensors
cp $OUTPUT_DIR/model.safetensors /cpfs/model/model.safetensors

# 200 for 32000  max_train_steps * train_batch_size
