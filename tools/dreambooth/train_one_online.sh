pip install -e ".[torch]"

helpFunction()
{
   echo ""
   echo "Usage: $0 -a INSTANCE_DIR -b INSTANCE_PROMPT -c OUTPUT_DIR"
   echo -e "\t-a Description of what is INSTANCE_DIR"
   echo -e "\t-b Description of what is INSTANCE_PROMPT"
   echo -e "\t-c Description of what is OUTPUT_DIR"
   exit 1 # Exit script after printing help
}

while getopts "i:p:o:" opt
do
   case "$opt" in
      i ) INSTANCE_DIR="$OPTARG" ;;
      p ) INSTANCE_PROMPT="$OPTARG" ;;
      o ) OUTPUT_DIR="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$INSTANCE_DIR" ] || [ -z "$INSTANCE_PROMPT" ] || [ -z "$OUTPUT_DIR" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$INSTANCE_DIR"
echo "$INSTANCE_PROMPT"
echo "$OUTPUT_DIR"

accelerate config default
export DATA_NAME="haomo"
export MODEL_NAME="/mnt/ve_share/generation/models/online/diffusions/base/stable-diffusion-v1-5"
# export INSTANCE_DIR="./data/train/finetune/$DATA_NAME"
# export INSTANCE_DIR="/share/generation/data/train/diffusions/5000/imgs"
# export OUTPUT_DIR="/share/generation/models/online/diffusions/res/finetune/dreambooth/${DATA_NAME}_5000_seg1"
# # export CLASS_DIR="./data/train/finetune/night_class"
# export INSTANCE_PROMPT="/share/generation/data/train/diffusions/5000/pmps_seg_test1"

accelerate launch --multi_gpu ./examples/dreambooth/train_dreambooth_one.py \
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
  --max_train_steps=6250 \
  --checkpointing_steps=3125

# 1 for 160  max_train_steps * train_batch_size * gpu
  # --with_prior_preservation --prior_loss_weight=1.0 \

  # --class_prompt="a photo of night traffic scene" \
#   --use_8bit_adam \


#  ./tools/dreambooth/train_one_online.sh -i /mnt/ve_share/generation/data/train/diffusions/5000/imgs -p /mnt/ve_share/generation/data/train/diffusions/5000/pmps_seg_test1 -o /mnt/ve_share/generation/models/online/diffusions/res/finetune/dreambooth/haomo_5000_seg1_ttt