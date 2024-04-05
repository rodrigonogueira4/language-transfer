
# Script arguments

PRETRAINED_LANGUAGE=${1}
MODEL_SIZE=small

if [ -z "$PRETRAINED_LANGUAGE" ]; then
  echo "Skipping finetune from a pretrained model"
else 
  echo "Running finetune from model trained on $PRETRAINED_LANGUAGE"
fi

# Environment

PROJECT_DIR="./lang_transfer/"
T5X_DIR="./t5x"  # directory where the t5x is cloned.
BUCKET_NAME=${BUCKET_NAME:="lang_agnostic_europe"}

export PYTHONPATH="./:./lang_transfer"

# Experiments definition
MODEL_BASE_DIR="gs://${BUCKET_NAME}/models/finetune"
PRETRAINED_MODEL_CHECKPOINT="gs://${BUCKET_NAME}/models/finetune/scratch_${PRETRAINED_LANGUAGE}_${MODEL_SIZE}_600M/checkpoint_11450/"

echo "Running experiment with size 0 (Zero shot). Bucket is ${BUCKET_NAME}" ;

FINETUNE_LANGUAGES=("ar" "de" "en" "es" "fi" "id" "ja" "ko" "pt" "ru" "zh")

for LANGUAGE in "${FINETUNE_LANGUAGES[@]}"; do
    # python3 ${T5X_DIR}/t5x/train.py \
    #     --gin_search_paths=${PROJECT_DIR} \
    #     --gin_file="lang_transfer/configs/runs/zeroshot.${MODEL_SIZE}.gin" \
    #     --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/scratch_${LANGUAGE}_${MODEL_SIZE}_0M\" \
    #     --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.6M"\" \
    #     --gin.VAL_MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\"

    if [ -n "$PRETRAINED_LANGUAGE" ]; then

      TRAIN_STEPS=$((10+11450))  # To account for pretraining steps

      python3 ${T5X_DIR}/t5x/train.py \
          --gin_search_paths=${PROJECT_DIR} \
          --gin_file="lang_transfer/configs/runs/zeroshot.${MODEL_SIZE}.gin" \
          --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/${PRETRAINED_LANGUAGE}_from_600M_${LANGUAGE}_${MODEL_SIZE}_0M\" \
          --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.6M"\" \
          --gin.VAL_MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\" \
          --gin.TRAIN_STEPS=${TRAIN_STEPS} \
          --gin.PRETRAINED_MODEL_PATH=\"${PRETRAINED_MODEL_CHECKPOINT}\"
    fi
done