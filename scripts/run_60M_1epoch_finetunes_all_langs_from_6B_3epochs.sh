# Script arguments

SKIP_SCRATCH=${1}
SPECIFIC_SIZE=${2}
MODEL_SIZE=small


# Environment

PROJECT_DIR="./lang_transfer/"
T5X_DIR="./t5x"  # directory where the t5x is cloned.
BUCKET_NAME=${BUCKET_NAME:="lang_agnostic_europe"}

export PYTHONPATH="./:./lang_transfer"

# Experiments definition
PRETRAINED_LANGUAGES=("ar" "de" "en" "es" "fi" "id" "ja" "ko" "pt" "ru" "zh")
FINETUNE_LANGUAGES=("ar" "de" "en" "es" "fi" "id" "ja" "ko" "pt" "ru" "zh")
FINETUNE_SIZES=("6M" "19M" "60M" "189M" "600M" "6B")
FINETUNE_EPOCH_STEPS=(12 37 115 361 1145 11445)  # number of steps to form an epoch
EPOCHS=(10 10 1 10 10 3)  # number of steps to form an epoch
WARMUP_STEPS=(0 0 30 100 300 3000)
MODEL_BASE_DIR="gs://${BUCKET_NAME}/models/finetune"

RUNS=${#FINETUNE_SIZES[@]}

for PRETRAINED_LANGUAGE in "${PRETRAINED_LANGUAGES[@]}"; do
    PRETRAINED_MODEL_CHECKPOINT="gs://${BUCKET_NAME}/models/finetune/scratch_${PRETRAINED_LANGUAGE}_${MODEL_SIZE}_6B/checkpoint_34335/"

    for LANGUAGE in "${FINETUNE_LANGUAGES[@]}"; do
        for (( i=0; i<$RUNS; i++ )); do
            DATA_SIZE=${FINETUNE_SIZES[$i]}
            EPOCH_STEPS=${FINETUNE_EPOCH_STEPS[$i]}
            EPOCHS_TO_TRAIN=${EPOCHS[$i]}
            WARMUP=${WARMUP_STEPS[$i]}

            TRAIN_STEPS=$((EPOCH_STEPS*EPOCHS_TO_TRAIN))
            EVAL_PERIOD=$((EPOCH_STEPS))

            if [ ! -z "$SPECIFIC_SIZE" ] && [ "$SPECIFIC_SIZE" != "$DATA_SIZE" ]; then
              echo "Skipping size $DATA_SIZE"
              continue
            fi

            echo "Running experiment with size ${DATA_SIZE}, # of train steps ${TRAIN_STEPS}, #warmup ${WARMUP}. Bucket is ${BUCKET_NAME}" ;

            if [ "$SKIP_SCRATCH" -ne "1" ]; then

              python3 ${T5X_DIR}/t5x/train.py \
                  --gin_search_paths=${PROJECT_DIR} \
                  --gin_file="lang_transfer/configs/runs/train_scratch.${MODEL_SIZE}.gin" \
                  --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/scratch_${LANGUAGE}_${MODEL_SIZE}_${DATA_SIZE}\" \
                  --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.${DATA_SIZE}"\" \
                  --gin.VAL_MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\" \
                  --gin.TRAIN_STEPS=${TRAIN_STEPS} \
                  --gin.EVAL_PERIOD=${EVAL_PERIOD} \
                  --gin.WARMUP_STEPS=${WARMUP}
            fi

            if [ -n "$PRETRAINED_LANGUAGE" ]; then

              TRAIN_STEPS=$((TRAIN_STEPS+34335))  # To account for pretraining steps

              python3 ${T5X_DIR}/t5x/train.py \
                  --gin_search_paths=${PROJECT_DIR} \
                  --gin_file="lang_transfer/configs/runs/finetune.${MODEL_SIZE}.gin" \
                  --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/${PRETRAINED_LANGUAGE}_from_6B_3epochs_${LANGUAGE}_${MODEL_SIZE}_${DATA_SIZE}_1epoch\" \
                  --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.${DATA_SIZE}"\" \
                  --gin.VAL_MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\" \
                  --gin.TRAIN_STEPS=${TRAIN_STEPS} \
                  --gin.EVAL_PERIOD=${EVAL_PERIOD} \
                  --gin.WARMUP_STEPS=0 \
                  --gin.PRETRAINED_MODEL_PATH=\"${PRETRAINED_MODEL_CHECKPOINT}\"
            fi
        done
    done
done