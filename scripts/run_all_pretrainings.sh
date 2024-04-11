# Script arguments

SPECIFIC_DATA_SIZE=${1}
SPECIFIC_LANGUAGE=${2}
MODEL_SIZE=${3}  # small and 550M

# Environment

PROJECT_DIR="./lang_transfer/"
T5X_DIR="./t5x"  # directory where the t5x is cloned.
BUCKET_NAME=${BUCKET_NAME:="lang_agnostic"}

export PYTHONPATH="./:./lang_transfer"

# Experiments definition
LANGUAGES=("ar" "de" "en" "es" "fi" "id" "ja" "ko" "pt" "ru" "zh")
DATASET_SIZES=("6M" "19M" "60M" "189M" "600M" "6B")
EPOCH_STEPS=(12 37 115 361 1145 11445)  # number of steps to form an epoch
EPOCHS=(10 10 10 10 10 10)  # number of steps to form an epoch
WARMUP_STEPS=(0 0 30 100 300 3000)
MODEL_BASE_DIR="gs://${BUCKET_NAME}/models/finetune"
NUM_FINETUNES=${#DATASET_SIZES[@]}


for LANGUAGE in "${LANGUAGES[@]}"; do

    if [ ! -z "$SPECIFIC_LANGUAGE" ] && [ "$SPECIFIC_LANGUAGE" != "$LANGUAGE" ]; then
          echo "Skipping pretrained language $LANGUAGE"
          continue
    fi

    for (( i=0; i<$NUM_FINETUNES; i++ )); do
        DATA_SIZE=${DATASET_SIZES[$i]}
        EPOCH_STEPS=${EPOCH_STEPS[$i]}
        EPOCHS_TO_TRAIN=${EPOCHS[$i]}
        WARMUP=${WARMUP_STEPS[$i]}

        TRAIN_STEPS=$((EPOCH_STEPS*EPOCHS_TO_TRAIN))
        EVAL_PERIOD=$((EPOCH_STEPS))

        if [ ! -z "$SPECIFIC_DATA_SIZE" ] && [ "$SPECIFIC_DATA_SIZE" != "$DATA_SIZE" ]; then
            echo "Skipping size $DATA_SIZE"
            continue
        fi

        echo "Running experiment with size ${DATA_SIZE}, # of train steps ${TRAIN_STEPS}, #warmup ${WARMUP}. Bucket is ${BUCKET_NAME}" ;

        python3 ${T5X_DIR}/t5x/train.py \
            --gin_search_paths=${PROJECT_DIR} \
            --gin_file="lang_transfer/configs/runs/train_scratch.${MODEL_SIZE}.gin" \
            --gin.MODEL_DIR=\"${MODEL_BASE_DIR}/scratch_${LANGUAGE}_${MODEL_SIZE}_${DATA_SIZE}_${EPOCHS_TO_TRAIN}epochs\" \
            --gin.MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.${DATA_SIZE}"\" \
            --gin.VAL_MIXTURE_OR_TASK_NAME=\""langagnostic.${LANGUAGE}.validation"\" \
            --gin.TRAIN_STEPS=${TRAIN_STEPS} \
            --gin.EVAL_PERIOD=${EVAL_PERIOD} \
            --gin.WARMUP_STEPS=${WARMUP}

    done
done