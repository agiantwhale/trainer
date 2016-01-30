#!/usr/bin/env bash
set -e

POS_PATH=""
NEG_PATH=""
OUTPUT_PATH=""
MODEL_NAME=""
WIDTH="64"
HEIGHT="64"
TRAINER_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="$PYTHONPATH:$LIBSVM_PATH/python"
# Build data from pos/neg path
python "$TRAINER_PATH/extract_trainset.py" \
  -o "$OUTPUT_PATH/$MODEL_NAME" \
  -p $POS_PATH \
  -n $NEG_PATH $WIDTH $HEIGHT

# Train
svm-train -c 0.01 -s 3 -t 0 \
  "$OUTPUT_PATH/$MODEL_NAME" \
  "$OUTPUT_PATH/$MODEL_NAME.model"

# Extract detector
python \
  "$TRAINER_PATH/extract_vector.py" \
  "$OUTPUT_PATH/$MODEL_NAME.model" \
  "$OUTPUT_PATH/$MODEL_NAME.features"

# Neg mine
for i in {1..10} # 10 times is probably an overkill...
do
  python "$TRAINER_PATH/extract_trainset.py" \
    -o "$OUTPUT_PATH/$MODEL_NAME" \
    -p $POS_PATH \
    -n $NEG_PATH \
    -m "$OUTPUT_PATH/$MODEL_NAME.features" $WIDTH $HEIGHT

  # Train
  svm-train -c 0.01 -s 3 -t 0 \
    "$OUTPUT_PATH/$MODEL_NAME" \
    "$OUTPUT_PATH/$MODEL_NAME.model"

  python \
    "$TRAINER_PATH/extract_vector.py" \
    "$OUTPUT_PATH/$MODEL_NAME.model" \
    "$OUTPUT_PATH/$MODEL_NAME.features"
done
