#!/usr/bin/env bash
set -e

while [[ $# > 1 ]]
do
key="$1"

case $key in
    -s|--svmpath)
    LIBSVM_PATH="$2"
    shift # past argument
    ;;
    -p|--pospath)
    POS_PATH="$2"
    shift # past argument
    ;;
    -n|--negpath)
    NEG_PATH="$2"
    shift # past argument
    ;;
    -o|--outputpath)
    OUTPUT_PATH="$2"
    shift # past argument
    ;;
    -m|--modelname)
    MODEL_NAME="$2"
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

TRAINER_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="$PYTHONPATH:$LIBSVM_PATH/python"
# Build data from pos/neg path
python "$TRAINER_PATH/extract_trainset.py" \
  -o "$OUTPUT_PATH/$MODEL_NAME" \
  -p $POS_PATH \
  -n $NEG_PATH 64 64

# Train
"$LIBSVM_PATH/svm-train" -c 0.01 -s 3 -t 0 \
  "$OUTPUT_PATH/$MODEL_NAME" \
  "$OUTPUT_PATH/$MODEL_NAME.model"

# Extract detector
python \
  "$TRAINER_PATH/extract_vector.py" \
  "$OUTPUT_PATH/$MODEL_NAME.model" \
  "$OUTPUT_PATH/$MODEL_NAME.features"

# Neg mine
for i in {1..10}
do
  python "$TRAINER_PATH/extract_trainset.py" \
    -o "$OUTPUT_PATH/$MODEL_NAME" \
    -p $POS_PATH \
    -n $NEG_PATH \
    -m "$OUTPUT_PATH/$MODEL_NAME.features" 64 64

  # Train
  "$LIBSVM_PATH/svm-train" -c 0.01 -s 3 -t 0 \
    "$OUTPUT_PATH/$MODEL_NAME" \
    "$OUTPUT_PATH/$MODEL_NAME.model"

  python \
    "$TRAINER_PATH/extract_vector.py" \
    "$OUTPUT_PATH/$MODEL_NAME.model" \
    "$OUTPUT_PATH/$MODEL_NAME.features"
done
