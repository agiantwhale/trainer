#!/usr/bin/env bash
set -e

SOURCE_PATH=""
OUTPUT_PATH=""
EXT="JPG"

for FILE in $SOURCE_PATH/*.$EXT;
do
    echo $FILE
    FILENAME=$(basename $FILE .$EXT)
    convert $FILE -auto-orient $OUTPUT_PATH/$FILENAME.$EXT
    convert $FILE -auto-orient -flop $OUTPUT_PATH/$FILENAME\_f.$EXT
    # convert $FILE -auto-orient -motion-blur 20x10+45 $OUTPUT_PATH/$FILENAME\_b.$EXT
    # convert $FILE -auto-orient -motion-blur 20x10+45 -flop $OUTPUT_PATH/$FILENAME\_bf.$EXT
done
