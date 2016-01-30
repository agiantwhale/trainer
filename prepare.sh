#!/usr/bin/env bash
set -e

SOURCE_PATH="/Users/iljae/Development/nav-train/roomba/train/positive"
OUTPUT_PATH="/Users/iljae/Development/nav-train/roomba/train/positive_sel"
EXT="JPG"

for FILE in $SOURCE_PATH/*.$EXT;
do
    echo $FILE
    FILENAME=$(basename $FILE .$EXT)
    convert $FILE -auto-orient $OUTPUT_PATH/$FILENAME.$EXT
    convert $FILE -auto-orient -flop $OUTPUT_PATH/$FILENAME\_f.$EXT
    convert $FILE -auto-orient -radial-blur 8 $OUTPUT_PATH/$FILENAME\_b.$EXT
    convert $FILE -auto-orient -radial-blur 8 -flop $OUTPUT_PATH/$FILENAME\_bf.$EXT
done
