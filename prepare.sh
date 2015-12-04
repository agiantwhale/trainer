#!/usr/bin/env bash
set -e

SOURCE_PATH=""
OUTPUT_PATH=""
EXT="JPG"

for FILE in $SOURCE_PATH/*.$EXT;
do
    echo $FILE
    FILENAME=$(basename $FILE .$EXT)
    cp $FILE $OUTPUT_PATH/$FILENAME.$EXT
    convert $FILE -flip $OUTPUT_PATH/$FILENAME\_f.$EXT
    convert $FILE -motion-blur 0x12+200 $OUTPUT_PATH/$FILENAME\_b.$EXT
    convert $FILE -motion-blur 0x12-200 $OUTPUT_PATH/$FILENAME\_b.$EXT
    convert $FILE -motion-blur 0x12+200 -flip $OUTPUT_PATH/$FILENAME\_bf.$EXT
    convert $FILE -motion-blur 0x12-200 -flip $OUTPUT_PATH/$FILENAME\_bf.$EXT
done
