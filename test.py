#!/usr/bin/env python
"""
Tests the trained data.
"""

import cv2
import os
import argparse
import cPickle

def draw_detections(img, rects):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a HOG classifier.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("width", metavar="W", type=int, help="width of the HOG window")
    parser.add_argument("height", metavar="H", type=int, help="height of the HOG window")
    parser.add_argument("-o", "--output", nargs="?", type=str, default=os.getcwd()+"/result.model", help="detecting vector output file")
    parser.add_argument("-s", "--source", nargs="?", type=int, default=0, help="camera source")

    args = parser.parse_args()

    width = args.width
    height = args.height
    output = args.output
    source = args.source

    SIZE = (width, height)

    hog = cv2.HOGDescriptor()
    hog.winSize = SIZE
    hog.setSVMDetector(cPickle.load(output))

    video_capture = cv2.VideoCapture(source)

    while True:
        ret, frame = video_capture.read()

        found, w = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        draw_detections(frame, found)

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()
