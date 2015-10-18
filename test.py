#!/usr/bin/env python
"""
Tests the trained data.
"""

import cv2
import numpy as np
import os
import argparse

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
    parser.add_argument("-o", "--output", nargs="?", type=str, default=os.getcwd()+"/result.feature",
                        help="detecting vector output file")
    parser.add_argument("-s", "--source", nargs="?", type=str, default=0, help="test source")

    args = parser.parse_args()

    width = args.width
    height = args.height
    output = args.output
    source = args.source

    SIZE = (width, height)

    winSize = SIZE
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

    detector = []
    with open(output, "r") as model:
        for line in model:
            detector.append(float(line))

    hog.setSVMDetector(np.array(detector, dtype=np.float32))

    image = cv2.imread(source)
    found, w = hog.detectMultiScale(image, winStride=(8,8), padding=(32,32), scale=1.05)
    draw_detections(image, found)
    cv2.imshow("Test",image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
