#!/usr/bin/env python
"""
Tests the trained data.
"""

import cv2
import numpy as np
import os
import time
import argparse

def draw_detections(img, rects):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a HOG classifier.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("width", metavar="W", type=int, help="width of the HOG window")
    parser.add_argument("height", metavar="H", type=int, help="height of the HOG window")
    parser.add_argument("-o", "--output", nargs="?", type=str, default=os.getcwd()+"/result.feature",
                        help="detecting vector output file")
    parser.add_argument("-s", "--source", nargs="?", type=str, default=0, help="test source")
    parser.add_argument("-t", "--scale", nargs="?", type=float, default=1, help="scale output")

    args = parser.parse_args()

    width = args.width
    height = args.height
    output = args.output
    source = args.source
    scale = args.scale

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

    try:
        feed_source = int(source)
    except ValueError:
        feed_source = source
    cap = cv2.VideoCapture(feed_source)
    while True:
        ret, image = cap.read()
        imheight, imwidth, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        found, w = hog.detectMultiScale(gray)
        fps = 1 / (time.time() - start_time)
        print "FPS:" , fps, "-", len(found), "roombas detected"
        draw_detections(image, found)
        image = cv2.resize(image, (int(imwidth * scale), int(imheight * scale)))
        cv2.imshow("Test", image)
        if cv2.waitKey(1) == 27:
            break

    cv2.waitKey()
    cv2.destroyAllWindows()
