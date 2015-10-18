#!/usr/bin/env python
"""
Extract train set.
"""

import multiprocessing as mp
import cv2
import numpy as np
import os
import random
import argparse

def compute_hog(frame, hog):
    """
    Computes and returns the HOG features.
    :param frame: Image matrix
    :param hog: OpenCV HOG Descriptor
    :return: features
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray)

def mine_image(frame, hog):
    """
    Run negative mining on image, and extract false positives.
    :param frame: Image matrix
    :param hog:  OpenCV HOG Descriptor
    :return: features
    """
    found, w = hog.detectMultiScale(frame)
    features = []
    for rect in found:
        print "\t\t - Mined!"
        x, y, w, h = rect
        roi = frame[y:y+h, x:x+w]
        feature = compute_hog(cv2.resize(roi, SIZE), hog)
        features.append(feature)
    return features

def extract_random_patch(frame, size):
    """
    Extracts a random patch from frame.
    :param frame: Image matrix
    :param size: size of the random patch
    :return: ROI
    """
    frame_h, frame_w = frame.shape[:2]
    width = size[0]
    height = size[1]
    x1 = random.randint(0, frame_w-width-1)
    x2 = x1 + width
    y1 = random.randint(0, frame_h-height-1)
    y2 = y1 + height
    roi = frame[y1:y2, x1:x2]
    return roi.copy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract trainset for SVM",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("width", metavar="W", type=int, help="width of the HOG window")
    parser.add_argument("height", metavar="H", type=int, help="height of the HOG window")
    parser.add_argument("-o", "--output", nargs="?", type=str, default=os.getcwd()+"/train",
                        help="detecting vector output file")
    parser.add_argument("-p", "--positive", nargs="?", type=str, default=os.getcwd()+"/positive",
                        help="positive sample directory")
    parser.add_argument("-n", "--negative", nargs="?", type=str, default=os.getcwd()+"/negative",
                        help="negative sample directory")
    parser.add_argument("-m", "--model", nargs="?", type=str, default=os.getcwd()+"/results.features",
                        help="run negative mining?")

    args = parser.parse_args()

    random.seed()

    width = args.width
    height = args.height
    output = args.output
    model = args.model
    positive_dir = args.positive
    negative_dir = args.negative

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

    # Load the sample paths
    positive_samples_path = [os.path.join(positive_dir,f)
                             for f in os.listdir(positive_dir)
                             if os.path.isfile(os.path.join(positive_dir,f))]
    negative_samples_path = [os.path.join(negative_dir,f)
                             for f in os.listdir(negative_dir)
                             if os.path.isfile(os.path.join(negative_dir,f))]

    if os.path.isfile(model):
        trainset = open(output, "a")

        detector = []
        with open(model, "r") as svm:
            for line in svm:
                detector.append(float(line))
        hog.setSVMDetector(np.array(detector, dtype=np.float32))

        output = mp.Queue()
        processes = []

        print "Applying negative mining..."
        for sample in negative_samples_path:
            print "\t - " + sample
            frame = cv2.imread(sample)
            if frame is None:
                continue
            processes.append(
                mp.Process(target=mine_image, args=(frame, hog))
            )

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        results = [output.get() for p in processes]

        for result in results:
            for feature in result:
                individual = "-1"
                for ind, f in enumerate(feature):
                    individual += (" "+str(ind+1)+":"+str(f[0]))
                individual += "\n"
                trainset.write(individual)
    else:
        trainset = open(output, "w")
        print "Loading samples..."
        for f in positive_samples_path:
            print "\t - " + f
            image = cv2.imread(f)
            if image is None:
                continue
            roi = cv2.resize(image, SIZE)
            feature = compute_hog(roi, hog)
            individual = "+1"
            for ind, f in enumerate(feature):
                individual += (" "+str(ind+1)+":"+str(f[0]))
            individual += "\n"
            trainset.write(individual)

        for f in negative_samples_path:
            print "\t - " + f
            image = cv2.imread(f)
            if image is None:
                continue
            roi = extract_random_patch(image, SIZE)
            feature = compute_hog(roi, hog)
            individual = "-1"
            for ind, f in enumerate(feature):
                individual += (" "+str(ind+1)+":"+str(f[0]))
            individual += "\n"
            trainset.write(individual)
