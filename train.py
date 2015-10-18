#!/usr/bin/env python
"""
Trains an SVM
"""

import cv2
import numpy as np
import os
import random
import argparse
import cPickle

def train_svm(positive, negative, k):
    """
    Trains and SVM and returns the detecting vector.
    :param positive: Positive feature sets
    :param negative: Negative feature sets
    :param k: k-values
    :return: Detecting vector
    """
    svm = cv2.SVM()
    params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC)
    samples = []
    samples.extend(positive)
    samples.extend(negative)
    x = np.array(samples, dtype=np.float32)
    results = [1. if i < len(positive) else 0. for i in range(len(samples))]
    y = np.array(results, dtype=np.float32)
    svm.train_auto(x, y, None, None, params, k)
    return svm.get_support_vector_count()

def compute_hog(frame, hog):
    """
    Computes and returns the HOG features.
    :param frame: Image matrix
    :param hog: OpenCV HOG Descriptor
    :return: features
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray)

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
    parser = argparse.ArgumentParser(description="Train a HOG classifier.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("width", metavar="W", type=int, help="width of the HOG window")
    parser.add_argument("height", metavar="H", type=int, help="height of the HOG window")
    parser.add_argument("-k", "--kfolds", nargs="?", type=int, default=10, help="k-cross validation count")
    parser.add_argument("-o", "--output", nargs="?", type=str, default=os.getcwd()+"/result.model", help="detecting vector output file")
    parser.add_argument("-p", "--positive", nargs="?", type=str, default=os.getcwd()+"/positive", help="positive sample directory")
    parser.add_argument("-n", "--negative", nargs="?", type=str, default=os.getcwd()+"/negative", help="negative sample directory")

    args = parser.parse_args()

    random.seed()

    width = args.width
    height = args.height
    k_value = args.kfolds
    output = args.output
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

    # Load the samples
    positive_features = []
    negative_features = []

    for f in positive_samples_path:
        image = cv2.imread(f)
        if image is None:
            continue
        roi = cv2.resize(image, SIZE)
        positive_features.append(compute_hog(roi, hog))

    for f in negative_samples_path:
        print f
        image = cv2.imread(f)
        if image is None:
            continue
        roi = extract_random_patch(image, SIZE)
        negative_features.append(compute_hog(roi, hog))

    # Train the SVM
    detector = train_svm(positive_features, negative_features, k_value)
    hog.setSVMDetector(detector)

    # Run negative mining
    for sample in negative_samples_path:
        frame = cv2.imread(sample)
        found, w = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        for rect in found:
            roi = frame[rect.y:rect.y+rect.h, rect.x:rect.x+rect.w]
            feature = compute_hog(cv2.resize(roi, SIZE), hog)
            negative_features.append(feature)

    # Run detection again
    detector = train_svm(positive_features, negative_features, k_value)
    cPickle.dump(detector, output)
