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
import functools

def mine_image(detector, size, path):
    """
    Run negative mining on image, and extract false positives.
    :param path: Path to image file
    :param hog: OpenCV HOG Descriptor
    :return: list of features
    """
    winSize = size
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    print size
    print len(detector)
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    hog.setSVMDetector(np.array(detector, dtype=np.float32))

    frame = cv2.imread(path)
    if frame is None:
        return []
    print "\t - Mining " + path
    found, w = hog.detectMultiScale(frame)
    features = []
    for rect in found:
        print "\t\t - Mined!"
        x, y, w, h = rect
        roi = frame[y:y+h, x:x+w]
        feature = compute_hog(cv2.resize(roi, hog.winSize), hog)
        features.append(feature)
    return features

def extract_positive_features(size, path):
    """
    Extract positive features from an image
    :param path: Path to image file
    :param hog: OpenCV HOG Descriptor
    :return: list of features
    """
    winSize = size
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

    frame = cv2.imread(path)
    if frame is None:
        return None
    print "\t - Loading " + path
    return compute_hog(cv2.resize(frame, hog.winSize), hog)

def extract_negative_features(size, path):
    """
    Extract negative features from an image
    :param path: Path to image file
    :param hog: OpenCV HOG Descriptor
    :return: list of features
    """
    winSize = size
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

    frame = cv2.imread(path)
    if frame is None:
        return None
    print "\t - Loading " + path
    return compute_hog(extract_random_patch(frame, hog.winSize), hog)

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

    # Load the sample paths
    positive_samples_path = [os.path.join(positive_dir,f)
                             for f in os.listdir(positive_dir)
                             if os.path.isfile(os.path.join(positive_dir,f))]
    negative_samples_path = [os.path.join(negative_dir,f)
                             for f in os.listdir(negative_dir)
                             if os.path.isfile(os.path.join(negative_dir,f))]

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(mp.cpu_count())

    if os.path.isfile(model):
        neg_mining = True
        trainset = open(output, "a")
        detector = [float(line) for line in open(model, "r")]
        print "Applying negative mining..."
    else:
        neg_mining = False
        trainset = open(output, "w")
        print "Loading samples..."

    if neg_mining:
        func = functools.partial(mine_image, detector, SIZE)
        for result in pool.imap(func, [path for path in negative_samples_path]):
            for feature in result:
                individual = "-1"
                for ind, f in enumerate(feature):
                    individual += (" "+str(ind+1)+":"+str(f[0]))
                individual += "\n"
                trainset.write(individual)
    else:
        pos_func = functools.partial(extract_positive_features, SIZE)
        for feature in pool.imap(pos_func, [path for path in positive_samples_path]):
            if feature is None:
                continue
            individual = "+1"
            for ind, f in enumerate(feature):
                individual += (" "+str(ind+1)+":"+str(f[0]))
            individual += "\n"
            trainset.write(individual)

        neg_func = functools.partial(extract_negative_features, SIZE)
        for feature in pool.imap(neg_func, [path for path in negative_samples_path]):
            if feature is None:
                continue
            individual = "-1"
            for ind, f in enumerate(feature):
                individual += (" "+str(ind+1)+":"+str(f[0]))
            individual += "\n"
            trainset.write(individual)
