#!/usr/bin/env python
"""
Trains an SVM
"""

import cv2
import numpy as np
import os
import random

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
    frame_w, frame_h = frame.shape[:2]
    x1 = random.randint(0, frame_w-width)
    x2 = x1 + width
    y1 = random.randint(0, frame_h-height)
    y2 = y1 + height
    roi = frame[y1:y2, x1:x2]
    return roi

if __name__ == "__main__":
    random.seed()

    width = 72
    height = 128
    output_dir = ""
    positive_dir = ""
    negative_dir = ""
    k_value = 10

    SIZE = (width, height)

    hog = cv2.HOGDescriptor()
    hog.winSize = SIZE

    # Load the sample paths
    positive_samples_path = [f for f in os.listdir(positive_dir) if os.path.isfile(os.path.join(positive_dir,f))]
    negative_samples_path = [f for f in os.listdir(negative_dir) if os.path.isfile(os.path.join(negative_dir,f))]

    # Load the samples
    positive_features = [
        compute_hog(cv2.resize(cv2.imread(f), SIZE), hog)
        for f in positive_samples_path]
    negative_features = [
        compute_hog(extract_random_patch(cv2.imread(f), SIZE), hog)
        for f in negative_samples_path]

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
