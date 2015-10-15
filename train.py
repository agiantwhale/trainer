#!/usr/bin/env python
"""
Trains an SVM
"""
import cv2
import numpy
import os

if __name__ == "__main__":
    width = 72
    height = 128
    output_dir = ""
    positive_dir = ""
    negative_dir = ""

    negative_samples = []
    for root, dirs, files in os.walk(negative_dir):
        negative_samples.append()

    for root, dirs, files in os.walk(positive_dir):
        for target in dirs:
            for root, dirs, files in os.walk(os.path.join(root, target)):
                break
