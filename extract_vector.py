#!/usr/bin/env python
"""
Extract single detecting vector from trained SVM file.
"""

import argparse
import svmutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract detector from SVM model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model", metavar="M", type=str, help="path to model file")
    parser.add_argument("output", metavar="O", type=str, help="path to output file")

    args = parser.parse_args()

    model = args.model
    output = args.output

    svm_model = svmutil.svm_load_model(model)

    support_vector_coefficients = svm_model.get_sv_coef()
    support_vectors = svm_model.get_sv()

    detector = []

    for index, support_vector in enumerate(support_vectors):
        alpha = support_vector_coefficients[0][index]
        for component_index, component in enumerate(support_vector):
            if index == 0:
                detector.append(component.value * alpha)
            else:
                detector[component_index] += component.value * alpha
