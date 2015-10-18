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
    support_vectors = svm_model.get_SV()

    detector = [0 for i in range(len(support_vectors[0]))]

    for index, support_vector in enumerate(support_vectors):
        alpha = support_vector_coefficients[index][0]
        for component_index, component in support_vector.iteritems():
            detector[component_index-1] += component * alpha

    detector[-1] = -svm_model.rho.contents.value

    detector_file = open(output, "w")
    for f in detector:
        detector_file.write(str(f)+"\n")
