#!/usr/bin/env python
"""
Collects data from a video source.
"""

import cv2
import argparse
import uuid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a HOG classifier.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("width", metavar="W", type=int, help="width of the HOG window")
    parser.add_argument("height", metavar="H", type=int, help="height of the HOG window")
    parser.add_argument("output", metavar="O", type=str, help="path to save the collected data")
    parser.add_argument("-s", "--source", nargs="?", type=int, default=0, help="camera source")
    parser.add_argument("-m", "--max", nargs="?", type=int, default=40, help="max datasets to collect (0 is unlimited)")

    args = parser.parse_args()

    width = args.width
    height = args.height
    output_dir = args.output
    source = args.source
    max = args.max

    SIZE = (width, height)

    video_capture = cv2.VideoCapture(source)

    video_capture.set(3,width)
    video_capture.set(4,height)

    record = False
    frames_count = 0
    total_frames_count = 0
    while True:
        ret, frame = video_capture.read()

        if record:
            if frames_count < max:
                cv2.imwrite(output_dir+"/"+str(uuid.uuid4())+".png", frame)
                frames_count += 1
                total_frames_count += 1
                print str(total_frames_count) + " framed recorded..."
            else:
                record = False

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            record = True
            frames_count = 0

    video_capture.release()
    cv2.destroyAllWindows()
