#!/usr/bin/env python

from random import sample
from re import I
import sys
import numpy as np
import cv2 as cv
import math
import argparse
import os

FOCAL_LEN = 910


def main():
     # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to input .hecv file")
    parser.add_argument('ground', type=str, help="path to input labels .txt file")
    parser.add_argument('--generated', type=str, default="", help="path to generate labels .txt file")
    args = parser.parse_args()
    cam = cv.VideoCapture(args.path)

    # Load ground truth labels
    with open(args.ground) as fd:
        gt = [l.split(" ") for l in fd.read().split("\n") if l != '']
        gt = [(float(l[0]), float(l[1])) for l in gt]

    # Load generated lables
    if args.generated != "":
        with open(args.generated) as fd:
            gen = [l.split(" ") for l in fd.read().split("\n") if l != '']
            gen = [(float(l[0]), float(l[1])) for l in gen]

    # Run video loop over frames
    frame_idx = 0
    while True:
        # Get next video frame
        _ret, frame = cam.read()
        if frame is None or not _ret:
            break
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw center of image
        im_center = [int(w // 2), int(h //2)]
        cv.circle(vis, im_center, 4, (0, 255, 0), -1)

        # Draw generated point (and line from center to it)
        if args.generated != "":
            gen_label = [0, 0]
            gen_label[0] = math.tan(gen[frame_idx][0]) * FOCAL_LEN + im_center[0]
            gen_label[1] = im_center[1] - math.tan(gen[frame_idx][1]) * FOCAL_LEN
            cv.circle(vis, (int(gen_label[0]), int(gen_label[1])), 10, (255, 0, 0), -1)
            cv.line(vis, (int(gen_label[0]), int(gen_label[1])), im_center, [255, 0, 0], 3)
            print("Gen label: ", gen[frame_idx], end='\t')

        # Draw true label point (and line from center to it)
        if not math.isnan(gt[frame_idx][0]):
            correct = [0, 0]
            correct[0] = math.tan(gt[frame_idx][0]) * FOCAL_LEN + im_center[0]
            correct[1] = im_center[1] - math.tan(gt[frame_idx][1]) * FOCAL_LEN
            cv.circle(vis, (int(correct[0]), int(correct[1])), 10, (0, 0, 255), -1)
            cv.line(vis, (int(correct[0]), int(correct[1])), im_center, [0, 0, 255], 3)
            print("GT: ", gt[frame_idx], end='')
        print('', end="\n")

        # Visualize frame
        cv.imshow('lk_track', vis)
        if cv.waitKey(1) == 27:     # Stop if ESC is pressed
            break
        frame_idx += 1
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
