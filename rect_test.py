import cv2
import os
import numpy as np
import argparse
import sys
import math

from rectification import rectify_image, get_homography

from scipy.spatial.transform import Rotation as rotation





# TODO --> using HAMMING distance, additional filtering should be done with L2
#DIM_DIST_THRESH = 50
#MATCH_DIST_THRESH = 60


class VideoProcessor:
    def __init__(self):
        """Parse arguments, initialize and return VideoCapture object"""
        # Parse args
        parser = argparse.ArgumentParser()
        parser.add_argument('path', type=str, help="path to input .hecv file")
        args = parser.parse_args()
        path, self.label_file = args.path.rsplit("/")
        self.label_file = path + "/" + self.label_file.split('.')[0] + ".txt"

        # Load video
        self.cap = cv2.VideoCapture(args.path)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        # Load labels, if applicable
        if "labeled" in args.path:
            with open(args.path.rsplit(".")[0] + ".txt") as fd:
                self.labels = [l.split(" ") for l in fd.read().split("\n") if l != '']
                self.labels = [(float(l[0]), float(l[1])) for l in self.labels]
        else:
            self.labels = None 

        # Initialize ORB 
        self.orb = cv2.ORB_create(2000)
        self.prev_frame_feat = None
        self.frame_cnt = -1

        # Initialize feature matcher
        self.fm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _get_next_frame(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None
        return frame


    def _homography_to_angles(self, homography, frame):
        f_x = len(frame) // 2
        f_y = len(frame[0]) // 2 
        intrinsic = np.array([[910, 0, f_x], [0, 910, f_y], [0, 0, 1]])
        ret, rot, trans, normal = cv2.decomposeHomographyMat(homography, intrinsic)
        output = set()
        for rot_mat in rot:
            rotate = rotation.from_matrix(rot_mat)
            pitch, roll, yaw = rotate.as_euler('XYZ', degrees=False)     # pitch, roll, yaw
            output.add((pitch, yaw))
            print(pitch, yaw)      
        return min(output, key=lambda x: abs(x[0]) + abs(x[1]))
        


    def process_single_frame(self):
        """Perform processing on the next video frame"""
        # Get next frame
        frame = self._get_next_frame()
        if frame is None:
            return None

        # Covert to grayscale
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate homography
        homography = get_homography(frame, 4, algorithm='independent')

        # Decompose homography into roll/pitch


        return self._homography_to_angles(homography, frame) #rectify_image(frame, 4, algorithm='3-line')


def main():
    # Iterate through video, processing frame by frame
    print("Creating VideoProcessor...")
    vp = VideoProcessor()
    with open(vp.label_file, "r") as fd:
        print("Entering infinite loop...")
        while True:
            # Process frame
            frame = vp.process_single_frame()
            if frame is None:
                break

            print("RESULT: ", frame, end="\t")
            correct = [float(i) for i in fd.readline().lstrip().rstrip().split(' ')]
            print("CORRECT: ", correct)

            # Display resulting frame
            #cv2.imshow('frame', frame)
            #if cv2.waitKey(1) == ord('q'):
            #    break

    # Cleanup
    print("Exiting...")
    vp.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    np.seterr(all='raise')
    main()







