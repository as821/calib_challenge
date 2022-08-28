import cv2
import os
import numpy as np
import argparse
import math
import sys

# Maximum acceptable L2 distance (in pixels) between pairs of points in a match
L2_DIST_THRESH = 20

# Number of features to generate
NUM_FEAT_GENREATE = 2000

# Number of top matches to process (ignore the rest)
NUM_MATCHES_FILTER = 500




class VideoProcessor:
    def __init__(self):
        """Parse arguments, initialize and return VideoCapture object"""
        # Parse args
        parser = argparse.ArgumentParser()
        parser.add_argument('path', type=str, help="path to input .hecv file")
        args = parser.parse_args()

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
        self.orb = cv2.ORB_create(NUM_FEAT_GENREATE)
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
        self.frame_cnt += 1 
        return frame, self.frame_cnt


    def _convert_int(self, point):
        u, v = map(lambda x: int(round(x)), point)
        return (u, v)

    def filter_matches(self, curr_keyp, matches, frame):
        def l2_distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        # Remove matches with large L2 distances. This inherently assumes nothing is moving super fast
        # through the frames. Handles reflections on hood at night, etc.
        output = []
        max_l2_dist = -1
        for m in matches:
            if m.queryIdx < len(curr_keyp) and m.queryIdx < len(self.prev_frame_feat[0]): 
                p1 = self._convert_int(curr_keyp[m.queryIdx].pt)
                p2 = self._convert_int(self.prev_frame_feat[0][m.queryIdx].pt)
                dist = l2_distance(p1, p2)
                if dist < L2_DIST_THRESH:
                    output.append((p1, p2, m.distance, dist))
                    max_l2_dist = max(max_l2_dist, dist)

                    # Draw line to represent the match
                    cv2.line(frame, p1, p2, [0, 255, 0], 2)
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) == ord('q'):
                        break

        foo = "ERROR" if len(output) < NUM_MATCHES_FILTER else ""
        print(foo, "\t", max_l2_dist, "\t", len(output))

        # Sort by Hamming distance and return top matches
        return sorted(output, key=lambda x:x[2])[:NUM_MATCHES_FILTER]


    def calc_avg_direction(self, matches, keyp):
        """Calculate direction of the average change between match points"""
        avg_vec = [0, 0]
        for m in matches:
            # Calculate vector between points of the match
            diff_vec = [0, 0]
            diff_vec[0] = keyp[m.trainIdx].pt[0] - self.prev_frame_feat[0][m.queryIdx].pt[0]
            diff_vec[1] = keyp[m.trainIdx].pt[1] - self.prev_frame_feat[0][m.queryIdx].pt[1]

            # Normalize vector to unit length
            norm = diff_vec[0]**2 + diff_vec[1]**2
            if norm > 0:
                diff_vec[0] /= norm
                diff_vec[1] /= norm

            # Accumulation
            avg_vec[0] += diff_vec[0]
            avg_vec[1] += diff_vec[1]

        # TODO think more about this, the magnitude of the components of these vectors matters... maybe have to calculate angle first, then average??
        avg_vec[0] /= len(matches)
        avg_vec[1] /= len(matches)
        return avg_vec


    def calc_angles(self, obs_vec, focal_len=910):
        """Calculate pitch and yaw angles of the observed vector, with respect to the reference vector"""
        def angle(im_len, focal):
            """Trigonometry to calculate camera angle using observed vector on the image plane and the focal length"""
            return math.sin(im_len/focal)
        # TODO double check this math, especially sine usage
        return angle(obs_vec[0], focal_len), angle(obs_vec[1], focal_len)

    def process_single_frame(self):
        """Perform processing on the next video frame"""
        # Get next frame
        frame, cnt = self._get_next_frame()
        if frame is None:
            return None

        # Covert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         
        # Detect ORB features in current frame
        keyp, desc = self.orb.detectAndCompute(frame, None)
        #for point in keyp:
            #u, v = map(lambda x: int(round(x)), point.pt)
            #cv2.circle(frame, (u, v), color=(0, 255, 0), radius=3)

        # Find matches with previous frame, if it exists
        if self.prev_frame_feat is not None: 
            # Match descriptors, select best matches
            matches = self.fm.match(self.prev_frame_feat[1],desc)  
            filtered = self.filter_matches(keyp, matches, frame)
            
            # (DEBUG) Output matches on screen
            # for m in filtered:
            #     # Draw keypoints and line to represent the match
            #     cv2.circle(frame, m[0], color=(0, 255, 0), radius=3)
            #     cv2.circle(frame, m[1], color=(0, 255, 0), radius=3)
            #     cv2.line(frame, m[0], m[1], [0, 255, 0], 2)
            
            # Calculate the average direction change
            #avg_vec = self.calc_avg_direction(matches, keyp) 

            # ASSUMPTION: car is driving straight, direction change is with respect to zero vector

            # Calculate pitch and yaw angles
            #pitch, yaw = self.calc_angles(avg_vec)
            #print("Pitch: ", pitch, ", yaw: ", yaw, ", avg_vec: ", avg_vec)


        self.prev_frame_feat = (keyp, desc)

        return frame


def main():
    # Iterate through video, processing frame by frame
    vp = VideoProcessor()
    while True:
        # Process frame
        frame = vp.process_single_frame()
        if frame is None:
           break

        #Display resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
           break

    # Cleanup
    print("Exiting...")
    vp.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()







