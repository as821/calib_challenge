#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

# TODO play with hyperparameters (VOTER_TOLERANCE), maybe add a moving average filter/smoothing, EKF
# TODO check error
# TODO clean up code (remove RANSAC (?), clean up interfaces and class, etc.)
# TODO add vanishing point along the lines of that paper and combine with this to get below 25%



from random import sample
import sys
import numpy as np
import cv2 as cv
import math
import argparse
import os

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.01, #0.3,
                       minDistance = 25,
                       blockSize = 7 )


# Length of feature history to store, in video frames
TRACK_LEN = 3 # 2

# Frequency at which feature detection should be run, in number of frames
FEAT_DETECT_FREQ = 1

# Number of RANSAC iterations to run
RANSAC_ITER = 1000

# Flag to enable selection of intersection points closest to the image center for RANSAC. False for random sampling
RANSAC_PROX_SAMPLE = True

# Proportion of intersections closest to the center of the image to sample from 
RANSAC_PROX_PERC = 0.75

# Acceptable L2 distance in pixels between proposed model and voter
VOTER_TOLERANCE = 20

# Focal length of camera in pixels
FOCAL_LEN = 910

# True to plot full-image lines, False to plot squiggles 
PLOT_LINES = False

# Set to True for debugging output + visualization, False to output only pitch, yaw in output format expected by evaluation script
DEBUG = True


class OpticalFlow:
    def __init__(self):
        # Initialization
        self.track_len = TRACK_LEN
        self.detect_interval = FEAT_DETECT_FREQ
        self.tracks = []
        self.frame_idx = 0
        self.w = self.h = -1
        self.prev_pred = None

        # Parse args
        parser = argparse.ArgumentParser()
        parser.add_argument('path', type=str, help="path to input .hecv file")
        parser.add_argument('--out', type=str, default="", help="path to output .txt file")
        args = parser.parse_args()
        self.cam = cv.VideoCapture(args.path)

        # Load labels, if applicable
        if "labeled" in args.path and os.path.isfile(args.path.rsplit(".")[0] + ".txt"):
            with open(args.path.rsplit(".")[0] + ".txt") as fd:
                self.labels = [l.split(" ") for l in fd.read().split("\n") if l != '']
                self.labels = [(float(l[0]), float(l[1])) for l in self.labels]
        else:
            self.labels = None 
        
        # Output file set up
        self.out = ""
        if args.out != "":
            self.out = args.out
            open(self.out, 'w').close()
        
    def run(self):
        """Run optical flow algorithm on the video clip."""

        while True:
            # Get next video frame
            _ret, frame = self.cam.read()
            if frame is None:
                break
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            self.h, self.w = vis.shape[:2]

            if len(self.tracks) > 0:
                # Run optical flow algorithm
                p1, good = self._calc_opt_flow(frame_gray)
                
                # Update tracks
                self._track_update(p1, good, vis)

                # Run RANSAC algorithm to find primary vanishing point in the image
                # vp = self._find_optimal_intersection()
                vp, vis = self._ransac_intersection(vis)

                # Determine pitch and yaw angles
                pitch, yaw = self._calc_angles(vp)

                if DEBUG:
                    if self.labels is not None:
                        print("\tPitch/yaw: " + str(pitch) + "/" + str(yaw)  + "\t(" + str(self.labels[self.frame_idx][0]) + "/" + str(self.labels[self.frame_idx][0]) + ")")
                    else:
                        print("\tPitch/yaw: " + str(pitch) + "/" + str(yaw))

                    # Plot tracks
                    self._plotting(vis, vp, pitch, yaw)

                # Write estimation to output file
                if self.out != "":
                    if self.frame_idx % 100 == 1:
                        print("Frame: ", self.frame_idx)
                    with open(self.out, "a") as fd:
                        fd.write(str(pitch) + " " + str(yaw) + "\n")

            if self.frame_idx % self.detect_interval == 0:
                # Re-detect features periodically using Shi-Tomasi corner detection
                self._feature_detection(frame_gray)

            # Display frame and tracked features
            self.frame_idx += 1
            self.prev_gray = frame_gray
            if DEBUG:
                cv.imshow('lk_track', vis)
                if cv.waitKey(1) == 27:     # Stop if ESC is pressed
                    break

    def _calc_angles(self, vp):
        # Normalize vanishing point with respect to the image center
        center = (self.w // 2, self.h // 2)

        # Due to origin of image plane being in top-left corner, need to subtract 
        # center and vp coordinates differently in order for domain of arctan to work correctly
        normed = (vp[0] - center[0], center[1] - vp[1])

        # Calculate horizontal angle (yaw)
        horiz = math.atan(normed[0] / FOCAL_LEN)

        # Calculate vertical angle (pitch)
        vert = math.atan(normed[1] / FOCAL_LEN)

        return vert, horiz

    def _l2_dist(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _compute_votes(self, lines, ind1, ind2, inter_map):
        """Calculate consensus for the proposed line intersection model. Model is an intersection of two 
        lines, so we iterate through all remaining third lines. The line intersection model should be 
        representative as possible of the general line intersections, so if the intersection of the third line
        with either of the two model lines is within the threshold of the model intersection, we count its vote 
        in favor of this model. Intersection with only one of the lines is required due to the possibility of it 
        being parallel with one of the model lines.
        
        Return number of votes, model.""" 

        # Get current model (solve for intersection of these lines)
        intersection = inter_map[tuple(sorted([ind1, ind2]))]

        # Reject degenerate/out of scope candidate models
        if intersection[0] < 0 or intersection[1] < 0 or intersection[0] > self.w or intersection[1] > self.h:
            return 0, None

        # Determine consensus
        votes = 0
        for ind3 in range(len(lines)):
            if ind3 == ind1 or ind3 == ind2:
                # Allow lines to vote in favor of their own intersection
                votes += 1
                continue
            l1_l3_tup = tuple(sorted([ind1, ind3]))
            l1_l2_tup = tuple(sorted([ind2, ind3]))

            if self._l2_dist(inter_map[l1_l3_tup], intersection) <= VOTER_TOLERANCE or self._l2_dist(inter_map[l1_l2_tup], intersection) <= VOTER_TOLERANCE:
                votes += 1         
        return votes, intersection

    def _calc_all_intersections(self, lines):
        """Return dictionary of sets, map index pair to their line intersection"""
        output = {}
        for ind1 in range(len(lines)-1):
            for ind2 in range(ind1, len(lines)):
                l1 = lines[ind1]
                l2 = lines[ind2]
                intersection = [0, 0]
                if l1[0] != l2[0]:
                    intersection[0] = (l2[1] - l1[1]) / (l1[0] - l2[0])
                    intersection[1] = l1[0] * intersection[0] + l1[1]
                output[(ind1, ind2)] = intersection
        return output

    def _get_lines(self):
        # Get valid lines from stored tracks
        lines = []
        for tr in self.tracks:
            p1 = tr[0]
            p2 = tr[-1]

            # Ignore vertical lines (for now)
            if p1[0] != p2[0]:
                # Calculate slope
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])

                # Calculate y-intercept and store results
                lines.append((slope, p1[1] - (p1[0] * slope)))
        return lines

    def _find_optimal_intersection(self):
        lines = self._get_lines()
        best_model = []
        best_votes = 0

        # Precompute all line intersections
        if DEBUG:
            print("Pre-calc, ", end='')
        inter_map = self._calc_all_intersections(lines)

        # Determine intersection with most consensus
        if DEBUG:
            print("Voting... ", end='')
        for ind1 in range(len(lines)-1):
            for ind2 in range(ind1+1, len(lines)):
                current_votes, curr_intersection = self._compute_votes(lines, ind1, ind2, inter_map)
                if current_votes >= best_votes:
                    # Handle multiple intersections with the same number of votes
                    if current_votes > best_votes:
                        best_model = [curr_intersection]
                        best_votes = current_votes
                    best_model.append(curr_intersection)

                    
        # Average the models that receive the most number of votes
        best_out = [0, 0]
        for mod in best_model:
            best_out[0] += mod[0]
            best_out[1] += mod[1]
        best_out[0] /= len(best_model)
        best_out[1] /= len(best_model)
        if DEBUG:
            print("Best model has {} votes (of {} possible)".format(best_votes, len(lines)))
        return best_out

    def _ransac_intersection(self, vis):
        lines = self._get_lines()
        best_model = []
        best_votes = 0

        # Precompute all line intersections
        if DEBUG:
            print("Pre-calc, ", end='')
        inter_map = self._calc_all_intersections(lines)

        # Convert intersection map into list of (intersection, line index) tuples sorted by proximity to center of the image. Bias sampling
        # towards intersections near the center of the image if proximity sampling is enabled
        ordered_inter = sorted([(inter_map[i], i) for i in inter_map], key=lambda x: self._l2_dist(x[0], (self.w // 2, self.h // 2)))
        sample_size = max(int(len(ordered_inter) * RANSAC_PROX_PERC), RANSAC_ITER)
        ordered_inter = ordered_inter[:sample_size]

        # Run RANSAC iterations
        for ransac_cnt in range(RANSAC_ITER):
            # If have already sampled every intersection, move on
            if RANSAC_PROX_SAMPLE and ransac_cnt >= len(ordered_inter):
                break

            # Randomly select lines
            ind1, ind2 = self._ransac_sampler(ordered_inter, lines)
            if ind1 == ind2:
                continue

            # Determine consensus
            current_votes, curr_intersection = self._compute_votes(lines, ind1, ind2, inter_map)
            if current_votes >= best_votes:
                # Handle multiple intersections with the same number of votes
                if current_votes > best_votes:
                    best_model = [curr_intersection]
                    best_votes = current_votes
                best_model.append(curr_intersection)
        if DEBUG:
            print("(tie of " + str(len(best_model)) + "), ", end='')
                    
        # Average the models that receive the most number of votes
        if len(best_model) > 0:
            best_out = [0, 0]
            for mod in best_model:
                cv.circle(vis, (int(mod[0]), int(mod[1])), 3, (0, 0, 255), -1)
                best_out[0] += mod[0]
                best_out[1] += mod[1]
            best_out[0] /= len(best_model)
            best_out[1] /= len(best_model)
            self.prev_pred = best_out
        else:
            # TODO ERROR
            if DEBUG:
                print("!!!!! ERROR: No acceptable models found in this frame !!!!")
            best_out = self.prev_pred
        if DEBUG:
            print("Best model has {} votes (of {} possible)".format(best_votes, len(lines)))
        return best_out, vis

    def _ransac_sampler(self, inters, lines): 
        """Return intersections closest to the center of the image if proximity "sampling" is enabled, else random sampling of lines."""
        if RANSAC_PROX_SAMPLE:
            cnt = np.random.choice(len(inters))
            indices = inters[cnt]
            return indices[1][0], indices[1][1]
        else:
            ind1 = np.random.choice(len(lines))
            ind2 = np.random.choice(len(lines))
        return ind1, ind2
   
    def _feature_detection(self, frame_gray):
        """Run Shi-Tomashi corner detection"""
        # Initialize mask, draw circles for all tracked features
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Run feature detection, detect new features
        p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            pre = len(self.tracks)
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(x, y)])
            print('Feature detector: {} (pre: {}, new: {})'.format(len(self.tracks), pre, len(np.float32(p).reshape(-1, 2))))
    
    def _calc_opt_flow(self, frame_gray):
        """Run Lucas-Kanade optical flow algorithm on new and previous frames. Perform in both directions as a consistency check."""
        img0, img1 = self.prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        # Validate difference between optical flows as sanity check
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        return p1, d < 1
        
    def _track_update(self, p1, good, vis):
        """Update tracked features and remove old instance of those features, draw circles for features"""
        new_tracks = []
        cnt = 0
        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                cnt += 1
                continue
            tr.append((x, y))
            while len(tr) > self.track_len:
                del tr[0]
            new_tracks.append(tr)
            cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        self.tracks = new_tracks
        print('tracks: {}, p1: {}, good: {} (rm cnt: {})'.format(len(self.tracks), len(p1.reshape(-1, 2)), len(good), cnt))

    def _plot_calc_line(self, img, p1, p2):
        """Draw a line through points p1 and p2 (of form (x, y)) in the specified image"""
        h, w = img.shape[:2]
        if p1[0] != p2[0]:
            # Calculate slope
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])

            # Calculate line end points
            px, qx = 0, w
            py = np.int32(p1[1] - slope * (p1[0] - 0))
            qy = np.int32(p2[1] + slope * (w - p2[0]))
        else:
            # Invalid slope, create vertical line
            px = qx = p1[0]
            py, qy = 0, h
        return (px, py), (qx, qy)

    def _plotting(self, vis, vp, pitch, yaw):
        """Plot optical flow lines, add text to frame"""
        # Plot lines defined by optical flow motion vectors in image
        if PLOT_LINES:
            for tr in self.tracks:
                p0, p1 = self._plot_calc_line(vis, tr[0], tr[-1])
                cv.line(vis, p0, p1, [0, 255, 0], 2)
        else:
            cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        x, y = (20, 20)
        s = 'track count: %d' % len(self.tracks)
        cv.putText(vis, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
        cv.putText(vis, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

        # Print pitch and yaw on image
        p = 'pitch: ' + str(pitch)
        ya = 'yaw: ' + str(yaw)
        y = 40
        cv.putText(vis, p, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
        cv.putText(vis, p, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
        y = 60
        cv.putText(vis, ya, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
        cv.putText(vis, ya, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

        # (DEBUG) plot line between image center and the primary vanishing point
        im_center = [int(self.w // 2), int(self.h //2)]
        cv.circle(vis, im_center, 4, (0, 0, 255), -1)

        # Plot primary vanishing point in the image
        cv.circle(vis, (int(vp[0]), int(vp[1])), 10, (0, 0, 255), -1)
        try_this = [math.tan(i) * FOCAL_LEN + offset for i, offset in zip((pitch, yaw), im_center)]
        cv.line(vis, (int(vp[0]), int(vp[1])), im_center, [255, 0, 0], 3)

        # (DEBUG) plot line between image center and the correct vanishing point
        if self.labels is not None:
            correct = [0, 0]
            correct[0] = math.tan(self.labels[self.frame_idx][0]) * FOCAL_LEN + im_center[0]
            correct[1] = im_center[1] - math.tan(self.labels[self.frame_idx][1]) * FOCAL_LEN
            cv.circle(vis, (int(correct[0]), int(correct[1])), 10, (0, 0, 255), -1)
            cv.line(vis, (int(correct[0]), int(correct[1])), im_center, [255, 0, 255], 3)



if __name__ == '__main__':
    OpticalFlow().run()
    cv.destroyAllWindows()
