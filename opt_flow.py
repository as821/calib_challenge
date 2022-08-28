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

import sys
import numpy as np
import cv2 as cv

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


# Length of feature history to store, in video frames
TRACK_LEN = 30

# Frequency at which feature detection should be run, in number of frames
FEAT_DETECT_FREQ = 1



class OpticalFlow:
    def __init__(self, video_src):
        self.track_len = TRACK_LEN
        self.detect_interval = FEAT_DETECT_FREQ
        self.tracks = []
        self.cam = cv.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):
        """Run optical flow algorithm on the video clip."""
        while True:
            # Get next video frame
            _ret, frame = self.cam.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                # Run optical flow algorithm
                p1, good = self._calc_opt_flow(frame_gray)
                
                # Update tracks
                self._track_update(p1, good, vis)

                # Plot tracks
                self._plotting(vis)

            if self.frame_idx % self.detect_interval == 0:
                # Re-detect features periodically using Shi-Tomasi corner detection
                self._feature_detection(frame_gray)

            # Display frame and tracked features
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)
            if cv.waitKey(1) == 27:     # Stop if ESC is pressed
                break
                
    def _feature_detection(self, frame_gray):
        """Run Shi-Tomashi corner detection"""
        # Initialize mask, draw circles for all tracked features
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv.circle(mask, (x, y), 5, 0, -1)

        # Run feature detection, detect new features
        p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(x, y)])
    
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
        for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > self.track_len:
                del tr[0]
            new_tracks.append(tr)
            cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        self.tracks = new_tracks
    
    def _plotting(self, vis):
        """Plot optical flow lines, add text to frame"""
        cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        x, y = (20, 20)
        s = 'track count: %d' % len(self.tracks)
        cv.putText(vis, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
        cv.putText(vis, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
    
        
        

def main():
    try:
        video_src = sys.argv[1]
    except:
        print("Expected path argument missing")
        exit(1)

    OpticalFlow(video_src).run()


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
