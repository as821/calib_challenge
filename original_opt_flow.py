import cv2
import os
import numpy as np
import argparse
import math





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
        self.frame_cnt += 1 
        return frame, self.frame_cnt

    def _display_circle(self, img, point):
        u, v = map(lambda x: int(round(x)), point.pt)
        cv2.circle(img, (u, v), color=(0, 255, 0), radius=3)
        return (u, v)

    def _dist_filter(self, p1, p2):
        # OpenCV Point stores (column, row)
        return max(math.sqrt((p1.pt[0] - p2.pt[0])**2), math.sqrt((p1.pt[1] - p2.pt[1])**2))

    def filter_matches(self, curr_keyp, matches):
        # Sort by Hamming distance
        matches = sorted(matches, key=lambda x:x.distance)  
        
        # Remove large horizontal displacements
        #output = []
        #for m in matches:
        #    if m.distance > MATCH_DIST_THRESH:
        #        continue
        #    if m.queryIdx < len(curr_keyp) and m.queryIdx < len(self.prev_frame_feat[0]): 
        #        horiz = self._dist_filter(curr_keyp[m.queryIdx], self.prev_frame_feat[0][m.queryIdx]) 
        #        if horiz <= DIM_DIST_THRESH: 
        #            print(m.distance, horiz) #, curr_keyp[m.queryIdx].pt, self.prev_frame_feat[0][m.queryIdx].pt)
        #            output.append(m)
        #print("\n") 
        return matches[:500]  #output

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
            matches = self.filter_matches(keyp, matches)
            
            # (DEBUG) Output matches on screen
            for m in matches:
                # Get relevant keypoints for the match, display them 
                prev_pt = self._display_circle(frame, self.prev_frame_feat[0][m.queryIdx])
                curr_pt = self._display_circle(frame, keyp[m.trainIdx])

                # Draw line to represent the match
                cv2.line(frame, prev_pt, curr_pt, [0, 255, 0], 2)
            
            # Calculate the average direction change
            avg_vec = self.calc_avg_direction(matches, keyp) 

            # ASSUMPTION: car is driving straight, direction change is with respect to zero vector

            # Calculate pitch and yaw angles
            pitch, yaw = self.calc_angles(avg_vec)
            print("Pitch: ", pitch, ", yaw: ", yaw, ", avg_vec: ", avg_vec)


        self.prev_frame_feat = (keyp, desc)

        return frame


def main():
    # Iterate through video, processing frame by frame
    #vp = VideoProcessor()
    #while True:
    

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to input .hecv file")
    args = parser.parse_args()

    # Load video
    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()



    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)



        # Process frame
        #frame = vp.process_single_frame()
        #if frame is None:
        #    break

        # Display resulting frame
        #cv2.imshow('frame', frame)
        #if cv2.waitKey(1) == ord('q'):
        #    break

    # Cleanup
    print("Exiting...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()







