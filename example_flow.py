"""Example video stabilization: https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/"""

import cv2
import os
import numpy as np
import argparse
import math
import sys

from scipy.spatial.transform import Rotation as rotation


SMOOTHING_RADIUS = 3
PROCESS_LEN = 50



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
        

    def _get_next_frame(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None
        return frame


    def _convert_int(self, point):
        u, v = map(lambda x: int(round(x)), point)
        return (u, v)


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size)/window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory

def main():
    # Iterate through video, processing frame by frame
    vp = VideoProcessor()

    # Get next frame
    prev = vp._get_next_frame()
    if prev is None:
        return None

    # Covert to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
         
    # Pre-define transformation-store array
    transforms = []
    for i in range(PROCESS_LEN):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                            maxCorners=200,
                                            qualityLevel=0.01,
                                            minDistance=30,
                                            blockSize=3)

        # Read next frame
        success, curr = vp.cap.read()
        if not success:
            break 

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 

        # Sanity check
        assert prev_pts.shape == curr_pts.shape 

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        #Find transformation matrix
        m = cv2.estimateAffine2D(prev_pts, curr_pts) #will only work with OpenCV-3 or less
        m = m[0]

        # Extract traslation
        dx = m[0,2]
        dy = m[1,2]


        # TODO testing, Extract rotation matrix
        rot_mat = [list(m[0]), list(m[1]), [0, 0, 1]]
        rotate = rotation.from_matrix(rot_mat)
        pitch, roll, yaw = rotate.as_euler('XYZ', degrees=False)     # pitch, roll, yaw
        print(pitch, yaw, "(", roll, ")")  



        # Extract rotation angle
        da = np.arctan2(m[1,0], m[0,0])

        # Store transformation
        transforms.append([dx,dy,da])

        # Move to next frame
        prev_gray = curr_gray

       # print("Frame: " + str(i) + " -  Tracked points : " + str(len(prev_pts)))
    
    # Compute trajectory
    trajectory = np.cumsum(transforms, axis=0)

    # ??
    smoothed_trajectory = smooth(trajectory)



    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference



    # Reset stream to first frame
    vp.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

    # Write n_frames-1 transformed frames
    for _ in range(PROCESS_LEN):
        # Read next frame
        success, frame = vp.cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (len(frame[0]), len(frame)))

        # Fix border artifacts
        #frame_stabilized = fixBorder(frame_stabilized) 

        # Write the frame to the file
        frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        #if(frame_out.shape[1] > 1920):
        #    frame_out = cv2.resize(frame_out, None, 0.5, 0.5)

        cv2.imshow("Before and After", frame_out)
        if cv2.waitKey(10) == ord('q'):
           break



    # Cleanup
    print("Exiting...")
    vp.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()







