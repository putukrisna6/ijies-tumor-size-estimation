import cv2
import numpy as np
import os
import shutil
import argparse

# Rescale the images
def rescale(img):
    scale = 1
    h, w = img.shape[:2]
    h = int(h * scale)
    w = int(w * scale)
    return cv2.resize(img, (w, h))

def main():
    parser = argparse.ArgumentParser(description="Extract key frames from a video based on ORB feature matching.")
    parser.add_argument('--video', required=True, help='Path to the input video file')
    parser.add_argument('--folder', required=True, help='Path to output folder to save selected frames')
    parser.add_argument('--cutoff', type=int, default=10, help='Minimum number of good keypoint matches to skip saving frame (default: 10)')
    args = parser.parse_args()

    # Delete and recreate output folder
    if os.path.isdir(args.folder):
        shutil.rmtree(args.folder)
    os.mkdir(args.folder)

    # Open video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{args.video}'")
        return

    counter = 0
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Read the first frame
    ret, last = cap.read()
    if not ret:
        print("Error: Could not read the first frame from the video.")
        return

    last = rescale(last)
    cv2.imwrite(os.path.join(args.folder, str(counter).zfill(5) + ".png"), last)
    kp1, des1 = orb.detectAndCompute(last, None)
    prev = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = rescale(frame)
        kp2, des2 = orb.detectAndCompute(frame, None)

        if des1 is None or des2 is None:
            continue
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)

        if len(good) < args.cutoff:
            counter += 1
            last = frame
            kp1 = kp2
            des1 = des2
            filename = os.path.join(args.folder, str(counter).zfill(5) + ".png")
            cv2.imwrite(filename, last)
            print(f"New Frame Saved: {filename}")

        cv2.waitKey(1)
        prev = frame

    # Save the last frame
    if prev is not None:
        counter += 1
        filename = os.path.join(args.folder, str(counter).zfill(5) + ".png")
        cv2.imwrite(filename, prev)

    print("Total Frames Saved:", counter)

if __name__ == "__main__":
    main()
