import cv2
import os
import sys
import argparse

def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))
    for file in filenames:
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    return images

def main():
    parser = argparse.ArgumentParser(description="Stitch images from a folder into a panorama.")
    parser.add_argument('--folder', required=True, help='Path to folder containing input images')
    parser.add_argument('--output', required=True, help='Path to save the stitched panorama image')
    parser.add_argument('--mode', type=str, choices=['panorama', 'scans'], default='scans',
                        help="Stitching mode: 'panorama' for wide-angle scenes, 'scans' for scanned docs (default: scans)")
    args = parser.parse_args()

    images = load_images_from_folder(args.folder)
    if not images:
        sys.exit(1)

    mode_map = {
        'panorama': cv2.Stitcher_PANORAMA,
        'scans': cv2.Stitcher_SCANS
    }
    stitcher = cv2.Stitcher.create(mode_map[args.mode])
    stitcher.setPanoConfidenceThresh(0.7)

    status, stitched = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        sys.exit(1)

    cv2.imwrite(args.output, stitched)

if __name__ == "__main__":
    main()
