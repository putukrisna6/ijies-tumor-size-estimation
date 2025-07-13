import cv2
import os
import numpy as np
import math
import argparse

def get_depth_map(output, w, h):
    depth = cv2.resize(output, (w, h))
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    return depth

def crop_and_calculate_depths(image_path, label_path, depth_data, output_folder, diagonals_file, depth_file):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Resize the depth map to match the image dimensions
    depth_map = cv2.resize(depth_data, (img_width, img_height))

    # Read the YOLO label file
    try:
        with open(label_path, 'r') as file:
            labels = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the output files for writing
    with open(diagonals_file, 'w') as diag_file, open(depth_file, 'w') as depth_file_obj:
        # Iterate through each bounding box in the file
        for i, label in enumerate(labels):
            data = label.strip().split()
            if len(data) != 5:
                print(f"Skipping invalid line: {label}")
                continue

            class_id, x_center, y_center, width, height = map(float, data)

            # Convert normalized coordinates to absolute pixel values
            x_center_pixel = int(x_center * img_width)
            y_center_pixel = int(y_center * img_height)
            box_width_pixel = int(width * img_width)
            box_height_pixel = int(height * img_height)

            # Calculate the top-left and bottom-right corners of the bounding box
            x1 = max(0, int(x_center_pixel - box_width_pixel / 2))
            y1 = max(0, int(y_center_pixel - box_height_pixel / 2))
            x2 = min(img_width, int(x_center_pixel + box_width_pixel / 2))
            y2 = min(img_height, int(y_center_pixel + box_height_pixel / 2))

            # Crop the object
            cropped_object = image[y1:y2, x1:x2]

            # Save the cropped object as a separate image
            output_file = os.path.join(output_folder, f"object_{i}_class_{int(class_id)}.jpg")
            cv2.imwrite(output_file, cropped_object)

            # Calculate the diagonal size
            diagonal = math.sqrt(box_width_pixel**2 + box_height_pixel**2)
            diag_file.write(f"{diagonal:.2f}\n")

            # Extract the corresponding depth data for the cropped region
            cropped_depth = depth_map[y1:y2, x1:x2]

            # Calculate the average depth (mean of all channels if depth_map is color)
            if len(cropped_depth.shape) == 3:
                avg_depth = np.mean(cropped_depth)
            else:
                avg_depth = np.mean(cropped_depth)
            depth_file_obj.write(f"{avg_depth:.2f}\n")

            print(f"Processed Object {i}: Saved cropped object, diagonal = {diagonal:.2f} pixels, average depth = {avg_depth:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Crop objects from YOLO-labeled image, calculate diagonals and average depth.")
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--label', required=True, help='Path to YOLO label file')
    parser.add_argument('--depth_data', required=True, help='Path to depth data file (numpy .npy format)')
    parser.add_argument('--output_folder', required=True, help='Folder to save cropped images')
    parser.add_argument('--diagonals_file', required=True, help='File to save diagonal sizes')
    parser.add_argument('--depth_file', required=True, help='File to save average depth values')

    args = parser.parse_args()

    depth_data = np.load(args.depth_data)
    crop_and_calculate_depths(args.image, args.label, depth_data, args.output_folder, args.diagonals_file, args.depth_file)

if __name__ == "__main__":
    main()
