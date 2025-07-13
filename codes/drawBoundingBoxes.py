import cv2
import argparse

def draw_yolo_bounding_boxes(image_path, label_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_height, img_width = image.shape[:2]
    
    try:
        with open(label_path, 'r') as file:
            labels = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    # Iterate through each bounding box in the file
    for index, label in enumerate(labels):
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
        x1 = int(x_center_pixel - box_width_pixel / 2)
        y1 = int(y_center_pixel - box_height_pixel / 2)
        x2 = int(x_center_pixel + box_width_pixel / 2)
        y2 = int(y_center_pixel + box_height_pixel / 2)
        
        # Draw the bounding box
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        label_text = f"{int(index)}"
        cv2.putText(image, label_text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    cv2.imwrite(output_path, image)
    print(f"Image with bounding boxes saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Draw YOLO bounding boxes on an image.")
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--label', required=True, help='Path to YOLO label file')
    parser.add_argument('--output', required=True, help='Path to save output image with bounding boxes')

    args = parser.parse_args()
    
    draw_yolo_bounding_boxes(args.image, args.label, args.output)

if __name__ == "__main__":
    main()
