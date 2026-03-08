import cv2
import numpy as np
import os
import sys
import argparse


def get_cell_boxes(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not read file {args.input}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    denoised = cv2.medianBlur(gray, 5)
    
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, args.block_size, args.c_value
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    ret, sure_fg = cv2.threshold(
        dist_transform, args.dist_coeff * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        sure_fg, connectivity=8
    )

    final_boxes = []
    vis_img = img.copy()

    scale_bar_x_limit = int(w_img * 0.10)
    scale_bar_y_start = int(h_img * 0.90)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        if area < args.min_area or area > args.max_area:
            continue

        if cx < scale_bar_x_limit and cy > scale_bar_y_start:
            continue

        if args.exclude_y is not None and cy < int(h_img * args.exclude_y):
            continue

        if args.exclude_x is not None and cx < int(w_img * args.exclude_x):
            continue

        roi = gray[y:y+h, x:x+w]
        if roi.size == 0 or np.min(roi) > args.blank_threshold:
            continue

        x_final = max(0, x - args.padding)
        y_final = max(0, y - args.padding)
        w_final = min(w_img - x_final, w + 2 * args.padding)
        h_final = min(h_img - y_final, h + 2 * args.padding)

        final_boxes.append([int(x_final), int(y_final), int(x_final + w_final), int(y_final + h_final)])
        
        cv2.rectangle(vis_img, (x_final, y_final), (x_final + w_final, y_final + h_final), (0, 255, 0), 2)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    img_output_path = os.path.join(args.output_dir, f"{args.prefix}_{base_name}.jpg")
    cv2.imwrite(img_output_path, vis_img)

    txt_output_path = os.path.join(args.output_dir, f"{args.prefix}_{base_name}.txt")
    with open(txt_output_path, 'w') as f:
        for idx, box in enumerate(final_boxes):
            f.write(f"{idx}\t{box[0]}\t{box[1]}\t{box[2]}\t{box[3]}\n")

    print(f"--- Processing Complete ---")
    print(f"Input: {args.input}")
    print(f"Detected Cells: {len(final_boxes)}")
    print(f"Visual result: {img_output_path}")
    print(f"Coordinates: {txt_output_path}")

    return final_boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell detection with adaptive thresholding and separation.")
    
    # IO Parameters
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output_dir", "-o", default="output", help="Directory for output files")
    parser.add_argument("--prefix", "-p", default="det", help="Prefix for output filenames")
    
    # OpenCV Parameters
    parser.add_argument("--block_size", type=int, default=21, help="Block size for adaptive thresholding (must be odd)")
    parser.add_argument("--c_value", type=int, default=4, help="C constant for adaptive thresholding")
    parser.add_argument("--dist_coeff", type=float, default=0.3, help="Coefficient for distance transform threshold")
    
    # Filtering Parameters
    parser.add_argument("--min_area", type=int, default=30, help="Minimum pixel area for a valid cell")
    parser.add_argument("--max_area", type=int, default=1000, help="Maximum pixel area for a valid cell")
    parser.add_argument("--blank_threshold", type=int, default=200, help="Brightness threshold to identify blank zones")
    parser.add_argument("--padding", type=int, default=5, help="Padding pixels to expand detected boxes")
    parser.add_argument("--exclude_y", type=float, default=None, help="Customized excluded region (y)")
    parser.add_argument("--exclude_x", type=float, default=None, help="Customized excluded region (x)")

    args = parser.parse_args()
    get_cell_boxes(args)
