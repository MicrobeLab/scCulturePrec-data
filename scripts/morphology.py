'''
Usage:
python morphology.py [image] [box_prompt] [output_prefix]
'''


import sys
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import math


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=1))    


def show_cv2_rect(rect_parameter, ax, pixel_size):
    c, w, h, a = rect_parameter[3], rect_parameter[5], rect_parameter[6], rect_parameter[4]
    h_um = rect_parameter[2]
    w_um = rect_parameter[1]
    box = rect_parameter[7]
    text_of_size = f"{w_um:.2f} x {h_um:.2f}um"
    text_x, text_y = c[0], c[1] - h
    x, y = box[1][0], box[1][1]
    ax.add_patch(plt.Rectangle((x, y), w, h, angle=a, edgecolor='blue', facecolor=(0,0,0,0), lw=1)) 
    ax.text(text_x, text_y, text_of_size, fontsize=8, ha='center', va='center', color='darkblue')


def show_circle(rect_parameter, ax):
    (x,y) = rect_parameter[10]
    center = (int(x),int(y))
    radius = int(rect_parameter[11])
    circle = Circle(center, radius, linewidth=1, edgecolor='pink', facecolor=(0,0,0,0))
    ax.add_patch(circle)


def show_ellipse(rect_parameter, ax):
    elli_obj = rect_parameter[12]
    ellipse = Ellipse(elli_obj[0], elli_obj[1][0], elli_obj[1][1], angle=elli_obj[2], facecolor=(0,0,0,0), edgecolor='orange')
    ax.add_patch(ellipse)


def show_scale_size(pixel_size, ax):
    ax.text(10, 10, f"Pixel Size: {pixel_size:.4f} um/px", fontsize=10, ha='left', va='top', color='green')
   

def measure_scale(prefix, scale_bar_length_actual=9):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    threshold[gray != 255] = 0
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        scale_bar_contour = max(contours, key=cv2.contourArea)
        #scale_bar_length_pixels = cv2.arcLength(scale_bar_contour, True)
        center, size, angle = cv2.minAreaRect(scale_bar_contour)
        w, h = size
        scale_bar_length_pixels = max(w, h)

        pixel_size = scale_bar_length_actual / scale_bar_length_pixels

    return pixel_size


def area_width_length(bool_mask, pixel_size):
    area = np.sum(bool_mask)*pixel_size*pixel_size
    binary_image = bool_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        contours = max(contours, key=cv2.contourArea)
    else:
        contours = contours[0]
    rect = cv2.minAreaRect(contours)
    box = cv2.boxPoints(rect)
    length = max(rect[1])*pixel_size
    width = min(rect[1])*pixel_size
    center = rect[0]
    angle = rect[2]
    rect_info = (area, width, length, center, angle, rect[1][0], rect[1][1], box)
    return rect_info


def get_median_radius(contour):
    moms = cv2.moments(contour)
    x = int((moms['m10'])/(moms['m00']))
    y = int((moms['m01'])/(moms['m00']))
    dists = []
    for c in contour:
        a = np.array((x, y))
        b = np.array((c[0][0], c[0][1]))
        dists.append(np.linalg.norm(a-b))
    dists.sort()
    radius = (dists[int((len(dists) - 1)/2)] + dists[int(len(dists)/2)]) / 2 
    return radius


def get_inertia_ratio(contour):
    moms = cv2.moments(contour)
    denom = np.sqrt((2*moms['mu11'])**2) + ((moms['mu20'] - moms['mu02'])**2)
    eps = .01
    inertiaRatio = 1
    if(denom > eps):
        cosmin = (moms['mu20'] - moms['mu02']) / denom
        sinmin = 2 * moms['mu11'] / denom
        cosmax = -cosmin
        sinmax = -sinmin
        imin = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] - moms['mu02']) * cosmin - moms['mu11'] * sinmin;
        imax = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] - moms['mu02']) * cosmax - moms['mu11'] * sinmax;
        inertiaRatio = imin / imax
    return inertiaRatio


def shrink_mask_border(mask):
    binary_image = np.uint8(mask)
    kernel_size = 3
    eroded_mask = cv2.erode(binary_image, np.ones((kernel_size, kernel_size), dtype=np.uint8))
    return eroded_mask


def various_morpho(bool_mask, pixel_size):
    area = np.sum(bool_mask)*pixel_size*pixel_size   # 3. area
    binary_image = bool_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        rect_info = ("NA",) * 21
    else:
        if len(contours) > 1:
            contours = max(contours, key=cv2.contourArea)
        else:
            contours = contours[0]
        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect)
        length = max(rect[1])*pixel_size  # 1. length
        width = min(rect[1])*pixel_size   # 2. width
        center = rect[0]
        angle = rect[2]
        perimeter = cv2.arcLength(contours,True)*pixel_size  # 8. perimeter
        (x_circle,y_circle), radius_circle = cv2.minEnclosingCircle(contours)
        diameter = radius_circle*2*pixel_size  # 6. diameter
        if len(contours) >= 5:
            ellipse = cv2.fitEllipse(contours)
        else:
            ellipse = 'NA'
        major_axis_ellipse = max(ellipse[1])
        minor_axis_ellipse = min(ellipse[1])
        # eccentricity = sqrt(1 - (b^2 / a^2))
        eccentricity = math.sqrt(1 - (minor_axis_ellipse ** 2 / major_axis_ellipse ** 2))  # 4. eccentricity 
        min_max_ratio = width / length  # 5. min max ratio
        extent = area / (width*length)  # 7. extent
        cylinder_r = width / 2
        cylinder_h = length - width
        cylinder_v = math.pi * cylinder_r**2 * cylinder_h
        sphere_v = (4/3) * math.pi * cylinder_r**3
        volume = cylinder_v + sphere_v   # 9. volumn
        med_radius = get_median_radius(contours)*pixel_size   # 10. median radius
        circularity = (4*np.pi*area) / (perimeter**2)   # 11. circularity
        hull_area = cv2.contourArea(cv2.convexHull(contours))  
        convexity_ratio = cv2.moments(contours)['m00'] / hull_area
        #convexity_ratio = area / (hull_area*pixel_size*pixel_size)  # 12. convexity
        inertia_ratio = get_inertia_ratio(contours)  # 13. inertia ratio
        rect_info = (area, width, length, center, angle, rect[1][0], rect[1][1], box, perimeter, diameter, 
            (x_circle,y_circle), radius_circle, ellipse, eccentricity, min_max_ratio, extent, volume, 
            med_radius, circularity, convexity_ratio, inertia_ratio)   
    return rect_info



# input image
image = sys.argv[1]
image = cv2.imread(image)

# input prompts (box)
prompts = sys.argv[2]
prompts = open(prompts).read().strip().split('\n')

# define output prefix
output_prefix = sys.argv[3]

# pixel size in um
#um_per_px = measure_scale(output_prefix)
um_per_px = 9 / 118


# load model
sam_checkpoint = "/path/to/segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
#device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)


# image embedding
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)


# output image showing mask, box, rect, size
plt.figure(figsize=(10,10))
plt.imshow(image)

# output size 
fh_size = open(output_prefix+'_size.txt', 'w')
#fh_size.write(f"cell_number\tarea\twidth\tlength\teccentricity\tmin_max_axis_ratio\tdiameter_enclosing_circle\textent_ratio\tperimeter\tvolume\tmedian_radius\tcircularity\tconvexity_ratio\tinertia_ratio\n")  
fh_size.write(f"cell_number\tarea\twidth\tlength\taspect_ratio\teccentricity\tcircularity\textent\tperimeter\tcentroid\n")  


# iterate, create numpy array
for cell_info in prompts:
    cell_info = cell_info.split('\t')
    number = cell_info[0]
    input_box = np.array([int(cell_info[1]), int(cell_info[2]), int(cell_info[3]), int(cell_info[4])])

    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box, multimask_output=False,)
    masks = masks[0]
    masks = shrink_mask_border(masks)
    mask_rect = various_morpho(masks, um_per_px)
    #fh_size.write(f"{number}\t{mask_rect[0]}\t{mask_rect[1]}\t{mask_rect[2]}\t{mask_rect[13]}\t{mask_rect[14]}\t{mask_rect[9]}\t{mask_rect[15]}\t{mask_rect[8]}\t{mask_rect[16]}\t{mask_rect[17]}\t{mask_rect[18]}\t{mask_rect[19]}\t{mask_rect[20]}\n")  
    fh_size.write(f"{number}\t{mask_rect[0]}\t{mask_rect[1]}\t{mask_rect[2]}\t{1/mask_rect[14]}\t{mask_rect[13]}\t{mask_rect[18]}\t{mask_rect[15]}\t{mask_rect[8]}\t{mask_rect[17]}\n")  
    show_mask(masks, plt.gca(), random_color=True)


show_scale_size(um_per_px, plt.gca())
plt.axis('off')
#plt.savefig(output_prefix+'_masks.png', bbox_inches='tight')
plt.savefig(output_prefix+'.tiff', dpi=300, format='tiff', bbox_inches='tight', pad_inches=0)

fh_size.close()

