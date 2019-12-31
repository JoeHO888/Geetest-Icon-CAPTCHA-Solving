import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from PIL import Image

def draw_bounding_box(pane, rect_coordinates):
    # Show bounding boxes

    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(pane)

    # Create a Rectangle patch
    for e in rect_coordinates:
        (x, y, w, h) = e
        rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    plt.show()
    
def find_bounding_box(pane, bounding_box_lower_thresholds, bounding_box_upper_thresholds, sort=True):
    # thresholds: turple
    # dimension_resized: turple
    
    segmented_pictures = []
    rect_coordinates = []
    
    
    width_lower_threshold, height_lower_threshold = bounding_box_lower_thresholds
    width_upper_threshold, height_upper_threshold = bounding_box_upper_thresholds
    
    _, contours, hierarchy = cv2.findContours(pane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if h > height_lower_threshold and w > width_lower_threshold and h <= height_upper_threshold and w <= width_upper_threshold:
            rect_coordinates.append((x, y, w, h))
        
        else:
            continue 
    if sort:
        x_coordinates = [x for (x,y,w,h) in rect_coordinates]
        rect_coordinates= [e for _,e in sorted(zip(x_coordinates,rect_coordinates))]
    return rect_coordinates

def segment_pictures(pane, rect_coordinates, dimension_resized, offset=2):
    segmented_pictures = []
    box_resized_width, box_resized_height = dimension_resized
    for rec_coordinate in rect_coordinates:
        (x, y, w, h) = rec_coordinate
        resized_pic = np.asarray(Image.fromarray(pane[max(0,y-offset):y+h+offset,max(0,x-offset):x+w+offset]).resize((box_resized_width,box_resized_height)))
        
        # if
        segmented_pictures.append(resized_pic)
    return segmented_pictures