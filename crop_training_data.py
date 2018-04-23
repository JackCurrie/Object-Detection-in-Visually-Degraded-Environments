import numpy as np
import imageio as io
from matplotlib import pyplot as plt
import cv2
import os
from glob import glob

SUB_SET = 'train'

OUTPUT_CAR_IMG_DIR = './cropped_split_dataset/' + SUB_SET + '/cars/'
OUTPUT_NON_CAR_DIR = './cropped_split_dataset/' + SUB_SET + '/non-cars/'
ORIGINAL_IMG_DIR = '/home/jc/Desktop/SeniorProjects/split_dataset/' + SUB_SET + '/data/'
ORIGINAL_GT = '/home/jc/Desktop/SeniorProjects/split_dataset/' + SUB_SET + '/gt.txt'


# Here we are going to sort our list of images by their file names
# so that we have them in their ACTUAL order. Then we will pull the images out of their
# dictionaries so that we can really see what is going on
import re
import operator

img_paths = os.path.join(ORIGINAL_IMG_DIR, '*.jpg')
img_files = glob(img_paths)

num_imgs = len(img_files)

imgs = []

for i, f in enumerate(img_files):
    if i % 100 == 0:
        print("image " + str(i))

    imgs.append({'f': f})

for img in imgs:
    img['i'] = int(re.findall(r'\d+', img['f'])[0])

imgs.sort(key=operator.itemgetter('i'))



# Read in ground truth from "ORIGINAL_GT" file and put it into a nicely formatted
# "img_boxes" variable
#
img_boxes = []
with open(ORIGINAL_GT) as file:
     content = file.readlines()

content = [str.strip() for str in content]
for line in content:
    single_img_content = [int(num) for num in line.split()]
    meta = single_img_content[0:2]
    coordinates = single_img_content[2:]
    img_dict = {
       'id': int(meta[0]),
        'num_boxes': int(meta[1]),
        'boxes': []
    }
    for i in range(img_dict.get('num_boxes')):
        car_dat = coordinates[i*4:i*4+4]
        car_dict = {
            'x': car_dat[0],
            'y': car_dat[1],
            'width': car_dat[2],
            'height': car_dat[3]
        }
        img_dict.get('boxes').append(car_dict)
    img_boxes.append(img_dict)





# !!!
# Here is where we output the car images, we need to do a dynamic loading
#   (Then also add the width and height of the image to the img_boxes from the dynamically loaded img)
#
for i, img in enumerate(imgs):
    box_list = img_boxes[i].get('boxes')

    # print('Img: ' + str(i))
    for j, box in enumerate(box_list):
        x = int(box.get('x'))
        y = int(box.get('y'))
        w = int(box.get('width'))
        h = int(box.get('height'))

        loaded_img = io.imread(img['f'])
        img_boxes[i]['img_height'] = loaded_img.shape[0]
        img_boxes[i]['img_width'] = loaded_img.shape[1]
        car_img = loaded_img[y:(y+h), x:(x+w)]

        # cv2.imwrite(OUTPUT_CAR_IMG_DIR + "img%s_car%s.jpg" %(i, j), car_img)



# Now we have to get a much larger quantity of non-vehiculaar sub-sections of
# the image
#
# In this block I will write the functions necessry, and in the next block I will sketch out
# I am going to use these functions
import pdb
import random
from datetime import datetime

random.seed(datetime.now())

# This will return a list of dicts which contain image data
# and a new file Names
def extractNonCarImages(img, img_data, new_img_size=(128, 128), num_new_imgs=12, img_name='none'):
    non_car_images = []

    for i in range(num_new_imgs):
        (x, y) = findValidImageCoordinates(img, img_data, new_img_size, prohibited_regions=img_data.get('boxes'))

        non_car_image = {}

        non_car_image['data'] = img[y:y+new_img_size[0], x:x+new_img_size[1]]
        non_car_image['name'] = img_name + str(i) + '.jpg'
        non_car_images.append(non_car_image)


    return non_car_images


# Return X, Y coordinates of top left corner of image
def findValidImageCoordinates(img, img_data, new_img_size=(128, 128), prohibited_regions=[]):
    # !!! Possible source of error: These img bounds may have been selected erroneously
    #
    # CHECKED: this appears to be good to go
    valid_img_bounds = {}
    valid_img_bounds['minX'] = 0
    valid_img_bounds['minY'] = 0
    valid_img_bounds['maxX'] = img_data.get('img_width') - new_img_size[0]
    valid_img_bounds['maxY'] = img_data.get('img_height') - new_img_size[1]

    valid_coordinate_found = False


    # The image proposals counter exists to act as a sort of "timeout" for images in which
    # an image will not ever be found
    image_proposals = 0
    while (not valid_coordinate_found and image_proposals < 600):

        (x, y) = stochasticXYProposal(valid_img_bounds)
        valid_coordinate_found = validImageProposal((x, y), new_img_size, prohibited_regions)
        image_proposals = image_proposals + 1


    return (x, y)

# Intakes Dictionary containing min/maxes of all X/Y values here
# Return tuple/dict with X/Y Coordinates for tlc of image
def stochasticXYProposal(imageBounds):
    x = random.randint(imageBounds['minX'], imageBounds['maxX'])
    y = random.randint(imageBounds['minY'], imageBounds['maxY'])

    return (x, y)



# Intakes top-left-coordinates, image size, and locations of all vehicle bounding boxes in images
# Returns true or false determining validity
#
# Works by checking if any of the corners of the prohbited regions are inside the proposed region,
# and vice-versa
def validImageProposal(tlc, img_size=(128, 128), prohibited_regions=[]):
    proposed_tlc = tlc
    proposed_trc = (tlc[0] + img_size[0], tlc[1])
    proposed_brc = (tlc[0] + img_size[0], tlc[1] + img_size[1])
    proposed_blc = (tlc[0]              , tlc[1] + img_size[1])
    proposed_corners = (proposed_tlc, proposed_trc, proposed_brc, proposed_blc)

    # Check if any of the corners of the proposed regions are within the prohibited regions
    for pr in prohibited_regions:
        for c in proposed_corners:
            if pointWithinRegion(c, pr):
                return False

    proposed_reg = {}
    proposed_reg['x'] = tlc[0]
    proposed_reg['y'] = tlc[1]
    proposed_reg['width'] = img_size[0]
    proposed_reg['height'] = img_size[1]

    # Check if any of the corners of the prohibited regions are within the proposed regions
    for pr in prohibited_regions:
        pr_tlc = (pr.get('x')                  , pr.get('y'))
        pr_trc = (pr.get('x') + pr.get('width'), pr.get('y'))
        pr_brc = (pr.get('x') + pr.get('width'), pr.get('y') + pr.get('height'))
        pr_blc = (pr.get('x')                  , pr.get('y') + pr.get('height'))
        pr_corners = (pr_tlc, pr_trc, pr_brc, pr_blc)
        for pr_c in pr_corners:
            if pointWithinRegion(pr_c, proposed_reg):
                return False

    # Now to check if there are any edge intersections between the proposed and prohibited regions
    # 1. Deal with the case where the proposed region is taller and narrower than the prohibited,
    #    and the sides intersect the top and botom of the prohibited.
    # 2. Deal with the case where the proposed region is wider and shorter than the prohibited region,
    #    and its tops/bottoms intersect the sides of the prohibited region.
    #
    for proh in prohibited_regions:
        if (proposed_reg['x'] > proh['x']):
            if (proposed_reg['x'] + proposed_reg['width']) < (proh['x'] + proh['width']):
                if (proposed_reg['y'] + proposed_reg['height']) > (proh.get('y') + proh.get('height')):
                    if (proposed_reg['y'] < proh.get('y')):
                        return False

    return True

# Intakes XY point tuple & prohibited regions (in form of box description from above)
# Outputs true if point is within the region, false otherwise
def pointWithinRegion(point, region):
    reg_min_x = region.get('x')
    reg_max_x = region.get('x') + region.get('width')
    reg_min_y = region.get('y')
    reg_max_y = region.get('y') + region.get('height')
    p_x = point[0]
    p_y = point[1]

    if p_x > reg_min_x and p_x < reg_max_x:
        if p_y > reg_min_y and p_y < reg_max_y:
            return True

    return False




# Iterate over new_img_size[1] all of the images in the data set
#    - Generate set of X sub-images either of randomly selected size within a reasonable distribution,
#      or of static size
#    - Put all of these images in the cropped_images/not_cars category
#
# !!! CONVENTION ALERT: Only extract non-vehicular sub-images is the phrase "BG" is in the file name.
#                       we will have to go back and add that to the img_boxes array or something
#
import sys

def writeImagesToFile(imgs_to_f=[]):
    for img in imgs_to_f:
        cv2.imwrite(OUTPUT_NON_CAR_DIR + img['name'], img['data'])

for i, img in enumerate(imgs[:4945]):
    try:
        if re.search('BG', img['f']):
            loaded_img = io.imread(img['f'])
            non_car_images = extractNonCarImages(loaded_img, img_boxes[i], img_name='img' + str(i) + '_')
            writeImagesToFile(non_car_images)

        if i % 50 == 0:
                print("Writing sub-images for image: " + str(i))

    except:
        db.set_trace()
