import os
import json
import cv2
import numpy as np
from shapely.geometry import Polygon, box
import argparse
from pathlib import Path
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# -- [ This function initialises the whole coco framework...data is appended to this ] -- #
def init_coco_data():
    coco_data = {
    "info": [],
    "categories": [],
    "images": [],
    "annotations": []
    }
    return coco_data

# -- [ Category data based on classes available...init and change data to suit dataset ] -- #
def get_categories():
    categories = {
            "id": 1, 
            "name": "nucleus",
            "supercategory": "cells"
    }
    return categories

def init_categories(id: int, name: str, supcat: str):
    categories = {
            "id": id, 
            "name": name,
            "supercategory": supcat
    }
    return categories


# -- [ My info for creating this tool, feel free to change when creating your own dataset <3 ] -- #
def get_info():
    info = {
    "description": "MoNuSeg dataset COCOfyed",
    "version": "1.0",
    "year": "2023",
    "contributor": "",
    "date_created": "21/09/2023",
    "author_creator": "ChakiCurtin",
    } 
    return info

# -- [ one image entry for each image present in the dataset folder ] -- #
def init_image_entry(image_id: int, filename: str, width: int, height: int):
    image_entry = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
    }
    return image_entry

# -- [ one annotation entry...need one for each annotation present in the image ] -- #
def init_annotation_entry(anno_id: int, image_id: int, is_crowd: int, area: float, category_id: int):
    annotation_entry = {
                    "id": anno_id,
                    "image_id": image_id,
                    "iscrowd": 0,
                    "area": area,
                    "category_id": category_id,
                    "bbox": [],
                    "segmentation": [],
                }
    return annotation_entry
    
def show(coco_data: dict, image_dir: Path):
    coco = COCO()
    coco.dataset = coco_data
    coco.createIndex()

    img_ids = coco.getImgIds() # there is only one category

    for img in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs([img])[0]
        image_path = img_info["file_name"]
        image_path = os.path.join(image_dir, image_path)
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.title(str(img_info["file_name"]))
        coco.showAnns(anns, draw_bbox=args.bbox)

        plt.show()

def save(coco_data: any, args: argparse):
    with open(args.save, "w") as json_file:
        json.dump(coco_data, json_file, indent= 4)

# ============================================================================

def main(args: argparse.Namespace):
    # -- [ Lets initialise a new coco-format template to use ] -- #
    coco_data = init_coco_data()
    print("[*] Creating coco template to fill with annotation data..")
    # -- [ initialise the classes through categories ] -- #
        # -- [ Read in classes.txt ] -- #
    print("[*] Reading classes.txt..")
    with open(args.classes, mode="r") as classes:
        for item in classes:
            full_line = item.split() #[ 0, nuclei, cells ]
            class_id = int(full_line[0]) + 1 # classess must start from 1 onwards
            class_name = str(full_line[1]) # nuclei
            super_cat = str(full_line[2]) # cells
            coco_data["categories"].append(init_categories(id=class_id, name=class_name, supcat=super_cat))
    # -- [ add in the info ] -- #
    print("[*] Adding info to template..")
    info = get_info()
    coco_data["info"].append(info)
    # -- [ Now lets get to the meat of the program ] -- #
    image_id = 0
    annotation_id = 0

    image_dir = args.dataset

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_id += 1
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            height, width, _ = img.shape
            # -- [ add a new image entry ] -- #
            print("[*] Found image: " + filename + "...Adding annotation data.." )
            coco_image_entry = init_image_entry(image_id=image_id,filename=filename, width=int(width),height=int(height))
            coco_data["images"].append(coco_image_entry)
            
            # Load and process the corresponding segmentation mask
            anno_filename = os.path.splitext(filename)[0] + ".txt"
            anno_path = os.path.join(image_dir, anno_filename)
            with open(anno_path, mode='r') as file:
                for item in file:   # per line (nuclei)
                    full_line = item.split()    # [0, 0.77, 0.66, 0.76, ...]
                    # class ID
                    class_id = int(full_line[0]) + 1 # Classes must start from 1..
                    is_crowd = 0                # dealing with seperate nuclei, so always 0
                    # gathering all but first index for segmentation points
                    segmentation = np.asfarray(full_line[1::1]) 
                    #incrementing annotation id
                    annotation_id += 1
                    # getting bounding box
                    all_x_coords = np.asfarray(full_line[1::2]) * width
                    all_y_coords = np.asfarray(full_line[2::2]) * height
                    # -- [ This next part will form two lists of x and y coords together ] -- #
                    # -- WILL OPTIMISE LATER -- #
                    merged_seg = []
                    for i in range(len(all_x_coords)):
                        merged_seg.append(all_x_coords[i])
                        merged_seg.append(all_y_coords[i])
                    # -- [ Create polygon of all x,y coords to find area and bounding box for each nuclei ] --#
                    pgon = Polygon(zip(all_x_coords, all_y_coords)) 
                    minx, miny, maxx, maxy = pgon.bounds
                    bound_box = box(minx,miny,maxx,maxy)
                    xx, yy = bound_box.exterior.coords.xy
                    # -- [ getting the proper box around the nuclei now ] -- #
                    g_minx = min(xx)
                    g_miny = min(yy)
                    g_width = max(xx) - min(xx)
                    g_height = max(yy) - min(yy)
                    # area of poly
                    area = pgon.area

                    # -- [ Now use all these values to create the annotation entry ] -- #
                    anno_entry = init_annotation_entry(anno_id=annotation_id, 
                                                       image_id=image_id, 
                                                       is_crowd=is_crowd, 
                                                       area=area,
                                                       category_id=class_id
                                                       )
                    anno_entry["bbox"] = [float(g_minx), float(g_miny), float(g_width), float(g_height)]
                    anno_entry["segmentation"] = [merged_seg]
                    coco_data["annotations"].append(anno_entry)

    if args.show:
        print("[*] Showing completed COCO formatted data...")
        show(coco_data=coco_data, image_dir=image_dir)
    elif args.save is not None:
        print("[*] Saving completed COCO dataset to: " + str(args.dataset))
        save(coco_data=coco_data, args=args)
    else:
        print("[!] It seems you have forgotten to either save or show dataset...")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path, default=None, help="The path to the dataset folder to convert (/home/dataset/COCO/test/)")
    parser.add_argument("--save", required=False, type=Path, default=None, help="The path and name of file to save coco formated file to (/home/output/train.json)")
    parser.add_argument("--show", required=False, action="store_true", help="Toogle showing images with output mask (use --bbox aswell to show bounding box)")
    parser.add_argument("--classes", required=True, type=Path, help="The path and filename to the classes list for yolo segmentation annotations")
    parser.add_argument("--bbox", required=False, action="store_true", help="Toggle bounding boxes when combined with --show option")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    main(args)
