# Yolo-Segmentation-to-COCO-format-Converter

Sometimes refered to as yolo darknet format. Each line in the annotation (*.txt) file is as follows:
` <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn> (x1,y1 are point coordinates) `

This converter will use these coords and create both segmentation map and bounding box. 

Should work on all platforms. Tested on both Ubuntu 22.04.2 LTS (WSL2) and Windows 11. Supports Windows path style 

## Dependencies
Dependencies file created through using `pipreqs`. To install all: (may need to change pip to pip3 on some platforms)
```bash
    pip install -r requirements.txt
```

## Assumptions about dataset style and format
Directory:
<pre>
    dataset_root_dir/
        train/
            Photo_00001.png
            Photo_00001.txt
            Photo_00002.png
            Photo_00002.txt
        .../
            *.png
            *.txt
        classes.txt
</pre>
classes.txt:
`<class id> <class name> <super category>`

1. Each folder in dataset contains the image and its associated annotation file (named the same)
2. classes.txt needs to be created including all the classes in the dataset, their name and a super category (if they have an official one, otherwise make one up?)


## Usage
Very simple to use and create COCO style annotation from yolo darknet annotations:

### Quick explanation of command line arguments:
Can also be viewed in command line: `python3 yolo-to-coco.py --help`

- --dataset : Path of the dataset you want to convert annotations for, including the set (train, val, test). (e.g: /home/dataset/COCO/test/) 
- --save : Path for the directory + file_name to save the annotations to. (e.g: ./output/test.json)
- --show : Toggles showing the output images generated through conversion from yolo to coco
- --classes: Path for directory + file_name for the classes.txt file for each class that exist in the dataset
- --bbox: toggles showing bounding boxes for each annotation that exist in image. (Only worked when toggled with --show)

### testing | demo
To test whether all dependencies are all installed and to see how file structure should be made for generating the json files:
```bash
python3 yolo-to-coco.py --dataset ./demo/train/ --classes ./demo/classes.txt --save ./output/train.json
```

### Actual Usages (few options)
1. To just view dataset converted from yolo segmentation to coco:
```bash
python3 yolo-to-coco.py --dataset ./demo/train/ --classes ./demo/classes.txt --show #change both dataset and classes to suit yours
```
2. To view dataset with both segmentation map and bounding boxes (good test to see whether annotation loaded properly and bounding boxes generated properly)
```bash
python3 yolo-to-coco.py --dataset ./demo/train/ --classes ./demo/classes.txt --show --bbox
```
3. To save dataset (Good to use after testing using number 1 or 2.):
```bash
python3 yolo-to-coco.py --dataset ./demo/train/ --classes ./demo/classes.txt --save ./output/train.json
```
### Other mentions
To change some fields in output json file, the python file yolo-to-coco.py should have enough comments to avoid confusion.

## Citation
Image used in demo folder is from the train set of the MICCAI 2018 Grand Challenge titled: "Multi-Organ Nuclei Segmentation Challenge". 
The official dataset is labeled MoNuSeg and contains 30 training images, 7 validation images and 14 test images with full annotations for each set.
> -- <cite>N. Kumar et al., "A Multi-Organ Nucleus Segmentation Challenge," in IEEE Transactions on Medical Imaging, vol. 39, no. 5, pp. 1380-1391, May 2020, doi: 10.1109/TMI.2019.2947628.</cite>