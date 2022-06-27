# Implementation of AST Object Detection Model
> This model adapts the [PyTorch Torchvision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) to trained a Faster R-CNN model using the for Pedestrian Detection and Segmentation. It contains 170 images with 345 instances of pedestrians, and we will use it to illustrate how to use the new features in torchvision in order to train an instance segmentation model on a custom dataset.


## Defining the Dataset

## Data Description
This dataset contains images,`512×512` pixels, with 6 different types of objects that will be utilized. 
```python
{'sedimentation_tank', 'water_tower', 'spherical_tank',
'closed_roof_tank', 'external_floating_roof_tank',
 'narrow_closed_roof_tank'}
```

## Inputs to model
Each image can contain one or more ground truth objects.
Each object is represented by –
- a bounding box in absolute boundary coordinates
- a label (one of the object types mentioned above)
- a perceived detection difficulty (either `0`, meaning _not difficult_, or `1`, meaning _difficult_)

## Images

Since we're using the Faster-RCNN, the images would need to be sized at `224×224` pixels and in the RGB format.

Remember, we're using a Resnet152 base pretrained on ImageNet that is already available in PyTorch's `torchvision` module. [This page](https://pytorch.org/docs/master/torchvision/models.html) details the preprocessing or transformation we would need to perform in order to use this model – pixel values must be in the range [0,1] and we must then normalize the image by the mean and standard deviation of the ImageNet images' RGB channels.

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Also, PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions.

Therefore, **images fed to the model must be a `Float` tensor of dimensions `N, 3, 224×224`**, and must be normalized by the aforesaid mean and standard deviation. `N` is the batch size.

### Objects' Bounding Boxes

We would need to supply, for each image, the bounding boxes of the ground truth objects present in it in fractional boundary coordinates `(x_min, y_min, x_max, y_max)`.

COCO Bounding box: ( x-top left, y-top left, width, height ) 
Pascal VOC Bounding box :( xmin-top left, ymin-top left,xmax-bottom right, ymax-bottom right )
Pytorch: (xmin,ymin,xmax,ymax) 



Since the number of objects in any given image can vary, we can't use a fixed size tensor for storing the bounding boxes for the entire batch of `N` images.

Therefore, **ground truth bounding boxes fed to the model must be a list of length `N`, where each element of the list is a `Float` tensor of dimensions `N_o, 4`**, where `N_o` is the number of objects present in that particular image.

### Objects' Labels

We would need to supply, for each image, the labels of the ground truth objects present in it.

Each label would need to be encoded as an integer from `1` to `20` representing the twenty different object types. In addition, we will add a _background_ class with index `0`, which indicates the absence of an object in a bounding box. (But naturally, this label will not actually be used for any of the ground truth objects in the dataset.)

Again, since the number of objects in any given image can vary, we can't use a fixed size tensor for storing the labels for the entire batch of `N` images.

Therefore, **ground truth labels fed to the model must be a list of length `N`, where each element of the list is a `Long` tensor of dimensions `N_o`**, where `N_o` is the number of objects present in that particular image.

## Data pipeline
### Parse raw data using parse.py
We must split images into train and test sets.
python parse.py -- complete_img_ids img_ids_txt_file
                  -- parent_directory /path/to/parent/directory
                
Example:
python parse.py -- complete_img_ids img_ids.txt -- parent_directory C:\chip_allocation_temp

#for height estimation
python parse.py --parent_directory \\oit-nas-fe13dc.oit.duke.edu\\data_commons-borsuk\\complete_dataset --img_directory chips_positive --annotation_directory chips_positive_corrected_xml --train_val_percent 1 --bbox_remove 0
                        
The `make_list_of_image_ids()` function creates a text file of image ids. 
The `split_train_val_test()` function randomly selects files to be in each split and creates a text file of the image ids for each split.  
The `create_data_lists()` function parses the data downloaded and saves the following files:

"""
This parses the data downloaded and saves the following files –
"""
- A **JSON file for each split with a list of the absolute filepaths of `I` images**, where `I` is the total number of images in the split.
- A **JSON file for each split with a list of `I` dictionaries containing ground truth objects, i.e. bounding boxes in absolute boundary coordinates, their encoded labels, and perceived detection difficulties**. The `i`th dictionary in this list will contain the objects present in the `i`th image in the previous JSON file.
- A **JSON file which contains the `label_map`**, the label-to-index dictionary with which the labels are encoded in the previous JSON file. This dictionary is also available in [`utils.py`] and directly importable.

### PyTorch Dataset (Data Loader)
   The `PascalVOCDataset` class (found in the util module) is used as the data loader.
This is a subclass of PyTorch, used to **define our training and test datasets.** 
It requires a `__len__` method defined, which returns the size of the dataset, and a `__getitem__` method which returns the `i`th image, bounding boxes of the objects in this image, and labels for the objects in this image, using the JSON files we saved earlier.

You will notice that it also returns the perceived detection difficulties of each of these objects, but these are not actually used in training the model. They are required only in the **evaluation** stage for computing the Mean Average Precision (mAP) metric. We also have the option of filtering out _difficult_ objects entirely from our data to speed up training at the cost of some accuracy.

Note: consider removing the difficult/truncated objects

#### Data Transforms (Augmentation)

Additionally, inside this class, **each image and the objects in them are subject to a slew of transformations** as described in the Data Augmentation section.

#### Data Transforms
Note:
Consider: 
1) Not transforming data
2) Transforming data once
3) Oversampling https://discuss.pytorch.org/t/increase-dataset-size-using-data-augmentation/118856/5


The `transform()` in the utils.py module is used to preform data augmentation inside the data loader.

This function applies the following transformations to the images and the objects in them –

- Randomly **adjust brightness, contrast, saturation, and hue**, each with a 50% chance and in random order.
- Randomly 
- With a 50% chance, **perform a _zoom out_ operation** on the image. This helps with learning to detect small objects. The zoomed out image must be between `1` and `4` times as large as the original. The surrounding space could be filled with the mean of the ImageNet data.

- Randomly crop image, i.e. **perform a _zoom in_ operation.** This helps with learning to detect large or partial objects. Some objects may even be cut out entirely. Crop dimensions are to be between `0.3` and `1` times the original dimensions. The aspect ratio is to be between `0.5` and `2`. Each crop is made such that there is at least one bounding box remaining that has a Jaccard overlap of either `0`, `0.1`, `0.3`, `0.5`, `0.7`, or `0.9`, randomly chosen, with the cropped image. In addition, any bounding boxes remaining whose centers are no longer in the image as a result of the crop are discarded. There is also a chance that the image is not cropped at all.

- With a 50% chance, **horizontally flip** the image.

- Remove this?: **Resize** the image to `224×224` pixels. This is a requirement of the SSD300.

- Convert all boxes from **absolute to fractional boundary coordinates.** At all stages in our model, all boundary and center-size coordinates will be in their fractional forms.

- **Normalize** the image with the mean and standard deviation of the ImageNet data that was used to pretrain our VGG base.

As mentioned in the paper, these transformations play a crucial role in obtaining the stated results.

#### PyTorch DataLoader

The `Dataset` described above, `PascalVOCDataset`, will be used by a PyTorch [`DataLoader`](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) in `train.py` to **create and feed batches of data to the model** for training or evaluation.

Since the number of objects vary across different images, their bounding boxes, labels, and difficulties cannot simply be stacked together in the batch. There would be no way of knowing which objects belong to which image.

Instead, we need to **pass a collating function to the `collate_fn` argument**, which instructs the `DataLoader` about how it should combine these varying size tensors. The simplest option would be to use Python lists.

### Base Convolutions


### Putting it all together

# Training
https://github.com/pytorch/vision/blob/master/references/detection/train.py



Before you begin, make sure to save the required data files for training and evaluation. To do this, run the contents of [`create_data_lists.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/create_data_lists.py) after pointing it to the `VOC2007` and `VOC2012` folders in your [downloaded data](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#download).

See [`train.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/train.py).

The parameters for the model (and training it) are at the beginning of the file, so you can easily check or modify them should you need to.

To **train your model from scratch**, run this file –

`python train.py`

To **resume training at a checkpoint**, point to the corresponding file with the `checkpoint` parameter at the beginning of the code.

python parse.py --complete_img_ids img_ids.txt --parent_directory ~/work/Test --path_to_predefined_classes ~/work/AST/object_detection/predefined_classes.txt --img_directory chips_positive --annotation_directory chips_positive_xml --train_val_percent 0.2

python model_train.py --parent_directory /home/jovyan/work/Test --path_to_predefined_classes /home/jovyan/work/AST/object_detection/predefined_classes.txt --batch_size 16 --pretrained True --keep_difficult True --scheduler_name exponentiallr --optimizer_name SGD --lr 0.005 --momentum 0.9 --milestones 5 --lr_gamma 0.1 --weight_decay 5e-4 --num_epochs 25 --num_workers 2 --val_size 0.95


Not used:
python voc2coco.py --ann_dir /home/jovyan/work/Test/chips_positive_xml --ann_ids /path/to/annotations/ids/list.txt --labels /home/jovyan/work/AST/object_detection/predefined_classes.txt --output /home/jovyan/work/Test/output.json --ext xml