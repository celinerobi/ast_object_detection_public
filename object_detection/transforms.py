import torch
import torchvision
import random 

import torchvision.transforms.functional as FT

import matplotlib.pyplot as plt
import matplotlib.patches as patches

    
def photometric_distort(image, p = 0.5):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image
    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]


    random.shuffle(distortions)

    for d in distortions:
        if random.random() < p:
            if d.__name__ == 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-0.5, 0.5)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)
            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image

def hflip(image, boxes, p = 0.5):
    """
    Flip image horizontally.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = image
    new_boxes = boxes 
    if random.random() < p:

        new_image = FT.hflip(new_image)

        # Flip boxes
        new_boxes = boxes
        new_boxes[:, 0] = new_image.width - boxes[:, 0] - 1
        new_boxes[:, 2] = new_image.width - boxes[:, 2] - 1
        new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes

def vflip(image, boxes, p = 0.5):
    """
    Flip image horizontally.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = image
    new_boxes = boxes 
    if random.random() < p:

        new_image = FT.vflip(new_image) 
        # Flip boxes
        new_boxes[:, 1] = new_image.height - boxes[:, 1] 
        new_boxes[:, 3] = new_image.height - boxes[:, 3] 
        new_boxes = new_boxes[:, [0, 3, 2, 1]]
    return new_image, new_boxes


def normalize():
    """
    Function to calculate the mean and standard deviation for the dataset to use
    for the normalization 
    
    loader = DataLoader(train_set, batch_size=len(train_set), num_workers=1)
    data = next(iter(loader))
    data[0].mean(), data[0].std()
    """
    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(trainloader):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    return(mean,std)

def get_transform(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    """
    Method specifies the level of transform
    """
    assert split.lower() in {'train', 'test'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties

    # Skip the following operations for evaluation/testing
    if split == 'train': #complete each transform with a 50% probability
        # A series of photometric distortions in random order
        new_image = photometric_distort(new_image)
        # Horizonally Flip image
        new_image, new_boxes = hflip(new_image, new_boxes)
        # Vertically Flip image
        new_image, new_boxes = vflip(new_image, new_boxes)

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)
    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    #new_image = FT.normalize(new_image, mean=mean, std=std)
    #torchvision.utils.draw_bounding_boxes(new_image,new_boxes)
    
    return new_image, new_boxes, new_labels, new_difficulties

def draw_transforms(image, boxes):
    """
    Draws detections in output image and stores this.
    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # Create plot
    img = FT.to_pil_image(image)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Bounding-box colors
    for x1, y1, x2, y2, in boxes:
        box_w = x2 - x1
        box_h = y2 - y1

        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor="gray", facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        
        # Save generated image with detections
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)



