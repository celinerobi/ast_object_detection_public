import time
import torchvision
import torch
import os
## Model

def save_checkpoint(epoch, model, optimizer, lr, lr_decay, num_epochs, path):
    """
    Save model checkpoint.
    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    model_characteristic = "".join(["lr",str(lr),"_","lrdecay",str(lr_decay),"_","epoch",str(num_epochs)])
    filename = 'checkpoint_frcnn.pth.tar'
    state = {'epoch': epoch,
             'model': model,
             'optimizer_stat_dict': optimizer}

    torch.save(state, os.path.join(path, model_characteristic + filename))
    
def get_frcnn_model(num_classes, pretrained):
    #Anchor Generator              
    #rpn_anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((16, 32, 64, 128, 256, 512),),
    #                                                                        aspect_ratios=((0.5, 1.0, 2.0),))
    # load a model pre-trained pre-trained on COCO
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(rpn_anchor_generator=rpn_anchor_generator, pretrained = pretrained)

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = pretrained)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

### Training
def make_optimizer(optimizer_name, model, **kwargs):
    if optimizer_name=='Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=kwargs['lr'])
    elif optimizer_name=='SGD':
        optimizer = torch.optim.SGD(model.parameters(),lr=kwargs['lr'],momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer
    
def make_scheduler(scheduler_name, optimizer, **kwargs):
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs['milestones'], gamma= kwargs['lr_gamma'])
    elif scheduler_name == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = kwargs['T_max'])    
    elif scheduler_name == 'exponentiallr':
        lr_scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = kwargs['lr_gamma'])
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR, ExponentialLR, and CosineAnnealingLR "
                           "are supported.".format(lr_scheduler))
    return lr_scheduler