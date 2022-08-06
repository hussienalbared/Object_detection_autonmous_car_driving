# %%
from torchvision import datasets, models, transforms  
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights,fasterrcnn_resnet50_fpn,fasterrcnn_resnet50_fpn,fasterrcnn_mobilenet_v3_large_320_fpn,retinanet_resnet50_fpn
import torchvision.io as io
import torch
import numpy as np
import argparse
import pickle
import torch
import cv2
import torchvision
from PIL import Image
import logging

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image



# %%
# model=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
MODELS = {
	"frcnn-resnet": fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": retinanet_resnet50_fpn
}
# load the model and set it to evaluation mode
# model = fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
# 	num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model = fasterrcnn_resnet50_fpn(pretrained=True, progress=True, pretrained_backbone=True).to(DEVICE)
model.eval()

# %%
def predict(input_tensor, model, device, detection_threshold):
    logging.info('predict Executed')
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    
    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices

def draw_boxes(boxes, labels, classes, image):
    logging.info('draw_boxes Executed')
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']


# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# %%
# image = np.array(cv2.imread('cat_dog.jpeg'))

def run_detection(image):
    print('run_detection Executed')
    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    input_tensor = transform(image)
    print(input_tensor.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval().to(device)

    # Run the model and display the detections
    boxes, classes, labels, indices = predict(input_tensor, model, device, 0.6)
    image = draw_boxes(boxes, labels, classes, image)

    # Show the image:
    Image.fromarray(image)
    # cv2.imwrite(filename, image)
    print('run_detection finished')
    return image
# %%
def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

# %%
def cam_(image,model,labels,boxes,input_tensor,classes):
    image_float_np = np.float32(image) / 255
    target_layers = [model.backbone]
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    cam = EigenCAM(model,
                target_layers, 
                use_cuda=torch.cuda.is_available(),
                reshape_transform=fasterrcnn_reshape_transform)

    grayscale_cam = cam(input_tensor, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    Image.fromarray(image_with_bounding_boxes)
    return image_with_bounding_boxes

# %% [markdown]
# 
def detect_object_and_draw_decision_visulaization(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    input_tensor = transform(image)
    print(input_tensor.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval().to(device)

    # Run the model and display the detections
    boxes, classes, labels, indices = predict(input_tensor, model, device, 0.6)
    image = draw_boxes(boxes, labels, classes, image)
    print('XXXXX:{}'.format(type(image)))
    image_with_cam=cam_(image,model,labels,boxes,input_tensor,classes)
    return image,image_with_cam

    

