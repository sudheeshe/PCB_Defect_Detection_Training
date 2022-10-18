
# Printed Circuit Board Defect Detection

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/3.jpg?raw=true)

### What is PCB ...??

- `Printed circuit boards (PCBs)` are the foundational building block of most modern electronic devices.  
- Whether simple single layered boards used in your garage door opener, to the six layer board in your smart watch, to a 60 layer, very high density and high-speed circuit boards used in super computers and servers, printed circuit boards are the foundation on which all of the other electronic components are assembled onto.

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/1.jpg?raw=true)

- Compared to traditional wired circuits, PCBs offer a number of advantages. Their small and lightweight design is appropriate for use in many modern devices, while their reliability and ease of maintenance suit them for integration in complex systems. 
- Additionally, their low cost of production makes them a highly cost-effective option.

## Business Scenario

- The Client is looking for an Effective PCB defects detection System which detect the following defects.

        1. missing_hole
        2. mouse_bite
        3. open_circuit
        4. short
        5. spur
        6. spurious_copper


## Data Understanding

- The available dataset have total 485 images for Training.
- 139 images for Validation and 69 images are provided for testing
- On an average 90 images per 6 defect class.


Let's see some sample from training data

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/4.jpg?raw=true)


## Data Labeling

- Labeling was done with `LabelImg` tool and labels are saved on `.txt` format

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/5.jpg?raw=true)


## Model Building and Evaluation

- Used YoloV5 model for detection.
- YoloV5 trained from scratch to 1500 epochs on Paperspace P4000 GPU.
- YoloV5 does image augmentation internally on training images.
- YOLOv5 applies online imagespace and colorspace augmentations in the trainloader (but not the val_loader) to present a new and unique augmented Mosaic (original image + 3 random images) each time an image is loaded for training. Images are never presented twice in the same way.

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/6.jpg?raw=true)

Let's visualize some of our `training image batch` and `validation image batch`

#### Training image batch with Mosaic augmentation applied

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/train_tile_batch.jpg?raw=true)

#### Validation image batch

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/imgs_readme/val_tile_batch.jpg?raw=true)


- Let's see the `mAP` for first `100 epochs`.

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/Reports/100_epoch_eport.png?raw=true)


- The mAP for first `1500 epochs` was `0.811`. 
- The model didn't show any improvement in mAP beyond 1500 epochs.

#### Precision-Recall Curve

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/YoloV5_Training/yolov5/runs/train/yolov5s_results8/PR_curve.png?raw=true)

#### F1 Curve

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/YoloV5_Training/yolov5/runs/train/yolov5s_results8/F1_curve.png?raw=true)

- The F1 curve shows that any threshold (confidence) value between 0.2 to almost 0.6 gives better results from the model

#### Confusion Matrix

![alt text](https://github.com/sudheeshe/PCB_Defect_Detection_Training/blob/main/YoloV5_Training/yolov5/runs/train/yolov5s_results8/confusion_matrix.png?raw=true)

- The model has less prediction power on `spurious_copper` class and very high confidence on `missing_hole` and other classes have decent prediction capability.

## Prediction Images





References:
### Precision-Recall curve blog 
[click here](https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248#:~:text=In%20a%20perfect%20classifier%2C%20AUC,have%20AUC%2DPR%20%3D%200.5.)
