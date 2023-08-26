import numpy as np
import os
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import ops

# change these, of course :)
SAVE_TEST_IMG = False
model_path = 'yolov8n.pt'
img_path = 'bus.jpg'

class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

# we are going to register a PyTorch hook on the important parts of the YOLO model,
# then reverse engineer the outputs to get boxes and logits
# first, we have to register the hooks to the model *before running inference*
# then, when inference is run, the hooks will save the inputs/outputs of their respective modules
model = YOLO(model_path)
detect = None
cv2_hooks = None
cv3_hooks = None
detect_hook = SaveIO()
for i, module in enumerate(model.model.modules()):
    if type(module) is Detect:
        module.register_forward_hook(detect_hook)
        detect = module

        cv2_hooks = [SaveIO() for _ in range(module.nl)]
        cv3_hooks = [SaveIO() for _ in range(module.nl)]
        for i in range(module.nl):
            module.cv2[i].register_forward_hook(cv2_hooks[i])
            module.cv3[i].register_forward_hook(cv3_hooks[i])
        break
input_hook = SaveIO()
model.model.register_forward_hook(input_hook)

# run inference
results = model(img_path)

# now reverse engineer the outputs to find the logits
# see Detect.forward(): https://github.com/ultralytics/ultralytics/blob/b638c4ed9a24270a6875cdd47d9eeda99204ef5a/ultralytics/nn/modules/head.py#L22
shape = detect_hook.input[0][0].shape  # BCHW
x = []
for i in range(detect.nl):
    x.append(torch.cat((cv2_hooks[i].output, cv3_hooks[i].output), 1))
x_cat = torch.cat([xi.view(shape[0], detect.no, -1) for xi in x], 2)
box, cls = x_cat.split((detect.reg_max * 4, detect.nc), 1)

# assumes batch size = 1 (i.e. you are just running with one image)
# if you want to run with many images, throw this in a loop
batch_idx = 0
xywh_sigmoid = detect_hook.output[0][batch_idx]
all_logits = cls[batch_idx]

# figure out the original img shape and model img shape so we can transform the boxes
img_shape = input_hook.input[0].shape[2:]
orig_img_shape = model.predictor.batch[1][batch_idx].shape[:2]

## **FINAL OUTPUTS -- DO WHAT YOU WANT WITH THIS** ##
# dictionary will have:
# bbox - [x0,y1,x1,y1], in original image coordinate space
# logits - raw logits vector, one entry per class
# activations - pred scores after calling logits.sigmoid() - this is what is usually returned
boxes = []

for i in range(xywh_sigmoid.shape[-1]): # for each predicted box...
    x0, y0, x1, y1, *class_probs_after_sigmoid = xywh_sigmoid[:,i]
    x0, y0, x1, y1 = ops.scale_boxes(img_shape, np.array([x0.cpu(), y0.cpu(), x1.cpu(), y1.cpu()]), orig_img_shape)
    logits = all_logits[:,i]
    boxes.append({
        'bbox': [x0.item(), y0.item(), x1.item(), y1.item()],
        'logits': logits.cpu().tolist(),
        'activations': [p.item() for p in class_probs_after_sigmoid]
    })

print("Processed", len(boxes), "boxes")
print("The first one is", boxes[0])

if SAVE_TEST_IMG:
    import matplotlib.pyplot as plt
    import cv2
    img = cv2.imread(img_path)
    for box in boxes:
        if max(box['probs_after_sigmoid']) > 0.5:
            x0, y0, x1, y1 = [int(b) for b in box['bbox']]
            img = cv2.rectangle(img,(x0,y0),(x1,y1),(0,255,0),3)

    plt.imshow(img[:,:,::-1])
    plt.savefig(f'{os.path.basename(img_path)}_test.jpg')