from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load the ground truth annotations (coco_gt) and detection results (coco_dt)
coco_gt = COCO('/home/emrekara/yl/vindr-mammo-study/data-preperation/data/out/coco/2_1/vindr-mammo-coco-2-1-val.json')
coco_dt = COCO('/data/emre/ms/vindr/model/yolo/runs/detect/val_v8m-22-yolo-2-1/predictions_coco.json')

# Create a COCO evaluation object
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

# Run the evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Access evaluation metrics, e.g., mAP
mAP = coco_eval.stats[0]
print(coco_eval.stats)

"""
coco_gt = COCO('/home/emrekara/yl/vindr-mammo-study/data-preperation/data/out/coco/2_1/vindr-mammo-coco-2-1-val.json')
coco_dt = COCO('/data/emre/ms/vindr/model/yolo/runs/detect/val_v8m-22-yolo-2-1/predictions_coco.json')
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.105
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.214
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.098
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.036
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.114
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.276
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.200
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.484
[0.10478393 0.21425201 0.09830284 0.         0.03604372 0.11435435
 0.27575758 0.43123543 0.45384615 0.         0.2        0.4838961 ]
"""