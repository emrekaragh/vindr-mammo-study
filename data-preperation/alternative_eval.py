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
