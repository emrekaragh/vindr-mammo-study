from pathlib import Path
import json

def calculate_area(x, y, w, h):
    return (w*h)

yolo_preds_path = Path('/data/emre/ms/vindr/model/yolo/runs/detect/val_v8m-22-yolo-2-1/predictions.json')
image_id_map_path = Path('/home/emrekara/yl/vindr-mammo-study/data-preperation/data/out/coco/2_1/image_id_map.json')
output_path = Path('/data/emre/ms/vindr/model/yolo/runs/detect/val_v8m-22-yolo-2-1/predictions_coco.json')

yolo_preds = None
with open(yolo_preds_path, 'r') as f:
    yolo_preds = json.load(f)

image_id_map = None
with open(image_id_map_path, 'r') as f:
    image_id_map = json.load(f)

output = {}
output['annotations'] = []
for index, pred in enumerate(yolo_preds):
    pred['category_id'] = 1
    pred['id'] = index + 1
    pred['image_id'] = image_id_map[pred['image_id']]
    pred['bbox'] = [int(item) for item in pred['bbox']]
    pred['iscrowd'] = 0
    pred['area'] = calculate_area(*pred['bbox'])
    output['annotations'].append(pred)

with open(output_path, 'w') as f:
    json.dump(output, f, indent=4)






