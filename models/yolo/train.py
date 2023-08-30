from ultralytics import YOLO

yolo_version = 'yolov8s'
dataset_yaml = '/data/emre/ms/vindr/dataset/yolo/vindr-mammo-2-0.yaml'
cfg = '/data/emre/ms/vindr/model/yolo/my-cfg.yaml'

# Load a model
#model = YOLO('{}.yaml'.format(yolo_version))  # build a new model from YAML
model = YOLO('{}.pt'.format(yolo_version))  # load a pretrained model (recommended for training)
#model = YOLO('{}.yaml'.format(yolo_version)).load('{}.pt'.format(yolo_version))  # build from YAML and transfer weights

# Train the model
results = model.train(data=dataset_yaml, project='vindr-mammo-study', cfg=cfg)