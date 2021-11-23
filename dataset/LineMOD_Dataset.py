import json
import torch
from pathlib import Path

class Dataset():
    def __init__(self, data_dir):
        self.model_dir = Path(data_dir) / 'models_eval'
        self.cam_file = Path(data_dir) / 'camera.json'
        
        with open(self.cam_file, 'r') as cam_f:
            self.cam_info = json.load(cam_f)
        
        self.cam_K = torch.tensor([
            [self.cam_info['fx'], 0, self.cam_info['cx']],
            [0.0, self.cam_info['fy'], self.cam_info['cy']],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        self.cam_height = self.cam_info['height']
        self.cam_width = self.cam_info['width']
        
        self.model_info_file = self.model_dir / 'models_info.json'
        with open(self.model_info_file, 'r') as model_f:
            self.model_info = json.load(model_f)
            
        self.obj_model_file = dict()
        self.obj_diameter = dict()
        
        for model_file in sorted(self.model_dir.iterdir()):
            if str(model_file).endswith('.ply'):
                obj_id = int(model_file.name.split('_')[-1].split('.')[0])
                self.obj_model_file[obj_id] = model_file
                self.obj_diameter[obj_id] = self.model_info[str(obj_id)]['diameter']
