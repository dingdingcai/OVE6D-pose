import json
import torch
from pathlib import Path


class Dataset():
    def __init__(self, data_dir, type='recon'):
        """
        type[cad, recon]: using cad model or reconstructed model
        """
        super().__init__()
        assert(type == 'cad' or type == 'recon'), "only support CAD model (cad) or reconstructed model (recon)"
        self.cam_file = Path(data_dir) / 'camera_primesense.json'
        with open(self.cam_file, 'r') as cam_f:
            self.cam_info = json.load(cam_f)

# The below is the ground truth camera information of this dataset, which is supposed to be utilized to generate the codebook
        # self.cam_K = torch.tensor([
        #     [self.cam_info['fx'], 0, self.cam_info['cx']],
        #     [0.0, self.cam_info['fy'], self.cam_info['cy']],
        #     [0.0, 0.0, 1.0]
        # ], dtype=torch.float32)
        # self.cam_height = self.cam_info['height']
        # self.cam_width = self.cam_info['width']

# But we use by chance the below information (of test_primesense/01/rgb/190.png) to generate object codebooks in our paper
        self.cam_K = torch.tensor([
            [1.0757e+03, 0.0000e+00, 3.6607e+02],
            [0.0000e+00, 1.0739e+03, 2.8972e+02],
            [0.0000e+00, 0.0000e+00, 1.0000e+00],
        ], dtype=torch.float32)
        self.cam_height = 540
        self.cam_width = 720
        

        if type == "recon":
            self.model_dir = Path(data_dir) / 'models_reconst'
        else:
            self.model_dir = Path(data_dir) / 'models_cad'
        
        
        self.model_info_file = self.model_dir / 'models_info.json'

        # self.cam_height = 540
        # self.cam_width = 720
        # self.depth_scale = 0.1
        
        with open(self.model_info_file, 'r') as model_f:
            self.model_info = json.load(model_f)
        
        self.obj_model_file = dict()
        self.obj_diameter = dict()

        for model_file in sorted(self.model_dir.iterdir()):
            if str(model_file).endswith('.ply'):
                obj_id = int(model_file.name.split('_')[-1].split('.')[0])
                self.obj_model_file[obj_id] = model_file
                self.obj_diameter[obj_id] = self.model_info[str(obj_id)]['diameter']

                