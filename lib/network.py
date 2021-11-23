
import torch
import math
from torch import nn
import torch.nn.functional as F
from lib.geometry import inplane_2D_spatial_transform
from lib import preprocess


class OVE6D(nn.Module):
    def __init__(self):
        super(OVE6D, self).__init__() 
        ###################################### backbone ############################################
        self.stem_layer1 = nn.Sequential(
                      nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(16), 
                      nn.ReLU())
        self.stem_layer2 = nn.Sequential(
                      nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(64), 
                      nn.ReLU())
        self.stem_layer3 = nn.Sequential(
                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(64), 
                      nn.ReLU())
        self.stem_layer4 = nn.Sequential(
                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(128), 
                      nn.ReLU())
        self.stem_layer5 = nn.Sequential(
                      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(128), 
                      nn.ReLU())
        self.stem_layer6 = nn.Sequential(
                      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(256), 
                      nn.ReLU())
        self.stem_layer7 = nn.Sequential(
                      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(256), 
                      nn.ReLU())
        self.stem_layer8 = nn.Sequential(
                      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(512), 
                      nn.ReLU())
        self.backbone_layers = list()
        self.backbone_layers.append(self.stem_layer1)
        self.backbone_layers.append(self.stem_layer2)
        self.backbone_layers.append(self.stem_layer3)
        self.backbone_layers.append(self.stem_layer4)
        self.backbone_layers.append(self.stem_layer5)
        self.backbone_layers.append(self.stem_layer6)
        self.backbone_layers.append(self.stem_layer7)
        self.backbone_layers.append(self.stem_layer8)
        ###################################### backbone ############################################

        ################################# viewpoint encoder head ########################################
        self.vp_enc_transition = nn.Sequential(
                        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
                        nn.BatchNorm2d(256), 
                        nn.ReLU())
        self.vp_enc_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.vp_enc_fc = nn.Linear(in_features=256, out_features=64)
        ################################# viewpoint encoder head ########################################


        ################################ in-plane transformation regression #######################################
        self.vp_inp_transition = nn.Sequential(
                        nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),
                        nn.BatchNorm2d(128), 
                        nn.ReLU())
        
        self.vp_rot_fc1 = nn.Sequential(
                        nn.Linear(in_features=4096, out_features=128),
                        nn.ReLU())
        self.vp_rot_fc2 = nn.Linear(in_features=128, out_features=2)
        
        self.vp_tls_fc1 = nn.Sequential(
                            nn.Linear(in_features=4096, out_features=128),
                            nn.ReLU())
        self.vp_tls_fc2 = nn.Linear(in_features=128, out_features=2)

        ################################  in-plane transformation regression #######################################


        ############################# orientation confidence #####################################
        self.vp_conf_layer1 = nn.Sequential(
                            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(128), 
                            nn.ReLU())
        self.vp_conf_layer2 = nn.Sequential(
                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(128), 
                            nn.ReLU())
        self.vp_conf_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.vp_conf_fc = nn.Linear(128, 1)
        ############################# orientation confidence #####################################

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def backbone(self, x):
        """
        The backbone network for extracting features
        """
        H, W = x.shape[-2:]
        x = x.view(-1, 1, H, W)
        for layer in self.backbone_layers:
            shortcut = x
            x = layer(x)
            if x.shape == shortcut.shape:
                x += shortcut
        return x
    
    def viewpoint_encoder_head(self, x):
        """
        encoder head for extracting viewpoint representation
        """
        x = self.vp_enc_transition(x)
        x = self.vp_enc_pool(x)       # B x c x 1 x 1
        x = x.view(x.shape[0], -1)     # BV x CHW
        x = self.vp_enc_fc(x)
        x = F.normalize(x, dim=1)
        return x

    def vipri_encoder(self, x, return_maps=False):
        """
        viewpoint in-plane rotation invariant encoding
        """
        ft_map = self.backbone(x) # B x 1 x H x W => B x C x h x w
        vp_enc = self.viewpoint_encoder_head(ft_map)
        if return_maps:
            vp_map = self.vp_inp_transition(ft_map)
            return vp_map, vp_enc
        return vp_enc

    def regression_head(self, x, y):
        """
        regression head for 2D in-plane transformation from x to y
        """
        bs, ch = x.shape[:2]
        x = F.normalize(x.view(bs, -1), dim=1).view(bs, ch, -1)
        y = F.normalize(y.view(bs, -1), dim=1).view(bs, ch, -1)
        # z = (x.unsqueeze(3) * y.unsqueeze(2)).sum(dim=1) # BV x C x 64 x 64
        z = torch.bmm(x.permute(0, 2, 1), y) # BV x 64 x 64, feature map correlation
        z = z.view(bs, -1) # BV x 4096

        Rz = self.vp_rot_fc1(z)        # BV x 4096 -> BV x 128
        Rz = self.vp_rot_fc2(Rz)       # BV x 128 -> BV x 2
        Rz = F.normalize(Rz, dim=1)

        TxTy = self.vp_tls_fc1(z)      # Bx4096 -> Bx128
        TxTy = self.vp_tls_fc2(TxTy)   # Bx128  -> Bx2 -> Bx1x2
        TxTy = torch.tanh(TxTy)        # range[-1.0, 1.0]

        row1 = torch.stack([Rz[:, 0], -Rz[:, 1], TxTy[:, 0]], dim=1) # cos(theta), -sin(theta), BV x 3
        row2 = torch.stack([Rz[:, 1], Rz[:, 0], TxTy[:, 1]], dim=1)  # sin(theta), cos(theta),  BV x 3

        theta = torch.stack([row1, row2], dim=1)           # BV x 2 x 3, 2D in-plane transformation matrix

        return theta

    def spatial_transformation(self, x, theta):
        """
        transform feature maps with the given transformation matrix
        x: BxCxHxW
        theta: Bx2x3
        """
        stn_theta = theta.clone() # Bx2x3
        y = preprocess.spatial_transform_2D(x=x, theta=stn_theta, 
                                            mode='bilinear', 
                                            padding_mode='border', 
                                            align_corners=False)
        return y

    def viewpoint_confidence(self, x, y):
        """
        calcuate the consistency
        """
        z = torch.cat([x, y], dim=1)  # Bx2Cx8x8
        z = self.vp_conf_layer1(z)
        z = self.vp_conf_layer2(z)
        z = self.vp_conf_pool(z).view(z.size(0), -1) # BxCx1x1 -> BxC
        z = self.vp_conf_fc(z)                       # Bx256 -> Bx1
        z = torch.sigmoid(z)
        return z

    def inference(self, anc_map, inp_map):
        pd_theta = self.regression_head(x=anc_map, y=inp_map)
        stn_inp_map = self.spatial_transformation(x=anc_map, theta=pd_theta) # transform anchor viewpoint with in-plane rotation
        pd_conf = self.viewpoint_confidence(x=inp_map, y=stn_inp_map)
        return pd_theta, pd_conf

    def forward(self, x_anc_gt, x_oup_gt, x_inp_aug, x_oup_aug, inp_gt_theta):
        """
        input:
           x_anc_gt: rendered clean anchor viewpoint depth: B x V x H x W 
           x_inp_gt: rendered clean inplane rotated depth (z-axis)
           x_inp_aug: augmented x_inp_gt 
           x_oup_aug: augmented out-of-inplane rotated depth (xy-axis)
        return:
            viewpoint embeddings for the viewpoint triplets (Nx64)
            in-plane rotation of intra-viewpoint pair (Nx2x3)
            feature maps of the rendered and transformed of intra-viewpoint (Nx128x8x8)
        """
        # feature extractions and viewpoint embeddings
        z_anc_gt_map, z_anc_gt_vec = self.vipri_encoder(x_anc_gt, return_maps=True)   # BV x 1 x 128 x 128 --> BV x 512 x 8 x 8
        z_oup_gt_map, _ = self.vipri_encoder(x_oup_gt, return_maps=True)   # BV x 1 x 128 x 128 --> BV x 512 x 8 x 8

        z_oup_aug_vec = self.vipri_encoder(x_oup_aug, return_maps=False)
        z_inp_aug_map, z_inp_aug_vec = self.vipri_encoder(x_inp_aug, return_maps=True)
        
        # the regression branch is only trained with in-plane views
        inp_pd_theta = self.regression_head(x=z_anc_gt_map, y=z_inp_aug_map)
        oup_pd_theta = self.regression_head(x=z_oup_gt_map, y=z_inp_aug_map)
        oup_pd_theta = oup_pd_theta.detach() # No grad for training the regression branc

        # the transformed anchor feature map is supposed to be equal to the gt inplane feature maps
        gt_stn_inp_map = self.spatial_transformation(x=z_anc_gt_map, theta=inp_gt_theta) # transform the feature map of anchor view
        # the transformation branch is trained with GT transformation only
        pd_stn_inp_map = self.spatial_transformation(x=z_anc_gt_map, theta=inp_pd_theta)
        pd_stn_oup_map = self.spatial_transformation(x=z_oup_gt_map, theta=oup_pd_theta)

        z_inp_aug_map = z_inp_aug_map.detach()
        gt_stn_inp_map = gt_stn_inp_map.detach()
        pd_stn_inp_map = pd_stn_inp_map.detach()
        pd_stn_oup_map = pd_stn_oup_map.detach()

        # the confidence branch is only trained with the predicted feature maps
        alpha = 0.2
        pd_stn_mix_map = alpha * pd_stn_inp_map + (1 - alpha) * pd_stn_oup_map
        pd_mix_cls = self.viewpoint_confidence(z_inp_aug_map, pd_stn_mix_map)

        gt_inp_cls = self.viewpoint_confidence(x=z_inp_aug_map, y=gt_stn_inp_map)
        pd_inp_cls = self.viewpoint_confidence(x=z_inp_aug_map, y=pd_stn_inp_map)
        pd_oup_cls = self.viewpoint_confidence(x=z_inp_aug_map, y=pd_stn_oup_map)
        
        return (inp_pd_theta, 
                gt_inp_cls, pd_inp_cls, pd_oup_cls, pd_mix_cls,
                z_anc_gt_vec, z_inp_aug_vec, z_oup_aug_vec) # viewpoint embedding triplets





