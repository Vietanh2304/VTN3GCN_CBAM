from utils.video_augmentation import *
from dataset.vtn_att_poseflow_model_dataset import VTN_ATT_PF_Dataset, VTN_GCN_Dataset, VTN_RGBheat_Dataset
from .three_viewpoints import ThreeViewsData
from dataset.i3d import InceptionI3D_Data
from dataset.swin_transformer import SwinTransformer
from dataset.mvit import MVIT
from dataset.vtn_hc_pf_three_view import VTNHCPF_ThreeViewsData,VTN3GCNData
from dataset.distilation import Distilation

def build_dataset(dataset_cfg, split,model = None,**kwargs):
    dataset = None

    if dataset_cfg['model_name'] == 'VTNGCN' or dataset_cfg['model_name'] == 'VTNGCN_Combine':
        dataset = VTN_GCN_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)
    if dataset_cfg['model_name'] == 'VTN_RGBheat' or dataset_cfg['model_name'] == '2s-CrossVTN':
        dataset = VTN_RGBheat_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)
    if dataset_cfg['model_name'] == 'VTN3GCN' or dataset_cfg['model_name'] == 'VTN3GCN_v2':
        dataset = VTN3GCNData(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == "vtn_att_poseflow" or 'HandCrop' in dataset_cfg['model_name'] or dataset_cfg['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference':
        dataset = VTN_ATT_PF_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    distillation_models = ['MvitV2_OneView_Sim_Knowledge_Distillation','I3D_OneView_Sim_Knowledge_Distillation','VideoSwinTransformer_OneView_Sim_Knowledge_Distillation']
    if 'ThreeView' in dataset_cfg['model_name'] or dataset_cfg['model_name'] in distillation_models:
        dataset = ThreeViewsData(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == "InceptionI3d" or dataset_cfg['model_name'] == 'I3D_OneView_Sim_Knowledge_Distillation_Inference':
        dataset = InceptionI3D_Data(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == "swin_transformer" or dataset_cfg['model_name'] == 'VideoSwinTransformer_OneView_Sim_Knowledge_Distillation_Inference':
        dataset = SwinTransformer(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)
    if 'mvit' in dataset_cfg['model_name'] or dataset_cfg['model_name'] == 'MvitV2_OneView_Sim_Knowledge_Distillation_Inference':
        dataset = MVIT(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    if dataset_cfg['model_name'] == 'VTNHCPF_Three_view' or dataset_cfg['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation':
        dataset = VTNHCPF_ThreeViewsData(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    assert dataset is not None
    return dataset



    