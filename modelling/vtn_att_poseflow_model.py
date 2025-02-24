import torch
from torch import nn

from .vtn_utils import FeatureExtractor, FeatureExtractorGCN, LinearClassifier, SelfAttention, CrossAttention
import torch.nn.functional as F
from pytorch_lightning.utilities.migration import pl_legacy_patch

class MMTensorNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim).unsqueeze(self.dim)
        std = torch.std(x, dim=self.dim).unsqueeze(self.dim)
        return (x - mean) / std


class VTNHCPF(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VTNHCPF")
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.num_attn_features  = embed_size
        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers)
        self.norm = MMTensorNorm(-1)
        self.bottle_mm = nn.Linear(106 + self.num_attn_features, self.num_attn_features)

        self.self_attention_decoder = SelfAttention(self.num_attn_features, self.num_attn_features,
                                                    [num_heads] * num_layers,
                                                    self.sequence_length, layer_norm=True,dropout = dropout)
        self.classifier = LinearClassifier(self.num_attn_features, self.num_classes, dropout)
        self.dropout = dropout
        self.relu = F.relu

    def reset_head(self,num_classes):
        self.classifier = LinearClassifier(self.num_attn_features, num_classes, self.dropout)
        print("Reset to ",num_classes)

    def forward_features(self, features = None,poseflow = None):

        # Reshape to put both hand crops on the same axis.
       
        zp = torch.cat((features, poseflow), dim=-1)

        zp = self.norm(zp)
        zp = self.relu(self.bottle_mm(zp))

        zp = self.self_attention_decoder(zp)

        return zp

    def forward(self, clip = None,poseflow = None,**kwargs):
        """Extract the image feature vectors."""
        rgb_clip, pose_clip = clip,poseflow

        # Reshape to put both hand crops on the same axis.
        b, t, x, c, h, w = rgb_clip.size()
        rgb_clip = rgb_clip.view(b, t * x, c, h, w)
        z = self.feature_extractor(rgb_clip)
        # Reshape back to extract features of both wrist crops as one feature vector.
        z = z.view(b, t, -1)

        zp = torch.cat((z, pose_clip), dim=-1)

        zp = self.norm(zp)
        zp = self.relu(self.bottle_mm(zp))

        zp = self.self_attention_decoder(zp)
        y = self.classifier(zp)

        return {'logits':y.mean(1)} # train
        # return y.mean(1) # convert to script


class VTNHCPF_GCN(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 gcn='AAGCN', freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VTNHCPF_GCN")
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers)
        self.feature_extractor_gcn = FeatureExtractorGCN(gcn, freeze_layers)

        num_attn_features = embed_size
        num_gcn_features = int(embed_size/2)
        pose_flow_features = 106
        # pose_flow_features = 0
        add_attn_features = 0
        self.norm = MMTensorNorm(-1)
        self.bottle_mm = nn.Linear(pose_flow_features + num_attn_features + num_gcn_features, num_attn_features + add_attn_features)

        self.self_attention_decoder = SelfAttention(num_attn_features + add_attn_features, num_attn_features + add_attn_features,
                                                    [num_heads] * num_layers,
                                                    self.sequence_length, layer_norm=True,dropout = dropout)
        self.classifier = LinearClassifier(num_attn_features + add_attn_features, num_classes, dropout)
        self.num_attn_features  = num_attn_features + add_attn_features
        self.dropout = dropout
        self.num_classes = num_classes
        self.relu = F.relu

    def reset_head(self,num_classes):
        self.classifier = LinearClassifier(self.num_attn_features, num_classes, self.dropout)
        print("Reset to ",num_classes)

    def forward_features(self, features = None,poseflow = None, features_keypoint = None):

        # Reshape to put both hand crops on the same axis.
       
        zp = torch.cat((features,features_keypoint,poseflow), dim=-1)

        zp = self.norm(zp)
        zp = self.relu(self.bottle_mm(zp))

        zp = self.self_attention_decoder(zp)

        return zp
    
    def forward(self, clip = None,poseflow = None,keypoints = None,**kwargs):
        """Extract the image feature vectors."""
        rgb_clip, pose_clip = clip,poseflow

        # Reshape to put both hand crops on the same axis.
        b, t, x, c, h, w = rgb_clip.size()
        rgb_clip = rgb_clip.view(b, t * x, c, h, w)
        zc = self.feature_extractor(rgb_clip)
        # Reshape back to extract features of both wrist crops as one feature vector.
        zc = zc.view(b, t, -1)    
        zk = self.feature_extractor_gcn(keypoints)

        zp = torch.cat((zc,pose_clip,zk), dim=-1)

        zp = self.norm(zp)
        zp = self.relu(self.bottle_mm(zp))

        zp = self.self_attention_decoder(zp)

        y = self.classifier(zp)

        return {'logits':y.mean(1)} # train
        # return y.mean(1) # convert to script



class VTNHCPF_Three_View(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VTNHCPF_Three_View")
        self.center = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.left = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.right = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.classifier = LinearClassifier(embed_size*2*3, num_classes, dropout)
        self.feature_extractor = None
    def add_backbone(self):
        self.feature_extractor = self.center.feature_extractor

    def remove_head_and_backbone(self):
        self.center.feature_extractor = nn.Identity()
        self.left.feature_extractor = nn.Identity()
        self.right.feature_extractor = nn.Identity()
        self.center.classifier = nn.Identity()
        self.left.classifier = nn.Identity()
        self.right.classifier = nn.Identity()
        print("Remove head and backbone")
    
    def freeze(self,layers = 2):
        print(f"Freeze {layers} layers attn")
        for i in range(layers):
            for param in self.center.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
            for param in self.left.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
            for param in self.right.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
    def forward_features(self,left = None,center = None,right = None,center_pf = None,left_pf = None,right_pf = None): 
        b, t, x, c, h, w = left.size()
        left_feature = self.feature_extractor(left.view(b, t * x, c, h, w)).view(b,t,-1)
        right_feature = self.feature_extractor(right.view(b, t * x, c, h, w)).view(b,t,-1)
        center_feature = self.feature_extractor(center.view(b, t * x, c, h, w)).view(b,t,-1)
        
        left_ft = self.left.forward_features(left_feature,left_pf).mean(1)
        center_ft = self.center.forward_features(center_feature,center_pf).mean(1)
        right_ft = self.right.forward_features(right_feature,right_pf).mean(1)
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)
       
        return output_features
    def forward(self,left = None,center = None,right = None,center_pf = None,left_pf = None,right_pf = None):  
        b, t, x, c, h, w = left.size()
        left_feature = self.feature_extractor(left.view(b, t * x, c, h, w)).view(b,t,-1)
        right_feature = self.feature_extractor(right.view(b, t * x, c, h, w)).view(b,t,-1)
        center_feature = self.feature_extractor(center.view(b, t * x, c, h, w)).view(b,t,-1)
        
        left_ft = self.left.forward_features(left_feature,left_pf).mean(1)
        center_ft = self.center.forward_features(center_feature,center_pf).mean(1)
        right_ft = self.right.forward_features(right_feature,right_pf).mean(1)
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)
       

        y = self.classifier(output_features)

        return {
            'logits':y
        }

class VTN3GCN(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 gcn='AAGCN', freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VTN3GCN")
        self.center = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.left = VTNHCPF_GCN(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,gcn,freeze_layers,dropout,**kwargs)
        self.right = VTNHCPF_GCN(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,gcn,freeze_layers,dropout,**kwargs)
        self.classifier = LinearClassifier(embed_size*3, num_classes, dropout)
        self.embed_size = embed_size
        self.feature_extractor = None
        self.feature_extractor_gcn_right = None
        self.feature_extractor_gcn_left = None
    def add_backbone(self):
        self.feature_extractor = self.center.feature_extractor

        self.feature_extractor_gcn_right = self.right.feature_extractor_gcn
        self.feature_extractor_gcn_left = self.left.feature_extractor_gcn

    def remove_head_and_backbone(self):
        self.center.feature_extractor = nn.Identity()
        self.left.feature_extractor = nn.Identity()
        self.right.feature_extractor = nn.Identity()

        self.left.feature_extractor_gcn = nn.Identity()
        self.right.feature_extractor_gcn = nn.Identity()

        self.center.classifier = nn.Identity()
        self.left.classifier = nn.Identity()
        self.right.classifier = nn.Identity()
        print("Remove head and backbone")
    
    def freeze(self,layers = 2):
        print(f"Freeze {layers} layers attn")
        for i in range(layers):
            for param in self.center.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
            for param in self.left.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
            for param in self.right.self_attention_decoder.layers[i].parameters():
                param.requires_grad = False
    def forward_features(self,left = None,center = None,right = None,left_kp = None,right_kp=None,center_kp=None,
                         center_pf = None,left_pf = None,right_pf = None): 
        b, t, x, c, h, w = left.size()
        left_feature = self.feature_extractor(left.view(b, t * x, c, h, w)).view(b,t,-1)
        right_feature = self.feature_extractor(right.view(b, t * x, c, h, w)).view(b,t,-1)
        center_feature = self.feature_extractor(center.view(b, t * x, c, h, w)).view(b,t,-1)

        left_kp_feature = self.feature_extractor_gcn_left(left_kp)
        right_kp_feature = self.feature_extractor_gcn_right(right_kp)
        
        left_ft = self.left.forward_features(left_feature,left_kp_feature,left_pf).mean(1)
        center_ft = self.center.forward_features(center_feature,center_pf).mean(1)
        right_ft = self.right.forward_features(right_feature,right_kp_feature,right_pf).mean(1)
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)
       
        return output_features
    def forward(self,left = None,center = None,right = None,left_kp = None,right_kp=None,center_kp=None,
                center_pf = None,left_pf = None,right_pf = None):  
        # b, t, x, c, h, w = left.size()

        # left_feature = self.feature_extractor(left.view(b, t * x, c, h, w)).view(b,t,-1)
        # right_feature = self.feature_extractor(right.view(b, t * x, c, h, w)).view(b,t,-1)
        # center_feature = self.feature_extractor(center.view(b, t * x, c, h, w)).view(b,t,-1)

        ###NO HANDCROP###
        b, t, c, h, w = left.size()

        left_feature = self.feature_extractor(left.view(b, t, c, h, w)).view(b,t,-1)
        right_feature = self.feature_extractor(right.view(b, t, c, h, w)).view(b,t,-1)
        center_feature = self.feature_extractor(center.view(b, t, c, h, w)).view(b,t,-1)
        ######

        left_kp_feature = self.feature_extractor_gcn_left(left_kp)
        right_kp_feature = self.feature_extractor_gcn_right(right_kp)
        
        left_ft = self.left.forward_features(left_feature,left_kp_feature,left_pf).mean(1)
        center_ft = self.center.forward_features(center_feature,center_pf).mean(1)
        right_ft = self.right.forward_features(right_feature,right_kp_feature,right_pf).mean(1)
        
        output_features = torch.cat([left_ft,center_ft,right_ft],dim = -1)

        y = self.classifier(output_features)

        return {
            'logits':y
        }
class VTNHCPF_OneView_Sim_Knowledge_Distilation(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VTNHCPF_OneView_Sim_Knowledge_Distillation")
        self.teacher = VTNHCPF_Three_View(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.teacher.add_backbone()
        self.teacher.remove_head_and_backbone()
        self.teacher.load_state_dict(torch.load("checkpoints/VTNHCPF_Three_view/vtn_att_poseflow three view finetune from one view with testings labels/best_checkpoints.pth",map_location='cpu'))
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        new_state_dict = {}
        with pl_legacy_patch():
            for key, value in torch.load('checkpoints/VTN_HCPF.ckpt',map_location='cpu')['state_dict'].items():
                new_state_dict[key.replace('model.','')] = value
        self.student.reset_head(226) # AUTSL
        self.student.load_state_dict(new_state_dict)
        self.student.classifier = nn.Identity()
        self.projection = nn.Linear(embed_size*2,embed_size*6)
        self.norm = MMTensorNorm(-1)
        self.relu = F.relu
    def forward(self,left = None,center = None,right = None,center_pf = None,left_pf = None,right_pf = None):  
        b, t, x, c, h, w = left.size()
        self.teacher.eval()
        teacher_features = None
        y = None
        teacher_features = self.teacher.forward_features(left = left,center = center,right = right,left_pf = left_pf,right_pf=right_pf,center_pf=center_pf)
        
        student_features = self.student(clip = center,poseflow = center_pf)['logits']
        student_features = self.projection(self.norm(student_features))
        if not self.training:
            y = self.teacher.classifier(student_features)
        
        return {
            'logits':y,
            'student_features': student_features,
            'teacher_features': teacher_features,
        }
    
class VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference")
        print("*"*20)
        self.student = VTNHCPF(num_classes,num_heads,num_layers,embed_size,sequence_length,cnn,freeze_layers,dropout,**kwargs)
        self.student.classifier = nn.Identity()
        self.projection = nn.Linear(embed_size*2,embed_size*6)
        self.norm = MMTensorNorm(-1)
        self.relu = F.relu
        self.classifier = LinearClassifier(embed_size*2*3, num_classes, dropout)
        print("*"*20)
    def forward(self,clip = None,poseflow = None):  
        center = clip
        center_pf = poseflow
        b, t, x, c, h, w = center.size()
       
        student_features = self.student(clip = center,poseflow = center_pf)['logits']
        student_features = self.projection(self.norm(student_features))
       
        y = self.classifier(student_features)
        
        return {
            'logits':y,
        }