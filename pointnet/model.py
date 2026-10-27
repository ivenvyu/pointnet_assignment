import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.
        # pass
        x = pointcloud
        
        if self.input_transform:
            trans1 = self.stn3(x.transpose(1, 2))
            x = torch.bmm(x, trans1)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.feature_transform:
            trans2 = self.stn64(x)
            x = torch.bmm(x.transpose(1, 2), trans2)
            x = x.transpose(1, 2)
        output = x
        feat_trans = trans2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.max(x, 2)[0]

        return (x, output, feat_trans)

        


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512))
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256))
        self.fc3 = nn.Linear(256, self.num_classes)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        # pass
        x = pointcloud
        pointnet_feat = self.pointnet_feat(x)
        x, feat_trans = pointnet_feat[0], pointnet_feat[2]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, feat_trans



class PointNetPartSeg(nn.Module):
    def __init__(self, m=50, input_transform = True, feature_transform = True):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        # pass
        self.m = m
        self.PointNetFeat = PointNetFeat(input_transform, feature_transform)
        self.conv1 = nn.Sequential(nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512))
        self.conv2 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128))
        self.conv4 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128))
        self.conv5 = nn.Sequential(nn.Conv1d(128, self.m, 1))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        # pass
        x = pointcloud
        global_feature, local_feature, feat_trans = self.PointNetFeat(x)
        N = local_feature.shape[2]
        global_feature = global_feature.unsqueeze(2).expand(-1, -1, N)
        seg = torch.concat((local_feature, global_feature), dim = 1)
        seg = F.relu(self.conv1(seg))
        seg = F.relu(self.conv2(seg))
        seg = F.relu(self.conv3(seg))
        seg = F.relu(self.conv4(seg))
        seg = self.conv5(seg)

        return seg, feat_trans
        




def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
