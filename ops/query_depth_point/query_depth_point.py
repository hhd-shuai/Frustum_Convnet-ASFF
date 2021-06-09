import math
from torch import nn
from torch.autograd import Function
import torch

from . import query_depth_point_cuda


# 指定半径进行球查询
class _query_depth_point(Function):
    # _query_depth_point.apply(self.dis_z, self.nsample, xyz1, xyz2)
    @staticmethod
    def forward(ctx, dis_z, nsample, xyz1, xyz2):
        '''
        Input:
            dis_z: float32, depth distance search distance
            nsample: int32, number of points selected in each ball region
            xyz1: (batch_size, 3, ndataset) float32 array, input points
            xyz2: (batch_size, 3, npoint) float32 array, query points
        Output:
            idx: (batch_size, npoint, nsample) int32 array, indices to input points
            pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
        '''
        assert xyz1.is_cuda and xyz1.size(1) == 3
        assert xyz2.is_cuda and xyz2.size(1) == 3
        assert xyz1.size(0) == xyz2.size(0)
        assert xyz1.is_contiguous()
        assert xyz2.is_contiguous()

        xyz1 = xyz1.permute(0, 2, 1).contiguous() #torch.Size([32, 1024, 3])
        xyz2 = xyz2.permute(0, 2, 1).contiguous() #torch.Size([32, 280, 3])

        b = xyz1.size(0) # b:32
        n = xyz1.size(1) # n:1024
        m = xyz2.size(1) # m:280

        # .new() 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。
        idx = xyz1.new(b, m, nsample).long().zero_() #torch.Size([32, 280, 32])
        pts_cnt = xyz1.new(b, m).int().zero_() #torch.Size([32, 280])

        query_depth_point_cuda.forward(b, n, m, dis_z, nsample, xyz1, xyz2, idx, pts_cnt)
        return idx, pts_cnt

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 6


class QueryDepthPoint(nn.Module):
    def __init__(self, dis_z, nsample):
        super(QueryDepthPoint, self).__init__()
        self.dis_z = dis_z
        self.nsample = nsample

    def forward(self, xyz1, xyz2):
        # indices, num = self.query_depth_point(pc, new_pc)  # b*npoint*nsample
        # dist=0.25, nsample=32, pc: torch.Size([32, 3, 1024]), new_pc: torch.Size([32, 3, 280])
        return _query_depth_point.apply(self.dis_z, self.nsample, xyz1, xyz2)
