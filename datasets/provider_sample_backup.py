# provider_sample的副本

''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017

Modified by Zhixin Wang
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import pickle
import sys
import os
import numpy as np

import torch
import logging
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from configs.config import cfg

from datasets.data_utils import rotate_pc_along_y, project_image_to_rect, compute_box_3d, extract_pc_in_box3d, roty
# from datasets.dataset_info import KITTICategory
from datasets.dataset_info import DATASET_INFO

logger = logging.getLogger(__name__)

# train_dataset = dataset_def(
#         cfg.DATA.NUM_SAMPLES,
#         split=cfg.TRAIN.DATASET,
#         one_hot=True,
#         random_flip=True,
#         random_shift=True,
#         extend_from_det=cfg.DATA.EXTEND_FROM_DET)


class ProviderDataset(Dataset):

    def __init__(self, npoints, split,
                 random_flip=False, random_shift=False,
                 one_hot=True,
                 from_rgb_detection=False,
                 overwritten_data_path='',
                 extend_from_det=False):

        super(ProviderDataset, self).__init__()
        self.npoints = npoints
        self.split = split
        self.random_flip = random_flip
        self.random_shift = random_shift

        self.one_hot = one_hot
        self.from_rgb_detection = from_rgb_detection

        dataset_name = cfg.DATA.DATASET_NAME
        assert dataset_name in DATASET_INFO
        self.category_info = DATASET_INFO[dataset_name]

        root_data = cfg.DATA.DATA_ROOT
        car_only = cfg.DATA.CAR_ONLY
        people_only = cfg.DATA.PEOPLE_ONLY

        if not overwritten_data_path:
            if not from_rgb_detection:
                if car_only:
                    overwritten_data_path = os.path.join(root_data, 'frustum_caronly_%s.pickle' % (split))
                elif people_only:
                    overwritten_data_path = os.path.join(root_data, 'frustum_pedcyc_%s.pickle' % (split))
                else:
                    overwritten_data_path = os.path.join(root_data, 'frustum_carpedcyc_%s.pickle' % (split))
            else:
                if car_only:
                    overwritten_data_path = os.path.join(root_data,
                                                         'frustum_caronly_%s_rgb_detection.pickle' % (split))
                elif people_only:
                    overwritten_data_path = os.path.join(root_data, 'frustum_pedcyc_%s_rgb_detection.pickle' % (split))
                else:
                    overwritten_data_path = os.path.join(
                        root_data, 'frustum_carpedcyc_%s_rgb_detection.pickle' % (split))

        overwritten_data_path = ROOT_DIR + '/' + overwritten_data_path
        if from_rgb_detection:

            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                print(f"id_list1 = {self.id_list}")

                self.box2d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
                self.prob_list = pickle.load(fp)
                self.calib_list = pickle.load(fp)

        else:
            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                print(f"id_list2 = {self.id_list}")
                self.box2d_list = pickle.load(fp)
                self.box3d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.label_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.heading_list = pickle.load(fp)
                self.size_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
                self.gt_box2d_list = pickle.load(fp)
                self.calib_list = pickle.load(fp)

            if extend_from_det:
                extend_det_file = overwritten_data_path.replace('.', '_det.')
                assert os.path.exists(extend_det_file), extend_det_file
                with open(extend_det_file, 'rb') as fp:
                    # extend
                    self.id_list.extend(pickle.load(fp))
                    print(f"id_list3 = {self.id_list}")
                    self.box2d_list.extend(pickle.load(fp))
                    self.box3d_list.extend(pickle.load(fp))
                    self.input_list.extend(pickle.load(fp))
                    self.label_list.extend(pickle.load(fp))
                    self.type_list.extend(pickle.load(fp))
                    self.heading_list.extend(pickle.load(fp))
                    self.size_list.extend(pickle.load(fp))
                    self.frustum_angle_list.extend(pickle.load(fp))
                    self.gt_box2d_list.extend(pickle.load(fp))
                    self.calib_list.extend(pickle.load(fp))
                logger.info('load dataset from {}'.format(extend_det_file))

        logger.info('load dataset from {}'.format(overwritten_data_path))

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):

        rotate_to_center = cfg.DATA.RTC
        with_extra_feat = cfg.DATA.WITH_EXTRA_FEAT

        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)   # 0.09249256688073482

        cls_type = self.type_list[index]                    # 对于Car-only来说全都是'Car'
        # assert cls_type in KITTICategory.CLASSES, cls_type
        # size_class = KITTICategory.CLASSES.index(cls_type)

        assert cls_type in self.category_info.CLASSES, '%s not in category_info' % cls_type
        size_class = self.category_info.CLASSES.index(cls_type)

        # Compute one hot vector
        if self.one_hot:
            one_hot_vec = np.zeros((len(self.category_info.CLASSES)))
            one_hot_vec[size_class] = 1

        # Get point cloud
        if rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]

        if not with_extra_feat:
            point_set = point_set[:, :3]

        # Resample
        if self.npoints > 0:     # npoints = 1024
            # choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)     replace=True表示有放回采样

            # point_set.shape = (111,4) ，shape[0]为point_set中point的数目，shape[1]的4个值分别为xyz，intensity
            # point_set = [[ 5.2755386e-01  1.0811783e+00  4.4682854e+01  3.4999999e-01],
            #              [ 3.9081648e-01  1.0973843e+00  4.5174202e+01  2.0999999e-01],
            #              [ 2.8093919e-01  1.2784263e+00  5.1086655e+01  2.0000000e-01],
            #              [ 1.1993008e-01  1.2816179e+00  5.1122944e+01  1.5000001e-01],
            #              [ 3.8709618e-02  1.2826408e+00  5.1132492e+01  3.1000000e-01],...
            choice = np.random.choice(point_set.shape[0], self.npoints, point_set.shape[0] < self.npoints)  # 从point_set中有放回采样出npoints个点

        else:
            choice = np.random.permutation(len(point_set.shape[0]))         # 全排列中随机取一个

        point_set = point_set[choice, :]

        box = self.box2d_list[index]
        P = self.calib_list[index]['P2'].reshape(3, 4)

        ref1, ref2, ref3, ref4 = self.generate_ref(box, P)
        # ref1, ref2, ref3，ref4是一个视锥经过复制为4个后，再对z值分别生成一个range(0, max_depth)列表（步长不同），
        # 最后分别通过project_image_to_rect转换为直角坐标系
        # ref1 = [[-6.15163738e-02  5.57402739e-03  1.19830504e-01], [-6.07425830e-02  1.73218543e-02  3.70975463e-01], [-5.99687921e-02  2.90696812e-02  6.22120422e-01], [-5.91950012e-02  4.08175081e-02  8.73265381e-01], [-5.84212104e-02  5.25653350e-02  1.12441034e+00], [-5.76474195e-02  6.43131619e-02  1.37555530e+00], [-5.68736287e-02  7.60609888e-02  1.62670026e+00], [-5.60998378e-02  8.78088157e-02  1.87784522e+00], [-5.53260469e-02  9.95566426e-02  2.12899018e+00], [-5.45522561e-02  1.11304469e-01  2.38013514e+00], [-5.37784652e-02  1.23052296e-01  2.63128010e+00], [-5.30046743e-02  1.34800123e-01  2.88242505e+00], [-5.22308835e-02  1.46547950e-01  3.13357001e+00], [-5.14570926e-02  1.58295777e-01  3.38471497e+00], [-5.06833017e-02  1.70043604e-01  3.63585993e+00], [-4.99095109e-02  1.81791431e-01  3.88700489e+00], [-4.91357200e-02  1.93539258e-01  4.13814985e+00], [-4.83619291e-02  2.05287085e-01  4.38929481e+00], [-4.75881383e-02  2.17034912e-01  4.64043977e+00], [-4.68143474e-02  2.28782738e-...
        # ref2 = [[-6.11294784e-02  1.14479408e-02  2.45402984e-01], [-5.95818967e-02  3.49435946e-02  7.47692902e-01], [-5.80343150e-02  5.84392484e-02  1.24998282e+00], [-5.64867332e-02  8.19349022e-02  1.75227274e+00], [-5.49391515e-02  1.05430556e-01  2.25456266e+00], [-5.33915698e-02  1.28926210e-01  2.75685257e+00], [-5.18439880e-02  1.52421864e-01  3.25914249e+00], [-5.02964063e-02  1.75917517e-01  3.76143241e+00], [-4.87488246e-02  1.99413171e-01  4.26372233e+00], [-4.72012428e-02  2.22908825e-01  4.76601225e+00], [-4.56536611e-02  2.46404479e-01  5.26830217e+00], [-4.41060794e-02  2.69900133e-01  5.77059208e+00], [-4.25584976e-02  2.93395786e-01  6.27288200e+00], [-4.10109159e-02  3.16891440e-01  6.77517192e+00], [-3.94633342e-02  3.40387094e-01  7.27746184e+00], [-3.79157524e-02  3.63882748e-01  7.77975176e+00], [-3.63681707e-02  3.87378402e-01  8.28204168e+00], [-3.48205890e-02  4.10874055e-01  8.78433159e+00], [-3.32730072e-02  4.34369709e-01  9.28662151e+00], [-3.17254255e-02  4.57865363e-...
        # ref3 = [[-6.03556875e-02  2.31957677e-02  4.96547943e-01], [-5.72605241e-02  7.01870753e-02  1.50112778e+00], [-5.41653606e-02  1.17178383e-01  2.50570762e+00], [-5.10701972e-02  1.64169691e-01  3.51028745e+00], [-4.79750337e-02  2.11160998e-01  4.51486729e+00], [-4.48798702e-02  2.58152306e-01  5.51944713e+00], [-4.17847068e-02  3.05143613e-01  6.52402696e+00], [-3.86895433e-02  3.52134921e-01  7.52860680e+00], [-3.55943798e-02  3.99126228e-01  8.53318663e+00], [-3.24992164e-02  4.46117536e-01  9.53776647e+00], [-2.94040529e-02  4.93108844e-01  1.05423463e+01], [-2.63088894e-02  5.40100151e-01  1.15469261e+01], [-2.32137260e-02  5.87091459e-01  1.25515060e+01], [-2.01185625e-02  6.34082766e-01  1.35560858e+01], [-1.70233991e-02  6.81074074e-01  1.45606657e+01], [-1.39282356e-02  7.28065382e-01  1.55652455e+01], [-1.08330721e-02  7.75056689e-01  1.65698253e+01], [-7.73790866e-03  8.22047997e-01  1.75744052e+01], [-4.64274520e-03  8.69039304e-01  1.85789850e+01], [-1.54758173e-03  9.16030612e-...
        # ref4 = [[-5.88081058e-02  4.66914215e-02  9.98837861e-01], [-5.26177789e-02  1.40674037e-01  3.00799753e+00], [-4.64274520e-02  2.34656652e-01  5.01715721e+00], [-4.02371250e-02  3.28639267e-01  7.02631688e+00], [-3.40467981e-02  4.22621882e-01  9.03547655e+00], [-2.78564712e-02  5.16604497e-01  1.10446362e+01], [-2.16661442e-02  6.10587113e-01  1.30537959e+01], [-1.54758173e-02  7.04569728e-01  1.50629556e+01], [-9.28549039e-03  7.98552343e-01  1.70721152e+01], [-3.09516346e-03  8.92534958e-01  1.90812749e+01], [ 3.09516346e-03  9.86517573e-01  2.10904346e+01], [ 9.28549039e-03  1.08050019e+00  2.30995943e+01], [ 1.54758173e-02  1.17448280e+00  2.51087539e+01], [ 2.16661442e-02  1.26846542e+00  2.71179136e+01], [ 2.78564712e-02  1.36244803e+00  2.91270733e+01], [ 3.40467981e-02  1.45643065e+00  3.11362330e+01], [ 4.02371250e-02  1.55041326e+00  3.31453926e+01], [ 4.64274520e-02  1.64439588e+00  3.51545523e+01], [ 5.26177789e-02  1.73837849e+00  3.71637120e+01], [ 5.88081058e-02  1.83236111e+...
        if rotate_to_center:
            ref1 = self.get_center_view(ref1, index)
            ref2 = self.get_center_view(ref2, index)
            ref3 = self.get_center_view(ref3, index)
            ref4 = self.get_center_view(ref4, index)

        if self.from_rgb_detection:

            data_inputs = {
                'point_cloud': torch.FloatTensor(point_set).transpose(1, 0),
                'rot_angle': torch.FloatTensor([rot_angle]),
                'rgb_prob': torch.FloatTensor([self.prob_list[index]]),
                'center_ref1': torch.FloatTensor(ref1).transpose(1, 0),
                'center_ref2': torch.FloatTensor(ref2).transpose(1, 0),
                'center_ref3': torch.FloatTensor(ref3).transpose(1, 0),
                'center_ref4': torch.FloatTensor(ref4).transpose(1, 0),

            }

            if not rotate_to_center:
                data_inputs.update({'rot_angle': torch.zeros(1)})

            if self.one_hot:
                data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})

            return data_inputs

        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index].astype(np.int64)
        seg = seg[choice]

        # Get center point of 3D box
        if rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)     # [-8.95501188e-03  1.56500000e+00  3.45267537e+01]
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        box3d_size = self.size_list[index]

        # Size
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            # 注意:如果我们使用random_flip，那么rot_angle将不正确
            # 所以不要使用它以防随机翻转。（那这里为什么还是使用了）
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle

                ref1[:, 0] *= -1
                ref2[:, 0] *= -1
                ref3[:, 0] *= -1
                ref4[:, 0] *= -1

        if self.random_shift:
            max_depth = cfg.DATA.MAX_DEPTH
            l, w, h = self.size_list[index]
            dist = np.sqrt(np.sum(l ** 2 + w ** 2))
            shift = np.clip(np.random.randn() * dist * 0.2, -0.5 * dist, 0.5 * dist)
            shift = np.clip(shift + box3d_center[2], 0, max_depth) - box3d_center[2]
            point_set[:, 2] += shift
            box3d_center[2] += shift

        labels_ref2 = self.generate_labels(box3d_center, box3d_size, heading_angle, ref2, P)        # ？？？？？？？？？

        data_inputs = {
            'point_cloud': torch.FloatTensor(point_set).transpose(1, 0),
            'rot_angle': torch.FloatTensor([rot_angle]),
            'center_ref1': torch.FloatTensor(ref1).transpose(1, 0),
            'center_ref2': torch.FloatTensor(ref2).transpose(1, 0),
            'center_ref3': torch.FloatTensor(ref3).transpose(1, 0),
            'center_ref4': torch.FloatTensor(ref4).transpose(1, 0),

            'cls_label': torch.LongTensor(labels_ref2),
            'box3d_center': torch.FloatTensor(box3d_center),
            'box3d_heading': torch.FloatTensor([heading_angle]),
            'box3d_size': torch.FloatTensor(box3d_size),
            'size_class': torch.LongTensor([size_class]),
            'seg_label': torch.LongTensor(seg.astype(np.int64))
        }

        if not rotate_to_center:
            data_inputs.update({'rot_angle': torch.zeros(1)})

        if self.one_hot:
            data_inputs.update({'one_hot': torch.FloatTensor(one_hot_vec)})

        return data_inputs

    def generate_labels(self, center, dimension, angle, ref_xyz, P):
        """
        generate_labels(box3d_center, box3d_size, heading_angle, ref2, P)

        """

        box_corner1 = compute_box_3d(center, dimension * 0.5, angle)
        box_corner2 = compute_box_3d(center, dimension, angle)

        labels = np.zeros(len(ref_xyz))
        inside1 = extract_pc_in_box3d(ref_xyz, box_corner1)
        inside2 = extract_pc_in_box3d(ref_xyz, box_corner2)

        labels[inside2] = -1
        labels[inside1] = 1
        # dis = np.sqrt(((ref_xyz - center)**2).sum(1))
        # print(dis.min())

        # 如果全是False
        if inside1.sum() == 0:
            dis = np.sqrt(((ref_xyz - center) ** 2).sum(1))
            argmin = np.argmin(dis)
            labels[argmin] = 1

        return labels

    def generate_ref(self, box, P):

        s1, s2, s3, s4 = cfg.DATA.STRIDE            # 0.25, 0.5, 1.0, 2.0
        max_depth = cfg.DATA.MAX_DEPTH
        # MAX_DEPTH: 70

        z1 = np.arange(0, max_depth, s1) + s1 / 2.  # [ 0.125  0.375  0.625  0.875  1.125  1.375  1.625  1.875  2.125  2.375,  2.625  2.875  3.125  3.375  3.625  3.875  4.125  4.375  4.625  4.875,  5.125  5.375  5.625  5.875  6.125  6.375  6.625  6.875  7.125  7.375,  7.625  7.875  8.125  8.375  8.625  8.875  9.125  9.375  9.625  9.875, 10.125 10.375 10.625 10.875 11.125 11.375 11.625 11.875 12.125 12.375, 12.625 12.875 13.125 13.375 13.625 13.875 14.125 14.375 14.625 14.875, 15.125 15.375 15.625 15.875 16.125 16.375 16.625 16.875 17.125 17.375, 17.625 17.875 18.125 18.375 18.625 18.875 19.125 19.375 19.625 19.875, 20.125 20.375 20.625 20.875 21.125 21.375 21.625 21.875 22.125 22.375, 22.625 22.875 23.125 23.375 23.625 23.875 24.125 24.375 24.625 24.875]
        z2 = np.arange(0, max_depth, s2) + s2 / 2.  # [ 0.25  0.75  1.25  1.75  2.25  2.75  3.25  3.75  4.25  4.75  5.25  5.75,  6.25  6.75  7.25  7.75  8.25  8.75  9.25  9.75 10.25 10.75 11.25 11.75, 12.25 12.75 13.25 13.75 14.25 14.75 15.25 15.75 16.25 16.75 17.25 17.75, 18.25 18.75 19.25 19.75 20.25 20.75 21.25 21.75 22.25 22.75 23.25 23.75, 24.25 24.75 25.25 25.75 26.25 26.75 27.25 27.75 28.25 28.75 29.25 29.75, 30.25 30.75 31.25 31.75 32.25 32.75 33.25 33.75 34.25 34.75 35.25 35.75, 36.25 36.75 37.25 37.75 38.25 38.75 39.25 39.75 40.25 40.75 41.25 41.75, 42.25 42.75 43.25 43.75 44.25 44.75 45.25 45.75 46.25 46.75 47.25 47.75, 48.25 48.75 49.25 49.75]
        z3 = np.arange(0, max_depth, s3) + s3 / 2.  # [ 0.5  1.5  2.5  3.5  4.5  5.5  6.5  7.5  8.5  9.5 10.5 11.5 12.5 13.5, 14.5 15.5 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5 24.5 25.5 26.5 27.5, 28.5 29.5 30.5 31.5 32.5 33.5 34.5 35.5 36.5 37.5 38.5 39.5 40.5 41.5, 42.5 43.5 44.5 45.5 46.5 47.5 48.5 49.5 50.5 51.5 52.5 53.5 54.5 55.5, 56.5 57.5 58.5 59.5 60.5 61.5 62.5 63.5 64.5 65.5 66.5 67.5 68.5 69.5]
        z4 = np.arange(0, max_depth, s4) + s4 / 2.  # [ 1.  3.  5.  7.  9. 11. 13. 15. 17. 19. 21. 23. 25. 27. 29. 31. 33. 35., 37. 39. 41. 43. 45. 47. 49. 51. 53. 55. 57. 59. 61. 63. 65. 67. 69.]

        cx, cy = (box[0] + box[2]) / 2., (box[1] + box[3]) / 2.,    # cx=678.73, cy=206.76

        xyz1 = np.zeros((len(z1), 3))
        xyz1[:, 0] = cx
        xyz1[:, 1] = cy
        xyz1[:, 2] = z1
        # xyz1 = [[6.7873e+02 2.0676e+02 1.2500e-01],
        #         [6.7873e+02 2.0676e+02 3.7500e-01],
        #         [6.7873e+02 2.0676e+02 6.2500e-01], ...

        xyz1_rect = project_image_to_rect(xyz1, P)
        # xyz1_rect = [[-5.01857942e-02  5.57402739e-03  1.25000000e-01],
        #               [-2.62193750e-02  1.73218543e-02  3.75000000e-01],
        #               [-2.25295574e-03  2.90696812e-02  6.25000000e-01], ...

        xyz2 = np.zeros((len(z2), 3))
        xyz2[:, 0] = cx
        xyz2[:, 1] = cy
        xyz2[:, 2] = z2
        xyz2_rect = project_image_to_rect(xyz2, P)

        xyz3 = np.zeros((len(z3), 3))
        xyz3[:, 0] = cx
        xyz3[:, 1] = cy
        xyz3[:, 2] = z3
        xyz3_rect = project_image_to_rect(xyz3, P)

        xyz4 = np.zeros((len(z4), 3))
        xyz4[:, 0] = cx
        xyz4[:, 1] = cy
        xyz4[:, 2] = z4
        xyz4_rect = project_image_to_rect(xyz4, P)

        return xyz1_rect, xyz2_rect, xyz3_rect, xyz4_rect

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]         # frustum_angle_list = [-1.4783037599141617, -1.963840149466865, -1.6572657455640256, -2.1443640967902238, -1.8928152999867967, -2.088206294584603, -1.7519085981737494, -0.9863351749082573, -1.5051384385058393, -1.3593086628158941, -1.1665693241821924, -2.0738265429766156, -1.2600995461515554, -1.3884296426632834, -1.8073476395364636, -1.7222694102151535, -1.3715553716821118, -1.635342336817339, -1.3979759833964476, -1.6095927899369071, -1.666029860425068, -1.773157030188972, -1.0064456089214047, -1.7982378267014032, -1.6003318105269728, -1.3618471454221808, -1.3810277878174049, -1.8407457794644575, -1.1820522686494388, -1.3412502306976473, -1.4670140513561314, -2.17165649144584, -1.5863689789512783, -1.7550446395672226, -2.0045683152299096, -1.5947074243402561, -1.715920392491195, -1.9896671147168352, -1.4086608926984927, -1.6509280973039806, -1.4791970027688615, -1.512766409489794, -1.7304056637711696, -1.6001725669432851, -1.5864475750601077, -2.2004587475139368, -1.9544760176369418, -1.923284714649051, -1.7716547978306965, -1.6705838583167325, -2.196366508496539, -2.0409705591129716, -1.3904073700280222, -1.8986768807061316, -0.9994812340089461, -1.5020341657842655, -1.3724077651787026, -1.504172526149584, -1.9087225250539337, -2.2260635635795976, -1.9644254074195246, -1.9768838031592444, -1.3536187892130414, -1.748406454271086, -1.6936908601536629, -1.4766138757148333, -1.662786398102108, -1.6167589849952302, -1.6887735464295792, -1.040495502723968, -0.8993278854987422, -2.2572005190162074, -2.196036858750769, -2.23499330241265, -1.7712678438556717, -1.6723149104366273, -1.5581741834022582, -1.1606097844965815, -1.855150643022007, -1.4617452867423384, -1.5693371719194262, -1.5932822329413914, -1.9328321364616081, -1.4794100285344374, -1.841936196648856, -1.4641425581560432, -1.726532881430088, -1.343314479380989, -1.5177472771890987, -2.2185140120013287, -2.12674633229918, -1.3577723343064954, -1.9356471975102751, -1.8314726135533637, -1.6302271870780067, -1.4436167501112236, -1.4578240157648419, -1.9953919092278394, -1.8733572471711761, -0.8755050451368148...

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] +
                        self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        # 取3Dbox的中心点坐标 对角点相加除以二
        box3d_center = (self.box3d_list[index][0, :] +
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0),
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners.
            3D边框框角的截锥旋转
            似乎未被使用
        '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view,
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        点云的截锥旋转。
        NxC points with first 3 channels as XYZ
        NxC 的矩阵，N个点，C=3分别是XYZ坐标
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index]) # input_list是NxC的矩阵，C若=4则分别是XYZ和intensity
        # point_set取到第index个对象的点集
        return rotate_pc_along_y(point_set,
                                 self.get_center_view_rot_angle(index))

    def get_center_view(self, point_set, index):
        ''' Frustum rotation of point clouds.
        点云的截锥旋转（与上面函数的不同之处在于不需要再从input_list中读取，而是从points_set=ref1234中获取，ref1234也是[[x,y,z],...]的直角坐标形式）
        NxC points with first 3 channels as XYZ
        NxC 的矩阵，N个点，C=3分别是XYZ坐标
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(point_set)
        return rotate_pc_along_y(point_set,
                                 self.get_center_view_rot_angle(index))


def from_prediction_to_label_format(center, angle, size, rot_angle, ref_center=None):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = size
    ry = angle + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()

    if ref_center is not None:
        tx = tx + ref_center[0]
        ty = ty + ref_center[1]
        tz = tz + ref_center[2]

    ty += h / 2.0
    return h, w, l, tx, ty, tz, ry

def compute_alpha(x, z, ry):
    """
    在test_net.py中才被使用
    计算alpha角，相关图可参考https://blog.csdn.net/cuichuanchen3307/article/details/80596689"""
    beta = np.arctan2(z, x)     # beta = arctan(z/x)  相当于博客图中的theta，下面假设beta\theta值为负的
    alpha = -np.sign(beta) * np.pi / 2 + beta + ry  # alpha = ry + pi/2 - |beta|        # 还是有些出入

    return alpha

def collate_fn(batch):
    return default_collate(batch)


if __name__ == '__main__':

    cfg.DATA.DATA_ROOT = 'kitti/data/pickle_data'
    cfg.DATA.RTC = True
    dataset = ProviderDataset(1024, split='val', random_flip=True, one_hot=True, random_shift=True)

    for i in range(len(dataset)):
        data = dataset[i]

        for name, value in data.items():
            print(name, value.shape)

        input()

    '''
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    tic = time.time()
    for i, data_dict in enumerate(train_loader):

        # for key, value in data_dict.items():
        #     print(key, value.shape)

        print(time.time() - tic)
        tic = time.time()

        # input()
    '''
