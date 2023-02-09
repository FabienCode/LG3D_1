# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector, lg3d_SingleStage3DDetector


@DETECTORS.register_module()
class lg3d_VoteNet(lg3d_SingleStage3DDetector):
    r"""`VoteNet <https://arxiv.org/pdf/1904.09664.pdf>`_ for 3D detection."""

    def __init__(self,
                 s_backbone,
                 t_backbone,
                 label_encoder,
                 anno_descriptor,
                 attention,
                 bbox_head=None,
                 train_cfg=None,                                                                                    
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(lg3d_VoteNet, self).__init__(
            s_backbone=s_backbone,
            t_backbone=t_backbone,
            label_encoder=label_encoder,
            anno_descriptor=anno_descriptor,
            attention=attention,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)
        self.feature_adapt = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1 ,1)
        )
        # self.layernorm = nn.GroupNorm(num_groups=1, num_channels=256, affine=False)

    def forward_train(self,
                      points,
                      label_points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        points_cat = torch.stack(points)

        x = self.s_backbone(points_cat)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        # obtain annotations
        anno = self.extract_para_scannet(gt_bboxes_3d, gt_labels_3d).cuda()
        anno_feature = self.anno_descriptor(anno)
        
        # label point clouds
        label_points_cat = torch.stack(label_points)
        label_points_feature = self.label_encoder(label_points_cat)
        
        # teacher feature
        teacher_feature = self.t_backbone(points_cat)
        
        label_attention_anno = self.attention(label_points_feature['fp_features'][-1],
                                                                 anno_feature,
                                                                 anno_feature)
        
        teacher_attention_label = self.attention(teacher_feature['fp_features'][-1],
                                                label_attention_anno,
                                                label_attention_anno)
        
        t_bbox_pred = self.bbox_head(teacher_feature, self.train_cfg.sample_mod)
        t_loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask, pts_instance_mask, img_metas)
        t_loss = self.bbox_head.loss(t_bbox_pred, *t_loss_inputs, gt_bboxes_ignore)
        for k, v in t_loss.items():
            losses['label_' + k] = v
            
        t_encodings = teacher_attention_label.detach()
        s_feature = x['fp_features'][-1]
        losses['Label-Guided-losses'] = 10 * F.mse_loss(t_encodings, s_feature)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)

        x = self.s_backbone(points_cat)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.s_backbone(points_cat, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(
                pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
    
    def extract_para_scannet(self, gt_bboxes_3d, gt_labels_3d):
        mean_sizes = [[0.76966727, 0.8116021, 0.92573744],
                      [1.876858, 1.8425595, 1.1931566],
                      [0.61328, 0.6148609, 0.7182701],
                      [1.3955007, 1.5121545, 0.83443564],
                      [0.97949594, 1.0675149, 0.6329687],
                      [0.531663, 0.5955577, 1.7500148],
                      [0.9624706, 0.72462326, 1.1481868],
                      [0.83221924, 1.0490936, 1.6875663],
                      [0.21132214, 0.4206159, 0.5372846],
                      [1.4440073, 1.8970833, 0.26985747],
                      [1.0294262, 1.4040797, 0.87554324],
                      [1.3766412, 0.65521795, 1.6813129],
                      [0.6650819, 0.71111923, 1.298853],
                      [0.41999173, 0.37906948, 1.7513971],
                      [0.59359556, 0.5912492, 0.73919016],
                      [0.50867593, 0.50656086, 0.30136237],
                      [1.1511526, 1.0546296, 0.49706793],
                      [0.47535285, 0.49249494, 0.5802117]]
        bbox_info = list()
        for i in range(len(gt_bboxes_3d)):
            size_class_target = gt_labels_3d[i]

            size_res_target = gt_bboxes_3d[i].dims - gt_bboxes_3d[i].tensor.new_tensor(
                mean_sizes)[size_class_target]
            # log_size_res_target = torch.zeros((size_res_target.shape[0], 3))
            # for a in range(size_res_target.shape[0]):
            #     for b in range(3):
            #         log_size_res_target[a][b] = torch.log(size_res_target[a][b].abs())

            # log_size_res_target = torch.log(size_res_target)

            # (dir_class_target, dir_res_target) = self.angle2class(gt_bboxes_3d[i].yaw) # sunrgbd
            box_num = gt_labels_3d[i].shape[0]
            dir_class_target = gt_labels_3d[i].new_zeros(box_num)  # scannet
            dir_res_target = gt_bboxes_3d[i].tensor.new_zeros(box_num)  # scannet
            size_class_target = size_class_target.reshape(box_num, 1)
            dir_class_target = dir_class_target.reshape(box_num, 1)
            dir_res_target = dir_res_target.reshape(box_num, 1)
            new_center = gt_bboxes_3d[i].gravity_center.cuda()
            # rx = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # ry = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # rz = torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2)
            # center_aug = torch.cat((rx, ry, rz)).repeat(gt_bboxes_3d[i].gravity_center.shape[0], 1)
            # center_aug = center_aug * gt_bboxes_3d[i].dims
            # new_center = new_center + center_aug
            # new_center = gt_bboxes_3d[i].gravity_center

            tmp_info = torch.cat(
                (new_center.cuda(), size_res_target.cuda(), size_class_target,  # log dim
                 dir_class_target.cuda(), dir_res_target.cuda()),
                1)
            bbox_info.append(tmp_info)
        # # bbox_info = DC(gt_bboxes_3d.tensor)
        # bbox_info_cat = torch.stack(bbox_info)
        label_one_hot = list()
        for i in range(len(gt_labels_3d)):
            # tmp_label_one_hot = to_one_hot(gt_labels_3d[i], 10)
            gt_labels_3d_tmp = gt_labels_3d[i].reshape(gt_labels_3d[i].shape[0], 1).cuda()
            tmp_label_one_hot = torch.zeros(gt_labels_3d_tmp.shape[0], 18).cuda().scatter_(1, gt_labels_3d_tmp, 1)
            tmp_label_one_hot = F.softmax(tmp_label_one_hot, dim=-1)
            # tmp_label_one_hot = self.embedding(gt_labels_3d[i].cuda())

            label_one_hot.append(tmp_label_one_hot)
        # bbox_label = torch.stack(gt_labels_3d)

        gt_para = list()
        for i in range(len(gt_labels_3d)):
            gt_para.append(torch.zeros((256, 27)))
            tmp = torch.cat((bbox_info[i].cuda(), label_one_hot[i].cuda()), 1)
            for j in range(tmp.shape[0]):
                gt_para[i][j] = tmp[j]

        gt_para_cat = torch.stack(gt_para)
        return gt_para_cat
            

        
