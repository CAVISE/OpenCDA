import os
import re
import time
import shutil
import logging
from tqdm import tqdm

import torch  # type: ignore
import open3d as o3d
from torch.utils.data import DataLoader  # type: ignore

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import simple_vis
from opencood.utils import eval_utils
from opencood.visualization import vis_utils

logger = logging.getLogger("cavise.coperception_model_manager")


class CoperceptionModelManager():

    def __init__(self, opt):
        self.opt = opt
        self.hypes = yaml_utils.load_yaml(None, self.opt)
        self.model = train_utils.create_model(self.hypes)

        if torch.cuda.is_available():
            self.model.cuda()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.saved_path = self.opt.model_dir
        _, self.model = train_utils.load_saved_model(self.saved_path, self.model)

    def make_pred(self):
        assert self.opt.fusion_method in ['late', 'early', 'intermediate']
        assert not (self.opt.show_vis and self.opt.show_sequence), 'you can only visualize ' \
                                                                   'the results in single '  \
                                                                   'image mode or video mode'
        logger.info('Dataset Building')
        opencood_dataset = build_dataset(self.hypes, visualize=True, train=False)
        logger.info(f"{len(opencood_dataset)} samples found.")
        data_loader = DataLoader(opencood_dataset,
                                batch_size=1,
                                num_workers=16,
                                collate_fn=opencood_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)

        self.model.eval()

        # Create the dictionary for evaluation.
        # also store the confidence score for each prediction
        result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                    0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                    0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

        if self.opt.show_sequence:
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            vis.get_render_option().point_size = 1.0
            vis.get_render_option().show_coordinate_frame = True

            # used to visualize lidar points
            vis_pcd = o3d.geometry.PointCloud()
            # used to visualize object bounding box, maximum 50
            vis_aabbs_gt = []
            vis_aabbs_pred = []
            for _ in range(50):
                vis_aabbs_gt.append(o3d.geometry.LineSet())
                vis_aabbs_pred.append(o3d.geometry.LineSet())

        for i, batch_data in tqdm(enumerate(data_loader)):
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, self.device)
                if self.opt.fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_late_fusion(batch_data,
                                                            self.model,
                                                            opencood_dataset)
                elif self.opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_early_fusion(batch_data,
                                                            self.model,
                                                            opencood_dataset)
                elif self.opt.fusion_method == 'intermediate':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        inference_utils.inference_intermediate_fusion(batch_data,
                                                                    self.model,
                                                                    opencood_dataset)
                else:
                    raise NotImplementedError('Only early, late and intermediate'
                                            'fusion is supported.')

                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.7)

                if self.opt.save_npy:
                    npy_save_path = os.path.join(self.opt.model_dir, 'npy')
                    if not os.path.exists(npy_save_path):
                        os.makedirs(npy_save_path)
                    inference_utils.save_prediction_gt(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'][0],
                                                    i,
                                                    npy_save_path)

                if self.opt.save_vis_n and self.opt.save_vis_n > i:

                    vis_save_path = "opencda/coperception_models/real_time_vis/vis_3d"
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '3d_%05d.png' % i)
                    simple_vis.visualize(pred_box_tensor,
                                        gt_box_tensor,
                                        batch_data['ego']['origin_lidar'][0],
                                        self.hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='3d',
                                        left_hand=True,
                                        vis_pred_box=True)

                    vis_save_path = "opencda/coperception_models/real_time_vis/vis_bev"
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, 'bev_%05d.png' % i)
                    simple_vis.visualize(pred_box_tensor,
                                        gt_box_tensor,
                                        batch_data['ego']['origin_lidar'][0],
                                        self.hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='bev',
                                        left_hand=True,
                                        vis_pred_box=True)

                if self.opt.show_vis or self.opt.save_vis:
                    vis_save_path = ''
                    if self.opt.save_vis:
                        vis_save_path = "opencda/coperception_models/real_time_vis/vis"
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                    opencood_dataset.visualize_result(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'],
                                                    self.opt.show_vis,
                                                    vis_save_path,
                                                    dataset=opencood_dataset)

                if self.opt.show_sequence:
                    pcd, pred_o3d_box, gt_o3d_box = \
                        vis_utils.visualize_inference_sample_dataloader(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data['ego']['origin_lidar'],
                            vis_pcd,
                            mode='constant')
                    if i == 0:
                        vis.add_geometry(pcd)
                        vis_utils.linset_assign_list(vis,
                                                    vis_aabbs_pred,
                                                    pred_o3d_box,
                                                    update_mode='add')

                        vis_utils.linset_assign_list(vis,
                                                    vis_aabbs_gt,
                                                    gt_o3d_box,
                                                    update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_pred,
                                                pred_o3d_box)
                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_gt,
                                                gt_o3d_box)
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.001)
        logger.info(result_stat)
        eval_utils.eval_final_results(result_stat,
                                    self.opt.model_dir,
                                    self.opt.global_sort_detections)
        if self.opt.show_sequence:
            vis.destroy_window()


class DirectoryProcessor:
    def __init__(self, source_directory="data_dumping", now_directory="data_dumping/sample/now"):
        self.source_directory = source_directory
        self.now_directory = now_directory

    def detect_cameras(self, data_directory):
        inner_subdirectories = sorted(
            [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
        )
        if not inner_subdirectories:
            return []

        sample_folder = os.path.join(data_directory, inner_subdirectories[0])
        camera_files = [f for f in os.listdir(sample_folder) if re.match(r'\d+_camera\d+\.png', f)]

        camera_ids = sorted(set(re.findall(r'_camera(\d+)\.png', f)[0] for f in camera_files if re.findall(r'_camera(\d+)\.png', f)))

        return [f"_camera{cam_id}.png" for cam_id in camera_ids]

    def process_directory(self, tick_number):
        number = f"{tick_number:06d}"
        postfixes = [".pcd", ".yaml"]

        subdirectories = sorted(
            [d for d in os.listdir(self.source_directory) if os.path.isdir(os.path.join(self.source_directory, d))]
        )

        if len(subdirectories) < 2:
            raise ValueError("Not enough subdirectories in source directory to process.")

        data_directory = os.path.join(self.source_directory, subdirectories[-2])

        camera_postfixes = self.detect_cameras(data_directory)
        postfixes.extend(camera_postfixes)

        inner_subdirectories = sorted(
            [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
        )

        shutil.copy(os.path.join(data_directory, "data_protocol.yaml"), self.now_directory)

        for folder in inner_subdirectories:
            destination_folder = os.path.join(self.now_directory, folder)
            os.makedirs(destination_folder, exist_ok=True)
            for postfix in postfixes:
                source_file_path = os.path.join(data_directory, folder, f"{number}{postfix}")
                destination_file_path = os.path.join(destination_folder, f"{number}{postfix}")
                if os.path.exists(source_file_path):
                    shutil.copy(source_file_path, destination_file_path)

    def clear_directory_now(self):
        for item in os.listdir(self.now_directory):
            item_path = os.path.join(self.now_directory, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # Удаляем файлы и символические ссылки
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
