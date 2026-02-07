import os
import re
import shutil
import logging
from tqdm import tqdm

import torch  # type: ignore
import open3d as o3d
from torch.utils.data import DataLoader  # type: ignore

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import simple_vis, vis_utils
from opencood.utils import eval_utils

logger = logging.getLogger("cavise.coperception_model_manager")


class CoperceptionModelManager:
    def __init__(self, opt, current_time, message_handler=None):
        self.opt = opt
        self.hypes = yaml_utils.load_yaml(None, self.opt)
        self.model = train_utils.create_model(self.hypes)
        self.current_time = current_time
        self.vis = None

        if torch.cuda.is_available():
            self.model.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.saved_path = self.opt.model_dir
        _, self.model = train_utils.load_saved_model(self.saved_path, self.model)

        self.opencood_dataset = None
        self.data_loader = None
        self.message_handler = message_handler

        logger.info("Initial Dataset Building")
        self.opencood_dataset = build_dataset(self.hypes, visualize=True, train=False, message_handler=self.message_handler)

        self.data_loader = DataLoader(
            self.opencood_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=self.opencood_dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

        self.final_result_stat = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }

        # AdvCP integration
        self.advcp_enabled = False
        self.advcp_manager = None
        self.attack_handler = None
        self.defense_handler = None
        self.metrics_logger = None
        self.visualization = None

        if hasattr(self.opt, "with_advcp") and self.opt.with_advcp:
            from .advcp.advcp_manager import AdvCPManager
            from .advcp.mvp.attack.attacker import Attacker
            from .advcp.mvp.defense.defender import Defender
            from .advcp.mvp.evaluate.accuracy import Accuracy
            from .advcp.mvp.evaluate.detection import Detection
            from .advcp.mvp.visualize.general import Visualizer

            self.advcp_enabled = True
            self.advcp_manager = AdvCPManager(self.opt, self.current_time)
            self.attacker = Attacker(self.opt)
            self.defender = Defender(self.opt)
            self.accuracy = Accuracy()
            self.detection = Detection()
            self.visualizer = Visualizer()

            logger.info("AdvCP module initialized successfully")

    def make_dataset(self):
        logger.info("Dataset Building")
        self.opencood_dataset = build_dataset(self.hypes, visualize=True, train=False, message_handler=self.message_handler)
        logger.info(f"{len(self.opencood_dataset)} samples found.")
        self.data_loader = DataLoader(
            self.opencood_dataset,
            batch_size=1,
            num_workers=16,
            collate_fn=self.opencood_dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

    def update_dataset(self):
        logger.debug("Refreshing dataset indices")
        self.opencood_dataset.update_database()

        if len(self.opencood_dataset) == 0:
            logger.warning("No samples found in dataset after update.")

    def make_prediction(self, tick_number):
        assert self.opt.fusion_method in ["late", "early", "intermediate"]
        assert not (self.opt.show_vis and self.opt.show_sequence), "you can only visualize the results in single image mode or video mode"
        self.model.eval()

        # Create the dictionary for evaluation.
        # also store the confidence score for each prediction
        result_stat = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }

        if self.advcp_enabled:
            # Execute attacks before prediction using MVP Attacker
            if self.advcp_enabled:
                self.advcp_manager._apply_attack_wrapper(self.data_loader, tick_number)
        # Apply defense mechanisms using MVP Defender
        if self.advcp_enabled and self.opt.apply_cad_defense:
            self.advcp_manager._apply_defense_wrapper(self.data_loader, tick_number)

        if self.opt.show_sequence:
            if self.vis is None:
                self.vis = o3d.visualization.Visualizer()  # noqa: DC05
                self.vis.create_window()  # noqa: DC05
                self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]  # noqa: DC05
                self.vis.get_render_option().point_size = 1.0  # noqa: DC05
                self.vis.get_render_option().show_coordinate_frame = True  # noqa: DC05
            # used to visualize lidar points
            vis_pcd = o3d.geometry.PointCloud()
            # used to visualize object bounding box, maximum 50
            vis_aabbs_gt = []
            vis_aabbs_pred = []
            for _ in range(50):
                vis_aabbs_gt.append(o3d.geometry.LineSet())
                vis_aabbs_pred.append(o3d.geometry.LineSet())

        for i, batch_data in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, self.device)
                if self.opt.fusion_method == "late":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_late_fusion(batch_data, self.model, self.opencood_dataset)
                elif self.opt.fusion_method == "early":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(batch_data, self.model, self.opencood_dataset)
                elif self.opt.fusion_method == "intermediate":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(
                        batch_data, self.model, self.opencood_dataset
                    )
                else:
                    raise NotImplementedError("Only early, late and intermediate fusion is supported.")

                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

                if self.opt.save_npy:
                    npy_dir = f"simulation_output/coperception/npy/{self.opt.test_scenario}_{self.current_time}"
                    npy_save_path = os.path.join(npy_dir, "npy")
                    os.makedirs(npy_save_path, exist_ok=True)
                    inference_utils.save_prediction_gt(pred_box_tensor, gt_box_tensor, batch_data["ego"]["origin_lidar"][0], i, npy_save_path)

                if self.opt.save_vis:
                    for mode in ["3d", "bev"]:
                        if self.hypes["postprocess"]["core_method"] == "BevPostprocessor" and mode == "3d":
                            continue
                        vis_dir = f"simulation_output/coperception/vis_{mode}/{self.opt.test_scenario}_{self.current_time}"
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_save_path = os.path.join(vis_dir, f"{mode}_{tick_number:05d}.png")
                        simple_vis.visualize(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data["ego"]["origin_lidar"][0],
                            self.hypes["postprocess"]["gt_range"],
                            vis_save_path,
                            method=mode,
                            left_hand=True,
                            vis_pred_box=True,
                        )

                if self.opt.show_vis:
                    vis_save_path = ""
                    self.opencood_dataset.visualize_result(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data["ego"]["origin_lidar"],
                        self.opt.show_vis,
                        vis_save_path,
                        dataset=self.opencood_dataset,
                    )

                # AdvCP: Log metrics and visualize attacks/defense using MVP components
                if self.advcp_enabled:
                    # Convert tensors to numpy arrays for MVP functions
                    pred_box_np = pred_box_tensor.cpu().numpy() if isinstance(pred_box_tensor, torch.Tensor) else pred_box_tensor
                    pred_score_np = pred_score.cpu().numpy() if isinstance(pred_score, torch.Tensor) else pred_score
                    gt_box_np = gt_box_tensor.cpu().numpy() if isinstance(gt_box_tensor, torch.Tensor) else gt_box_tensor
                    
                    # Create dummy dataset for MVP accuracy function
                    class DummyDataset:
                        def case_generator(self, index=True, tag="multi_frame"):
                            # Convert OpenCOOD batch_data to MVP case format
                            case = {}
                            for vehicle_id, vehicle_data in batch_data.items():
                                case[vehicle_id] = {
                                    "lidar": vehicle_data.get("lidar", np.zeros((100, 4))),
                                    "lidar_pose": vehicle_data.get("lidar_pose", np.eye(4)),
                                    "gt_bboxes": gt_box_np,
                                    "result_bboxes": pred_box_np
                                }
                            yield 0, case
                    
                    # Calculate accuracy metrics using MVP Accuracy
                    accuracy_report = self.advcp_manager.accuracy.get_accuracy(DummyDataset(), self.model)
                    logger.info(f"Tick {tick_number} - Accuracy metrics: {accuracy_report}")
                    
                    # Calculate detection metrics using MVP Detection
                    detection_report = self.advcp_manager.detection.evaluate_single_vehicle(gt_box_np, pred_box_np)
                    logger.info(f"Tick {tick_number} - Detection metrics: {detection_report}")
                    
                    # Visualize results using MVP Visualizer
                    self.advcp_manager._visualize_results_wrapper(pred_box_tensor, gt_box_tensor, batch_data, tick_number)

                if self.opt.show_sequence and pred_box_tensor is not None and self.hypes["postprocess"]["core_method"] != "BevPostprocessor":
                    self.vis.clear_geometries()
                    pcd, pred_o3d_box, gt_o3d_box = vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor, gt_box_tensor, batch_data["ego"]["origin_lidar"], vis_pcd, mode="constant"
                    )
                    if i == 0:
                        self.vis.add_geometry(pcd)
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_pred, pred_o3d_box, update_mode="add")
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_gt, gt_o3d_box, update_mode="add")
                    else:
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_pred, pred_o3d_box)
                        vis_utils.linset_assign_list(self.vis, vis_aabbs_gt, gt_o3d_box)
                    self.vis.update_geometry(pcd)
                    self.vis.poll_events()
                    self.vis.update_renderer()

                # AdvCP: Log metrics and visualize attacks/defense
                if self.advcp_enabled:
                    # Use the same visualization wrapper as before
                    self.advcp_manager._visualize_results_wrapper(pred_box_tensor, gt_box_tensor, batch_data, tick_number)

        for iou in [0.3, 0.5, 0.7]:
            self.final_result_stat[iou]["gt"] += result_stat[iou]["gt"]
            self.final_result_stat[iou]["tp"] += result_stat[iou]["tp"]
            self.final_result_stat[iou]["fp"] += result_stat[iou]["fp"]
            self.final_result_stat[iou]["score"] += result_stat[iou]["score"]

        # AdvCP: Save metrics and generate final report using MVP components
        if self.advcp_enabled:
            # Create dummy dataset for MVP accuracy function
            class DummyDataset:
                def case_generator(self, index=True, tag="multi_frame"):
                    # Convert OpenCOOD batch_data to MVP case format
                    case = {}
                    for vehicle_id, vehicle_data in batch_data.items():
                        case[vehicle_id] = {
                            "lidar": vehicle_data.get("lidar", np.zeros((100, 4))),
                            "lidar_pose": vehicle_data.get("lidar_pose", np.eye(4)),
                            "gt_bboxes": np.zeros((0, 7)),
                            "result_bboxes": np.zeros((0, 7))
                        }
                    yield 0, case
            
            # Calculate and save accuracy metrics
            accuracy_report = self.advcp_manager.accuracy.get_accuracy(DummyDataset(), self.model)
            with open(os.path.join(eval_dir, f"accuracy_{tick_number:05d}.pkl"), "wb") as f:
                pickle.dump(accuracy_report, f)
            
            # Calculate and save detection metrics
            detection_report = self.advcp_manager.detection.evaluate_single_vehicle(np.zeros((0, 7)), np.zeros((0, 7)))
            with open(os.path.join(eval_dir, f"detection_{tick_number:05d}.pkl"), "wb") as f:
                pickle.dump(detection_report, f)
            
            # Save visualizations using MVP Visualizer
            self.advcp_manager._save_visualizations_wrapper(eval_dir, tick_number)

    def final_eval(self):
        eval_dir = f"simulation_output/coperception/results/{self.opt.test_scenario}_{self.current_time}"
        os.makedirs(eval_dir, exist_ok=True)
        eval_utils.eval_final_results(self.final_result_stat, eval_dir, self.opt.global_sort_detections)

    def execute_attack(self, tick_number):
        """Execute attacks if AdvCP is enabled and it's time for an attack"""
        if self.advcp_enabled:
            self.attack_handler.execute_attack(tick_number)

    def apply_defense(self, tick_number):
        """Apply defense mechanisms if AdvCP is enabled"""
        if self.advcp_enabled and self.opt.apply_cad_defense:
            self.defense_handler.apply_defense(tick_number)

    def get_attack_metrics(self):
        """Get attack metrics if AdvCP is enabled"""
        if self.advcp_enabled:
            return self.accuracy.get_attack_metrics()
        return None

    def get_defense_metrics(self):
        """Get defense metrics if AdvCP is enabled"""
        if self.advcp_enabled:
            return self.detection.get_defense_metrics()
        return None

    def visualize_advcp_results(self):
        """Visualize AdvCP results if enabled"""
        if self.advcp_enabled:
            # Create dummy dataset for MVP Visualizer
            dummy_case = {0: {0: {"lidar": np.zeros((100, 4)), "lidar_pose": np.eye(4)}}}
            
            # Save visualizations using MVP Visualizer
            self.advcp_manager._save_visualizations_wrapper(f"simulation_output/advcp/results/{self.opt.test_scenario}_{self.current_time}", tick_number)

    def save_advcp_results(self, output_dir):
        """Save AdvCP results to specified directory"""
        if self.advcp_enabled:
            # Create dummy dataset for MVP accuracy function
            class DummyDataset:
                def case_generator(self, index=True, tag="multi_frame"):
                    # Convert OpenCOOD batch_data to MVP case format
                    case = {}
                    for vehicle_id, vehicle_data in batch_data.items():
                        case[vehicle_id] = {
                            "lidar": vehicle_data.get("lidar", np.zeros((100, 4))),
                            "lidar_pose": vehicle_data.get("lidar_pose", np.eye(4)),
                            "gt_bboxes": np.zeros((0, 7)),
                            "result_bboxes": np.zeros((0, 7))
                        }
                    yield 0, case
            
            # Calculate and save accuracy metrics
            accuracy_report = self.advcp_manager.accuracy.get_accuracy(DummyDataset(), self.model)
            with open(os.path.join(output_dir, f"accuracy_{tick_number:05d}.pkl"), "wb") as f:
                pickle.dump(accuracy_report, f)
            
            # Calculate and save detection metrics
            detection_report = self.advcp_manager.detection.evaluate_single_vehicle(np.zeros((0, 7)), np.zeros((0, 7)))
            with open(os.path.join(output_dir, f"detection_{tick_number:05d}.pkl"), "wb") as f:
                pickle.dump(detection_report, f)
            
            # Save visualizations using MVP Visualizer
            self.advcp_manager._save_visualizations_wrapper(output_dir, tick_number)

    def reset_advcp_state(self):
        """Reset AdvCP state for new simulation"""
        if self.advcp_enabled:
            self.advcp_manager.reset()
            self.attacker.cleanup()
            self.defender.cleanup()
            self.accuracy.cleanup()
            self.detection.cleanup()
            self.visualizer.cleanup()


class DirectoryProcessor:
    def __init__(self, source_directory="data_dumping", now_directory="data_dumping/sample/now"):
        self.source_directory = source_directory
        self.now_directory = now_directory

    def process_advcp_data(self, tick_number):
        """Process AdvCP-specific data for the given tick"""
        if not hasattr(self, "advcp_enabled") or not self.advcp_enabled:
            return

        # Process attack data
        attack_data_dir = os.path.join(self.source_directory, f"advcp_attacks_{tick_number:06d}")
        if os.path.exists(attack_data_dir):
            # Process attack data files
            pass

        # Process defense data
        defense_data_dir = os.path.join(self.source_directory, f"advcp_defense_{tick_number:06d}")
        if os.path.exists(defense_data_dir):
            # Process defense data files
            pass

    def detect_cameras(self, data_directory):
        inner_subdirectories = sorted([d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))])
        if not inner_subdirectories:
            return []

        sample_folder = os.path.join(data_directory, inner_subdirectories[0])
        camera_files = [f for f in os.listdir(sample_folder) if re.match(r"\d+_camera\d+\.png", f)]

        camera_ids = sorted(set(re.findall(r"_camera(\d+)\.png", f)[0] for f in camera_files if re.findall(r"_camera(\d+)\.png", f)))

        return [f"_camera{cam_id}.png" for cam_id in camera_ids]

    def process_directory(self, tick_number):
        number = f"{tick_number:06d}"
        postfixes = [".pcd", ".yaml"]

        subdirectories = sorted([d for d in os.listdir(self.source_directory) if os.path.isdir(os.path.join(self.source_directory, d))])

        if len(subdirectories) < 2:
            raise ValueError("Not enough subdirectories in source directory to process.")

        data_directory = os.path.join(self.source_directory, subdirectories[-2])

        camera_postfixes = self.detect_cameras(data_directory)
        postfixes.extend(camera_postfixes)

        inner_subdirectories = sorted([d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))])

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
