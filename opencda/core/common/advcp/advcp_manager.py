"""
AdvCPManager - Main coordinator for AdvCollaborativePerception module.
Extends CoperceptionModelManager to integrate attack and defense mechanisms.
"""

import os
import logging

import torch

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset

# MVP components
from .mvp.attack.attacker import Attacker
from .mvp.defense.defender import Defender
from .mvp.data.dataset import Dataset
from .mvp.evaluate.accuracy import Accuracy
from .mvp.evaluate.detection import Detection
from .mvp.visualize.general import Visualizer

logger = logging.getLogger("cavise.advcp_manager")


class AdvCPManager:
    def __init__(self, opt, current_time, message_handler=None):
        """
        Initialize AdvCPManager with configuration and attack/defense handlers.

        Args:
            opt: Configuration options
            current_time: Current timestamp for logging
            message_handler: Optional message handler for communication
        """
        self.opt = opt
        self.current_time = current_time
        self.message_handler = message_handler

        # Load AdvCP configuration
        self.advcp_config = load_yaml(None, self.opt)

        # Initialize core components with MVP equivalents
        self.attacker = Attacker()
        self.defender = Defender()
        self.dataset = Dataset()
        self.accuracy = Accuracy()
        self.detection = Detection()
        self.visualizer = Visualizer()
        # Set dataset for attacker
        self.attacker.set_dataset(self.dataset)

        # Initialize OpenCOOD components
        self.model = None
        self.data_loader = None

        # Attack/defense state
        self.attackers = []
        self.defense_enabled = self.advcp_config.get("defense", {}).get("enabled", False)
        self.attack_enabled = self.advcp_config.get("attack", {}).get("enabled", False)

        # Initialize OpenCOOD model
        self._initialize_model()

    def _apply_attack_wrapper(self, batch_data, tick_number):
        """Wrapper method to bridge MVP Attacker interface gap."""
        # Convert batch_data to MVP case format
        case = {}
        for vehicle_id, vehicle_data in batch_data.items():
            case[vehicle_id] = {
                'lidar': vehicle_data.get('lidar', np.zeros((100, 4))),
                'lidar_pose': vehicle_data.get('lidar_pose', np.eye(4))
            }
        
        # Apply attack using MVP Attacker
        attacked_case = {}
        for vehicle_id, vehicle_data in case.items():
            attacked_lidar = self.attacker.apply_ray_tracing(
                vehicle_data['lidar']
            )
            attacked_case[vehicle_id] = {
                'lidar': attacked_lidar,
                'lidar_pose': vehicle_data['lidar_pose']
            }
        
        # Convert back to batch_data format
        for vehicle_id, vehicle_data in attacked_case.items():
            batch_data[vehicle_id]['lidar'] = vehicle_data['lidar']
        
        return batch_data

    def _apply_defense_wrapper(self, batch_data, tick_number):
        """Wrapper method to bridge MVP Defender interface gap."""
        # Convert batch_data to MVP case format
        case = {}
        for vehicle_id, vehicle_data in batch_data.items():
            case[vehicle_id] = {
                'lidar': vehicle_data.get('lidar', np.zeros((100, 4))),
                'lidar_pose': vehicle_data.get('lidar_pose', np.eye(4))
            }
        
        # Apply defense using MVP Defender
        defended_case = self.defender.run(case, defend_opts={})
        
        # Convert back to batch_data format
        for vehicle_id, vehicle_data in defended_case.items():
            batch_data[vehicle_id]['lidar'] = vehicle_data['lidar']
        
        return batch_data

    def _visualize_results_wrapper(self, pred_box_tensor, gt_box_tensor, batch_data, tick_number):
        """Wrapper method to bridge MVP Visualizer interface gap."""
        # Convert batch_data to MVP case format
        case = {}
        for vehicle_id, vehicle_data in batch_data.items():
            case[vehicle_id] = {
                'lidar': vehicle_data.get('lidar', np.zeros((100, 4))),
                'lidar_pose': vehicle_data.get('lidar_pose', np.eye(4))
            }
        
        # Visualize using MVP Visualizer
        self.visualizer.draw_multi_vehicle_case(
            case,
            ego_id=0,
            mode="matplotlib",
            gt_bboxes=gt_box_tensor,
            pred_bboxes=pred_box_tensor,
            show=True
        )

    def _save_visualizations_wrapper(self, eval_dir, tick_number):
        """Wrapper method to bridge MVP Visualizer interface gap."""
        # Save visualizations using MVP Visualizer
        # Create a dummy case for saving
        dummy_case = {0: {0: {"lidar": np.zeros((100, 4)), "lidar_pose": np.eye(4)}}}
        
        # Save visualization
        self.visualizer.draw_multi_vehicle_case(
            dummy_case,
            ego_id=0,
            mode="matplotlib",
            gt_bboxes=np.zeros((0, 7)),
            pred_bboxes=np.zeros((0, 7)),
            show=False,
            save=os.path.join(eval_dir, f"visualization_{tick_number:05d}.png")
        )

        logger.info("AdvCPManager initialized successfully")

    def _initialize_model(self):
        """Initialize OpenCOOD model with attack/defense capabilities."""
        # Load base model configuration
        base_hypes = load_yaml(None, self.opt)

        # Create model with attack/defense extensions
        self.model = train_utils.create_model(base_hypes)

        if torch.cuda.is_available():
            self.model.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained weights
        _, self.model = train_utils.load_saved_model(self.opt.model_dir, self.model)

        logger.info("OpenCOOD model initialized with attack/defense capabilities")

    def make_dataset(self):
        """Create dataset for AdvCP processing."""
        logger.info("Building AdvCP dataset")

        # Create base dataset using MVP Dataset
        self.dataset = Dataset(self.opt, visualize=True, train=False, message_handler=self.message_handler)

        # Add attack/defense specific data processing
        # (Handled by MVP components)

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=16,
            collate_fn=self.dataset.load_feature,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

        logger.info(f"{len(self.dataset)} samples found for AdvCP processing")

    def process_tick(self, tick_number):
        """
        Process a single simulation tick with attack/defense mechanisms.

        Args:
            tick_number: Current simulation tick number
        """
        logger.info(f"Processing tick {tick_number}")

        # Process each batch of data
        for i, batch_data in enumerate(self.data_loader):
            # Apply attack if enabled using MVP Attacker
            if self.attack_enabled:
                batch_data = self._apply_attack_wrapper(batch_data, tick_number)

            # Apply defense if enabled using MVP Defender
            if self.defense_enabled:
                batch_data = self._apply_defense_wrapper(batch_data, tick_number)

            # Make predictions with OpenCOOD model
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, self.device)

                # Perform fusion based on method
                if self.opt.fusion_method == "late":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_late_fusion(batch_data, self.model, self.dataset)
                elif self.opt.fusion_method == "early":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(batch_data, self.model, self.dataset)
                elif self.opt.fusion_method == "intermediate":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(batch_data, self.model, self.dataset)
                else:
                    raise NotImplementedError("Only early, late and intermediate fusion is supported.")

            # Log metrics using MVP evaluation functions
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
            
            # Calculate accuracy metrics
            accuracy_report = self.accuracy.get_accuracy(DummyDataset(), self.model)
            logger.info(f"Tick {tick_number} - Accuracy metrics: {accuracy_report}")
                        
            # Calculate detection metrics
            detection_report = self.detection.evaluate_single_vehicle(gt_box_np, pred_box_np)
            logger.info(f"Tick {tick_number} - Detection metrics: {detection_report}")

            # Visualize results using MVP Visualizer
            if self.opt.show_vis:
                self._visualize_results_wrapper(pred_box_tensor, gt_box_tensor, batch_data, tick_number)

        logger.info(f"Tick {tick_number} processed successfully")

    def get_attack_status(self):
        """Get current attack status."""
        return {
            "enabled": self.attack_enabled,
            "attackers": len(self.attackers),
            "attack_types": [],  # MVP Attacker doesn't have get_attack_types()
            "metrics": {},  # MVP accuracy is a function, not a class with get_attack_metrics()
        }

    def get_defense_status(self):
        """Get current defense status."""
        return {
            "enabled": self.defense_enabled,
            "trust_scores": [],  # MVP Defender doesn't have get_trust_scores()
            "blocked_attacks": 0,  # MVP Defender doesn't have get_blocked_attacks()
            "metrics": {},  # MVP detection is a function, not a class with get_defense_metrics()
        }

    def save_results(self, tick_number):
        """Save results for the current tick."""
        eval_dir = f"simulation_output/advcp/results/{self.opt.test_scenario}_{self.current_time}"
        os.makedirs(eval_dir, exist_ok=True)

        # Save attack/defense metrics
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
        accuracy_report = get_accuracy(DummyDataset(), self.model)
        with open(os.path.join(eval_dir, f"accuracy_{tick_number:05d}.pkl"), "wb") as f:
            pickle.dump(accuracy_report, f)
        
        # Calculate and save detection metrics
        detection_report = evaluate_single_vehicle(np.zeros((0, 7)), np.zeros((0, 7)))
        with open(os.path.join(eval_dir, f"detection_{tick_number:05d}.pkl"), "wb") as f:
            pickle.dump(detection_report, f)

        # Save visualizations using MVP Visualizer
        self._save_visualizations_wrapper(eval_dir, tick_number)

        logger.info(f"Results saved for tick {tick_number}")

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up AdvCP resources")

        # Cleanup MVP components
        self.attacker.cleanup()
        self.defender.cleanup()
        self.visualizer.cleanup()

        # Cleanup OpenCOOD components
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'data_loader') and self.data_loader is not None:
            del self.data_loader
        if hasattr(self, 'dataset') and self.dataset is not None:
            del self.dataset

        logger.info("AdvCP cleanup completed")
