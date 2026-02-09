import os
import logging
import yaml
from typing import Dict, List, Optional, Tuple
from opencda.core.common.coperception_model_manager import CoperceptionModelManager


from mvp.attack.lidar_remove_early_attacker import LidarRemoveEarlyAttacker
from mvp.attack.lidar_remove_intermediate_attacker import LidarRemoveIntermediateAttacker
from mvp.attack.lidar_remove_late_attacker import LidarRemoveLateAttacker
from mvp.attack.lidar_spoof_early_attacker import LidarSpoofEarlyAttacker
from mvp.attack.lidar_spoof_intermediate_attacker import LidarSpoofIntermediateAttacker
from mvp.attack.lidar_spoof_late_attacker import LidarSpoofLateAttacker
from mvp.attack.adv_shape_attacker import AdvShapeAttacker

from mvp.defense.perception_defender import PerceptionDefender
from mvp.perception.opencood_perception import OpencoodPerception

logger = logging.getLogger("cavise.advcp_manager")


class AdvCPManager:
    """
    Advanced Collaborative Perception (AdvCP) Manager for applying attacks and defenses
    on collaborative perception data in real-time.
    """

    def __init__(self, opt: Dict, current_time: str, coperception_manager: "CoperceptionModelManager", message_handler: Optional[callable] = None):
        """
        Initialize AdvCP Manager.

        Args:
            opt: Configuration options including AdvCP settings
            current_time: Current timestamp for logging
            coperception_manager: Instance of CoperceptionModelManager
            message_handler: Optional message handler for communication
        """
        self.opt = opt
        self.current_time = current_time
        self.coperception_manager = coperception_manager
        self.message_handler = message_handler

        # Load AdvCP configuration
        self.advcp_config = self._load_advcp_config()

        # Initialize attack and defense components
        self.attacker = None
        self.defender = None
        self.perception = None
        self._initialize_perception()
        self._initialize_attacker()
        self._initialize_defender()

        # Attack/Defense flags
        self.with_advcp = opt.get("with_advcp", False)
        self.apply_cad_defense = opt.get("apply_cad_defense", False)

        # Attack parameters
        self.attackers_ratio = opt.get("attackers_ratio", 0.2)
        self.attack_type = opt.get("attack_type", "lidar_remove_early")
        self.attack_target = opt.get("attack_target", "random")

        # Defense parameters
        self.defense_threshold = opt.get("defense_threshold", 0.7)

        logger.info("AdvCP Manager initialized with configuration:")
        logger.info(f"  with_advcp: {self.with_advcp}")
        logger.info(f"  attack_type: {self.attack_type}")
        logger.info(f"  attack_target: {self.attack_target}")
        logger.info(f"  apply_cad_defense: {self.apply_cad_defense}")
        logger.info(f"  defense_threshold: {self.defense_threshold}")

    def _load_advcp_config(self) -> Dict:
        """Load AdvCP configuration from YAML file."""
        config_path = self.opt.get("advcp_config", "opencda/core/common/advcp/advcp_config.yaml")

        if not os.path.exists(config_path):
            logger.warning(f"AdvCP config file not found at {config_path}. Using default settings.")
            return {}

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config or {}

    def _initialize_attacker(self) -> None:
        """Initialize the appropriate attacker based on configuration."""
        if not self.with_advcp:
            return

        attack_class_map = {
            "lidar_remove_early": LidarRemoveEarlyAttacker,
            "lidar_remove_intermediate": LidarRemoveIntermediateAttacker,
            "lidar_remove_late": LidarRemoveLateAttacker,
            "lidar_spoof_early": LidarSpoofEarlyAttacker,
            "lidar_spoof_intermediate": LidarSpoofIntermediateAttacker,
            "lidar_spoof_late": LidarSpoofLateAttacker,
            "adv_shape": AdvShapeAttacker,
        }

        attack_class = attack_class_map.get(self.attack_type)
        if not attack_class:
            logger.error(f"Unsupported attack type: {self.attack_type}")
            return

        # Create attacker instance with appropriate parameters
        # Late and intermediate attackers require perception parameter
        if self.attack_type in ["lidar_remove_late", "lidar_spoof_late", "lidar_remove_intermediate", "lidar_spoof_intermediate"]:
            if self.perception is None:
                logger.error(f"Perception not initialized, cannot create {self.attack_type} attacker")
                return
            self.attacker = attack_class(perception=self.perception, dataset=self.coperception_manager.opencood_dataset)
        elif self.attack_type == "adv_shape":
            # AdvShapeAttacker accepts perception as optional parameter
            self.attacker = attack_class(perception=self.perception, dataset=self.coperception_manager.opencood_dataset)
        else:
            # Early attackers only need dataset
            self.attacker = attack_class(dataset=self.coperception_manager.opencood_dataset)
        logger.info(f"Initialized {self.attack_type} attacker")

    def _initialize_defender(self) -> None:
        """Initialize the defender if CAD defense is enabled."""
        if not self.apply_cad_defense:
            return

        self.defender = PerceptionDefender()
        logger.info("Initialized CAD defense mechanism")

    def _initialize_perception(self) -> None:
        """Initialize OpencoodPerception for preprocessing raw data to OpenCOOD format."""
        if not self.with_advcp:
            return

        try:
            # Get fusion method and model name from coperception manager
            fusion_method = self.coperception_manager.opt.get("fusion_method", "early")
            model_name = self.opt.get("model_name", "pointpillar")

            # Initialize OpencoodPerception
            self.perception = OpencoodPerception(fusion_method=fusion_method, model_name=model_name)
            logger.info(f"Initialized OpencoodPerception with fusion_method={fusion_method}, model_name={model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpencoodPerception: {e}")
            self.perception = None

    def _get_coperception_data(self, tick_number: int) -> Dict:
        """
        Get raw perception data from coperception manager without causing circular dependency.
        This method directly accesses the raw data from real-time simulations.

        Args:
            tick_number: Current simulation tick number

        Returns:
            Dictionary containing raw perception data
        """
        # Direct access to raw data without calling make_prediction
        raw_data = self.coperception_manager._get_raw_data(tick_number)

        if raw_data is None:
            logger.warning(f"No raw data available for tick {tick_number}")
            return {}

        return raw_data

    def process_tick(
        self, tick_number: int, batch_data: Optional[Dict] = None, predictions: Optional[Dict] = None
    ) -> Tuple[Optional[Dict], Optional[float], Optional[Dict]]:
        """
        Process a single simulation tick with AdvCP capabilities.

        Args:
            tick_number: Current simulation tick number
            batch_data: Pre-inference batch data (for early/intermediate attacks)
            predictions: Post-inference predictions (for late attacks)

        Returns:
            Tuple of (modified_data, defense_score, defense_metrics)
        """
        if not self.with_advcp:
            return None, None, None

        # Determine attack stage and prepare data accordingly
        if self.attack_type in ["lidar_remove_late", "lidar_spoof_late"]:
            # Late attacks need both raw data and predictions
            # Raw data is needed for the attack to work (to extract target bbox, etc.)
            # Predictions are the output that will be modified
            raw_data = self._get_coperception_data(tick_number)
            if raw_data is None:
                logger.warning(f"No raw data available for tick {tick_number}")
                return None, None, None

            if predictions is None:
                logger.error("Late attacks require predictions, but none provided")
                return None, None, None

            # Apply late attack to raw data (which internally uses predictions)
            if self.attacker:
                modified_predictions = self._apply_attack(raw_data, predictions, tick_number)
            else:
                modified_predictions = predictions

            # Apply defense if enabled
            defense_score = None
            defense_metrics = None
            if self.apply_cad_defense and self.defender:
                # For late attacks, we need to convert predictions back to format expected by defender
                # The defender expects multi_frame_case with raw data + predictions
                defended_data, defense_score, defense_metrics = self._apply_defense(raw_data, modified_predictions, tick_number)
                return defended_data, defense_score, defense_metrics

            return modified_predictions, defense_score, defense_metrics
        else:
            # Early/intermediate attacks need RAW data (not preprocessed batch_data)
            # Get raw data from coperception manager
            raw_data = self._get_coperception_data(tick_number)
            if raw_data is None:
                logger.warning(f"No raw data available for tick {tick_number}")
                return None, None, None

            # Apply attack to raw data
            if self.attacker:
                attacked_data = self._apply_attack(raw_data, tick_number)
            else:
                attacked_data = raw_data

            # Convert attacked raw data back to OpenCOOD format using preprocessor
            if self.perception is not None:
                try:
                    # Determine ego vehicle ID (typically the first vehicle or "ego")
                    ego_id = self._get_ego_vehicle_id(raw_data)

                    # Apply appropriate preprocessor based on attack type
                    if self.attack_type in ["lidar_remove_early", "lidar_spoof_early"]:
                        preprocessed_data = self.perception.early_preprocess(attacked_data, ego_id)
                    elif self.attack_type in ["lidar_remove_intermediate", "lidar_spoof_intermediate"]:
                        preprocessed_data = self.perception.intermediate_preprocess(attacked_data, ego_id)
                    else:
                        logger.warning(f"Unknown attack type for preprocessing: {self.attack_type}")
                        preprocessed_data = None

                    if preprocessed_data is not None:
                        # Apply defense if enabled
                        defense_score = None
                        defense_metrics = None
                        if self.apply_cad_defense and self.defender:
                            preprocessed_data, defense_score, defense_metrics = self._apply_defense(preprocessed_data, tick_number)

                        return preprocessed_data, defense_score, defense_metrics
                except Exception as e:
                    logger.error(f"Failed to preprocess attacked data: {e}")
                    return None, None, None
            else:
                logger.warning("OpencoodPerception not initialized, cannot preprocess attacked data")
                return None, None, None

    def _apply_attack(self, data: Dict, predictions: Optional[Dict] = None, tick_number: int = None) -> Dict:
        """
        Apply attack to the perception data.

        Args:
            data: Perception data from coperception manager (raw format for all attacks)
            predictions: Optional predictions dict (for late attacks)
            tick_number: Current simulation tick number

        Returns:
            Modified perception data with attack applied
        """
        if not self.attacker:
            return data

        # Check if this is a late attack
        if self.attack_type in ["lidar_remove_late", "lidar_spoof_late"]:
            # Late attacks need raw data and predictions
            if predictions is None:
                logger.error("Late attacks require predictions parameter")
                return predictions if predictions is not None else data

            # Format data as multi-frame case for late attacks
            # Late attacks expect: multi_frame_case[frame_id][vehicle_id]
            multi_frame_case = {tick_number: data}

            # Add predictions to the multi-frame case for the ego vehicle
            ego_id = self._get_ego_vehicle_id(data)
            if ego_id in multi_frame_case[tick_number]:
                # Convert predictions to numpy arrays if they are tensors
                pred_bboxes = predictions["pred_bboxes"]
                pred_scores = predictions["pred_scores"]

                if hasattr(pred_bboxes, "cpu"):
                    pred_bboxes = pred_bboxes.cpu().numpy()
                if hasattr(pred_scores, "cpu"):
                    pred_scores = pred_scores.cpu().numpy()

                multi_frame_case[tick_number][ego_id]["pred_bboxes"] = pred_bboxes
                multi_frame_case[tick_number][ego_id]["pred_scores"] = pred_scores

            # Determine attacker and victim vehicles
            attacker_vehicles = self._select_attacker_vehicles()
            if len(attacker_vehicles) == 0:
                logger.warning("No attacker vehicles available")
                return predictions

            attacker_id = attacker_vehicles[0]
            victim_id = ego_id  # Attack the ego vehicle

            # Select attack target
            attack_target = self._select_attack_target(data[attacker_id], attacker_id)

            # Prepare attack options
            attack_opts = {
                "frame_ids": [tick_number],
                "attacker_vehicle_id": attacker_id,
                "victim_vehicle_id": victim_id,
            }

            if attack_target:
                if "object_id" in attack_target:
                    attack_opts["object_id"] = attack_target["object_id"]
                if "bboxes" in attack_target:
                    attack_opts["bboxes"] = attack_target["bboxes"]
                if "positions" in attack_target:
                    attack_opts["positions"] = {tick_number: attack_target["positions"]}

            # Apply late attack
            try:
                attacked_case, attack_info = self.attacker.run(multi_frame_case, attack_opts)

                # Extract the modified predictions for the ego vehicle
                if tick_number in attacked_case and ego_id in attacked_case[tick_number]:
                    modified_predictions = {
                        "pred_bboxes": attacked_case[tick_number][ego_id].get("pred_bboxes", predictions["pred_bboxes"]),
                        "pred_scores": attacked_case[tick_number][ego_id].get("pred_scores", predictions["pred_scores"]),
                        "gt_bboxes": predictions.get("gt_bboxes"),
                    }
                    return modified_predictions
                else:
                    logger.warning(f"Late attack did not return data for tick {tick_number}")
                    return predictions
            except Exception as e:
                logger.error(f"Late attack failed: {e}")
                return predictions

        # For early/intermediate attacks, format data as multi-frame case
        # Attacks expect: multi_frame_case[frame_id][vehicle_id]
        multi_frame_case = {tick_number: data}

        # Determine which vehicles are attackers based on ratio
        attacker_vehicles = self._select_attacker_vehicles()

        # Prepare attack options
        attack_opts = {
            "frame_ids": [tick_number],
            "attacker_vehicle_id": None,  # Will be set per vehicle
            "object_id": None,  # Will be set per vehicle
            "bboxes": None,  # Will be set per vehicle
            "positions": None,  # For spoofing attacks
        }

        # Apply attack to each attacker vehicle
        try:
            # Run the attacker on the multi-frame case
            attacked_case, attack_info = self.attacker.run(multi_frame_case, attack_opts)

            # Extract the attacked data for the current tick
            if tick_number in attacked_case:
                return attacked_case[tick_number]
            else:
                logger.warning(f"Attack did not return data for tick {tick_number}")
                return data
        except Exception as e:
            logger.error(f"Attack failed: {e}")
            return data

    def _select_attacker_vehicles(self) -> List[str]:
        """Select which vehicles will be attackers based on ratio."""
        if self.attack_target == "all_non_attackers":
            # All vehicles except one are attackers
            all_vehicles = list(self.coperception_manager.opencood_dataset.vehicle_ids)
            return all_vehicles[:-1] if len(all_vehicles) > 1 else all_vehicles

        # Randomly select attackers based on ratio
        all_vehicles = list(self.coperception_manager.opencood_dataset.vehicle_ids)
        num_attackers = max(1, int(len(all_vehicles) * self.attackers_ratio))

        import random

        random.seed(42)  # For reproducibility
        return random.sample(all_vehicles, num_attackers)

    def _select_attack_target(self, vehicle_data: Dict, vehicle_id: str) -> Optional[Dict]:
        """Select attack target based on strategy."""
        if self.attack_target == "random":
            # Randomly select a target object
            if len(vehicle_data["object_ids"]) > 0:
                import random

                random.seed(22)
                obj_idx = random.randint(0, len(vehicle_data["object_ids"]) - 1)
                return {"object_id": vehicle_data["object_ids"][obj_idx], "bboxes": [vehicle_data["gt_bboxes"][obj_idx]], "positions": None}

        elif self.attack_target == "specific_vehicle":
            # Attack a specific predefined vehicle (for testing)
            # This would need to be configured in the attack options
            pass

        elif self.attack_target == "all_non_attackers":
            # Attack all objects from non-attacker vehicles
            # This would require coordination between attackers
            pass

        return None

    def _get_ego_vehicle_id(self, raw_data: Dict) -> str:
        """
        Get the ego vehicle ID from raw data.

        Args:
            raw_data: Raw perception data dictionary

        Returns:
            Ego vehicle ID (typically "ego" or the first vehicle ID)
        """
        # Try to find "ego" key first
        if "ego" in raw_data:
            return "ego"

        # Otherwise, return the first vehicle ID
        if isinstance(raw_data, dict) and len(raw_data) > 0:
            return list(raw_data.keys())[0]

        # Fallback
        return "ego"

    def _apply_defense(self, data: Dict, tick_number: int) -> Tuple[Dict, Optional[float], Optional[Dict]]:
        """
        Apply CAD defense to the perception data.

        Args:
            data: Perception data (possibly already attacked)
            tick_number: Current simulation tick number

        Returns:
            Tuple of (defended_data, defense_score, defense_metrics)
        """
        if not self.defender:
            return data, None, None

        try:
            # Prepare multi-frame case for defense (using current tick data)
            multi_frame_case = {tick_number: data}

            # Defense options
            defend_opts = {"frame_ids": [tick_number], "vehicle_ids": list(data.keys())}

            # Run defense
            defended_data, score, metrics = self.defender.run(multi_frame_case, defend_opts)

            # Return only the data for the current tick
            return defended_data[tick_number], score, metrics

        except Exception as e:
            logger.error(f"Defense failed: {e}")
            return data, None, None

    def get_attack_statistics(self) -> Dict:
        """Get statistics about applied attacks."""
        if not self.attacker:
            return {}

        # This would need to be implemented based on the attacker's capabilities
        # For now, return basic information
        return {"attack_type": self.attack_type, "attackers_count": len(self._select_attacker_vehicles()), "enabled": self.with_advcp}

    def get_defense_statistics(self) -> Dict:
        """Get statistics about applied defenses."""
        if not self.defender:
            return {}

        return {
            "defense_enabled": self.apply_cad_defense,
            "threshold": self.defense_threshold,
            "applied": False,  # Would need to track this
        }
