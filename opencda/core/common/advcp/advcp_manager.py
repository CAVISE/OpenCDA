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

        # Create attacker instance with dataset from coperception manager
        self.attacker = attack_class(dataset=self.coperception_manager.opencood_dataset)
        logger.info(f"Initialized {self.attack_type} attacker")

    def _initialize_defender(self) -> None:
        """Initialize the defender if CAD defense is enabled."""
        if not self.apply_cad_defense:
            return

        self.defender = PerceptionDefender()
        logger.info("Initialized CAD defense mechanism")

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

    def process_tick(self, tick_number: int, batch_data: Optional[Dict] = None, 
                  predictions: Optional[Dict] = None) -> Tuple[Optional[Dict], Optional[float], Optional[Dict]]:
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
    
        # Determine attack stage
        if self.attack_type in ["lidar_remove_late", "lidar_spoof_late"]:
            # Late attacks need predictions
            if predictions is None:
                logger.error("Late attacks require predictions, but none provided")
                return None, None, None
            data_to_attack = predictions
        else:
            # Early/intermediate attacks need raw data
            if batch_data is None:
                data_to_attack = self._get_coperception_data(tick_number)
            else:
                data_to_attack = batch_data
        
        # Apply attack
        if self.attacker:
            modified_data = self._apply_attack(data_to_attack, tick_number)
        else:
            modified_data = data_to_attack
        
        # Apply defense if enabled
        defense_score = None
        defense_metrics = None
        if self.apply_cad_defense and self.defender:
            modified_data, defense_score, defense_metrics = self._apply_defense(
                modified_data, tick_number
            )
        
        return modified_data, defense_score, defense_metrics

    def _apply_attack(self, data: Dict, tick_number: int) -> Dict:
        """
        Apply attack to the perception data.

        Args:
            data: Perception data from coperception manager
            tick_number: Current simulation tick number

        Returns:
            Modified perception data with attack applied
        """
        if not self.attacker:
            return data

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

        modified_data = {}
        for vehicle_id, vehicle_data in data.items():
            if vehicle_id in attacker_vehicles:
                # This vehicle is an attacker
                attack_opts["attacker_vehicle_id"] = vehicle_id

                # Select target based on attack_target strategy
                target_info = self._select_attack_target(vehicle_data, vehicle_id)
                if target_info:
                    attack_opts["object_id"] = target_info["object_id"]
                    attack_opts["bboxes"] = target_info["bboxes"]
                    attack_opts["positions"] = target_info["positions"]

                # Apply attack
                try:
                    modified_vehicle_data, _ = self.attacker.run({vehicle_id: vehicle_data}, attack_opts)
                    modified_data[vehicle_id] = modified_vehicle_data[vehicle_id]
                except Exception as e:
                    logger.error(f"Attack failed for vehicle {vehicle_id}: {e}")
                    modified_data[vehicle_id] = vehicle_data  # Fallback to original data
            else:
                # Non-attacker vehicle, keep original data
                modified_data[vehicle_id] = vehicle_data

        return modified_data

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

                random.seed(42)
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
