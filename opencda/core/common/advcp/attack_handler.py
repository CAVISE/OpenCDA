"""
AttackHandler - Manages attack scheduling and execution for AdvCP.
Handles different attack types and coordinates with attack modules.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import AdvCPUtils

logger = logging.getLogger("cavise.advcp.attack_handler")


class AttackHandler:
    def __init__(self, config: Dict):
        """
        Initialize attack handler with configuration.
        
        Args:
            config: AdvCP configuration dictionary
        """
        self.config = config
        self.attack_types = config.get("attack", {}).get("types", [])
        self.attackers_ratio = config.get("attack", {}).get("attackers_ratio", 0.2)
        self.attack_interval = config.get("attack", {}).get("interval", 10)
        self.current_tick = 0
        
        # Initialize attack modules
        self.attack_modules = self._initialize_attack_modules()
        
        # Track active attacks
        self.active_attacks = []
        self.attack_history = []
        
        logger.info("AttackHandler initialized with %d attack types", len(self.attack_types))
    
    def _initialize_attack_modules(self) -> Dict[str, object]:
        """Initialize attack modules based on configuration."""
        attack_modules = {}
        
        # Initialize each attack type
        for attack_type in self.attack_types:
            if attack_type == "lidar_remove_early":
                from mvp.attack.lidar_remove_early_attacker import LidarRemoveEarlyAttacker
                attack_modules[attack_type] = LidarRemoveEarlyAttacker()
            elif attack_type == "lidar_remove_intermediate":
                from mvp.attack.lidar_remove_intermediate_attacker import LidarRemoveIntermediateAttacker
                attack_modules[attack_type] = LidarRemoveIntermediateAttacker()
            elif attack_type == "lidar_remove_late":
                from mvp.attack.lidar_remove_late_attacker import LidarRemoveLateAttacker
                attack_modules[attack_type] = LidarRemoveLateAttacker()
            elif attack_type == "lidar_spoof_early":
                from mvp.attack.lidar_spoof_early_attacker import LidarSpoofEarlyAttacker
                attack_modules[attack_type] = LidarSpoofEarlyAttacker()
            elif attack_type == "lidar_spoof_intermediate":
                from mvp.attack.lidar_spoof_intermediate_attacker import LidarSpoofIntermediateAttacker
                attack_modules[attack_type] = LidarSpoofIntermediateAttacker()
            elif attack_type == "lidar_spoof_late":
                from mvp.attack.lidar_spoof_late_attacker import LidarSpoofLateAttacker
                attack_modules[attack_type] = LidarSpoofLateAttacker()
            elif attack_type == "adv_shape":
                from mvp.attack.adv_shape_attacker import AdvShapeAttacker
                attack_modules[attack_type] = AdvShapeAttacker()
            else:
                logger.warning("Unknown attack type: %s", attack_type)
        
        return attack_modules
    
    def select_attackers(self, total_cavs: int) -> List[int]:
        """
        Select attackers based on the configured ratio.
        
        Args:
            total_cavs: Total number of CAVs in the simulation
            
        Returns:
            List of attacker CAV IDs
        """
        num_attackers = max(1, int(total_cavs * self.attackers_ratio))
        
        # Select random CAVs as attackers
        attackers = random.sample(range(total_cavs), num_attackers)
        
        logger.info("Selected %d attackers from %d CAVs", len(attackers), total_cavs)
        return attackers
    
    def schedule_attack(self, tick_number: int, cav_states: Dict) -> Optional[Dict]:
        """
        Schedule an attack if conditions are met.
        
        Args:
            tick_number: Current simulation tick
            cav_states: Current states of all CAVs
            
        Returns:
            Attack configuration if attack is scheduled, None otherwise
        """
        self.current_tick = tick_number
        
        # Check if attack should be scheduled
        if tick_number % self.attack_interval != 0:
            return None
        
        # Select target CAVs
        target_cavs = self._select_targets(cav_states)
        
        if not target_cavs:
            return None
        
        # Select attack type
        attack_type = random.choice(self.attack_types)
        
        # Create attack configuration
        attack_config = {
            "tick": tick_number,
            "attack_type": attack_type,
            "attackers": self.attackers,
            "targets": target_cavs,
            "attack_module": self.attack_modules[attack_type]
        }
        
        self.active_attacks.append(attack_config)
        self.attack_history.append(attack_config)
        
        logger.info("Scheduled attack: %s on %d targets", attack_type, len(target_cavs))
        return attack_config
    
    def _select_targets(self, cav_states: Dict) -> List[int]:
        """Select target CAVs for the attack."""
        # For now, select all non-attacker CAVs as targets
        targets = [cav_id for cav_id in cav_states if cav_id not in self.attackers]
        
        # Filter targets based on distance/visibility
        valid_targets = []
        for target_id in targets:
            # Check if target is within attack range
            if self._is_target_valid(cav_states[target_id]):
                valid_targets.append(target_id)
        
        return valid_targets
    
    def _is_target_valid(self, cav_state: Dict) -> bool:
        """Check if a target CAV is valid for attack."""
        # Check if CAV is visible and within range
        return cav_state.get("visible", True) and cav_state.get("distance", 100) < 50
    
    def apply_attack(self, batch_data: Dict, tick_number: int) -> Dict:
        """
        Apply active attacks to the batch data.
        
        Args:
            batch_data: Current batch data
            tick_number: Current simulation tick
            
        Returns:
            Modified batch data with attacks applied
        """
        modified_data = batch_data.copy()
        
        # Apply each active attack
        for attack in self.active_attacks:
            if attack["tick"] == tick_number:
                attack_module = attack["attack_module"]
                
                # Apply attack based on type
                if attack["attack_type"].startswith("lidar_remove"):
                    modified_data = self._apply_lidar_remove_attack(
                        modified_data, attack_module, attack
                    )
                elif attack["attack_type"].startswith("lidar_spoof"):
                    modified_data = self._apply_lidar_spoof_attack(
                        modified_data, attack_module, attack
                    )
                elif attack["attack_type"] == "adv_shape":
                    modified_data = self._apply_adv_shape_attack(
                        modified_data, attack_module, attack
                    )
        
        return modified_data
    
    def _apply_lidar_remove_attack(self, batch_data: Dict, attack_module: object, attack: Dict) -> Dict:
        """Apply LiDAR remove attack."""
        # Get attacker and target CAVs
        attacker_cav = attack["attackers"][0]  # For now, use first attacker
        target_cavs = attack["targets"]
        
        # Apply attack to each target
        for target_cav in target_cavs:
            # Get LiDAR data for target
            lidar_data = batch_data.get(target_cav, {}).get("lidar", None)
            
            if lidar_data is not None:
                # Apply attack using the attack module
                modified_lidar = attack_module.run(lidar_data, attack)
                
                # Update batch data
                if target_cav not in modified_data:
                    modified_data[target_cav] = {}
                modified_data[target_cav]["lidar"] = modified_lidar
        
        return modified_data
    
    def _apply_lidar_spoof_attack(self, batch_data: Dict, attack_module: object, attack: Dict) -> Dict:
        """Apply LiDAR spoof attack."""
        # Get attacker and target CAVs
        attacker_cav = attack["attackers"][0]
        target_cavs = attack["targets"]
        
        # Apply attack to each target
        for target_cav in target_cavs:
            # Get LiDAR data for target
            lidar_data = batch_data.get(target_cav, {}).get("lidar", None)
            
            if lidar_data is not None:
                # Apply attack using the attack module
                modified_lidar = attack_module.run(lidar_data, attack)
                
                # Update batch data
                if target_cav not in modified_data:
                    modified_data[target_cav] = {}
                modified_data[target_cav]["lidar"] = modified_lidar
        
        return modified_data
    
    def _apply_adv_shape_attack(self, batch_data: Dict, attack_module: object, attack: Dict) -> Dict:
        """Apply adversarial shape attack."""
        # Get attacker and target CAVs
        attacker_cav = attack["attackers"][0]
        target_cavs = attack["targets"]
        
        # Apply attack to each target
        for target_cav in target_cavs:
            # Get LiDAR data for target
            lidar_data = batch_data.get(target_cav, {}).get("lidar", None)
            
            if lidar_data is not None:
                # Apply attack using the attack module
                modified_lidar = attack_module.run(lidar_data, attack)
                
                # Update batch data
                if target_cav not in modified_data:
                    modified_data[target_cav] = {}
                modified_data[target_cav]["lidar"] = modified_lidar
        
        return modified_data
    
    def cleanup(self):
        """Cleanup attack handler resources."""
        logger.info("Cleaning up attack handler")
        
        # Cleanup attack modules
        for module in self.attack_modules.values():
            if hasattr(module, "cleanup"):
                module.cleanup()
        
        logger.info("Attack handler cleanup completed")
    
    def get_attack_types(self) -> List[str]:
        """Get list of available attack types."""
        return list(self.attack_modules.keys())
    
    def get_active_attacks(self) -> List[Dict]:
        """Get list of currently active attacks."""
        return self.active_attacks.copy()
    
    def get_attack_metrics(self) -> Dict:
        """Get attack metrics."""
        return {
            "total_attacks": len(self.attack_history),
            "successful_attacks": len([a for a in self.attack_history if a.get("success", False)]),
            "failed_attacks": len([a for a in self.attack_history if not a.get("success", False)]),
            "active_attacks": len(self.active_attacks)
        }