"""
AdvCPManager - Main coordinator for AdvCollaborativePerception module.
Extends CoperceptionModelManager to integrate attack and defense mechanisms.
"""

import os
import logging
import yaml
from typing import Dict, List, Optional

import torch
import open3d as o3d

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import simple_vis, vis_utils
from opencood.utils import eval_utils

from .attack_handler import AttackHandler
from .defense_handler import DefenseHandler
from .data_simulator import DataSimulator
from .metrics_logger import MetricsLogger
from .visualization import Visualization
from .utils import AdvCPUtils

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
        
        # Initialize core components
        self.attack_handler = AttackHandler(self.advcp_config)
        self.defense_handler = DefenseHandler(self.advcp_config)
        self.data_simulator = DataSimulator()
        self.metrics_logger = MetricsLogger()
        self.visualization = Visualization()
        
        # Initialize OpenCOOD components
        self.model = None
        self.dataset = None
        self.data_loader = None
        
        # Attack/defense state
        self.attackers = []
        self.defense_enabled = self.advcp_config.get("defense", {}).get("enabled", False)
        self.attack_enabled = self.advcp_config.get("attack", {}).get("enabled", False)
        
        # Initialize OpenCOOD model
        self._initialize_model()
        
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
        
        # Create base dataset
        self.dataset = build_dataset(self.opt, visualize=True, train=False, message_handler=self.message_handler)
        
        # Add attack/defense specific data processing
        self.data_simulator.prepare_dataset(self.dataset)
        
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=16,
            collate_fn=self.dataset.collate_batch_test,
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
            # Apply attack if enabled
            if self.attack_enabled:
                batch_data = self.attack_handler.apply_attack(batch_data, tick_number)
            
            # Apply defense if enabled
            if self.defense_enabled:
                batch_data = self.defense_handler.apply_defense(batch_data, tick_number)
            
            # Make predictions with OpenCOOD model
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, self.device)
                
                # Perform fusion based on method
                if self.opt.fusion_method == "late":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_late_fusion(
                        batch_data, self.model, self.dataset
                    )
                elif self.opt.fusion_method == "early":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(
                        batch_data, self.model, self.dataset
                    )
                elif self.opt.fusion_method == "intermediate":
                    pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(
                        batch_data, self.model, self.dataset
                    )
                else:
                    raise NotImplementedError("Only early, late and intermediate fusion is supported.")
            
            # Log metrics
            self.metrics_logger.log_metrics(
                pred_box_tensor, pred_score, gt_box_tensor, tick_number
            )
            
            # Visualize results
            if self.opt.show_vis:
                self.visualization.visualize_results(
                    pred_box_tensor, gt_box_tensor, batch_data, tick_number
                )
        
        logger.info(f"Tick {tick_number} processed successfully")
    
    def get_attack_status(self):
        """Get current attack status."""
        return {
            "enabled": self.attack_enabled,
            "attackers": len(self.attackers),
            "attack_types": self.attack_handler.get_attack_types(),
            "metrics": self.metrics_logger.get_attack_metrics()
        }
    
    def get_defense_status(self):
        """Get current defense status."""
        return {
            "enabled": self.defense_enabled,
            "trust_scores": self.defense_handler.get_trust_scores(),
            "blocked_attacks": self.defense_handler.get_blocked_attacks(),
            "metrics": self.metrics_logger.get_defense_metrics()
        }
    
    def save_results(self, tick_number):
        """Save results for the current tick."""
        eval_dir = f"simulation_output/advcp/results/{self.opt.test_scenario}_{self.current_time}"
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save attack/defense metrics
        self.metrics_logger.save_metrics(eval_dir, tick_number)
        
        # Save visualizations
        self.visualization.save_visualizations(eval_dir, tick_number)
        
        logger.info(f"Results saved for tick {tick_number}")
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up AdvCP resources")
        
        # Cleanup handlers
        self.attack_handler.cleanup()
        self.defense_handler.cleanup()
        self.metrics_logger.cleanup()
        
        logger.info("AdvCP cleanup completed")