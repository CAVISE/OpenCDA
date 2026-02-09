import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("cavise.advcp_manager")


class AdvCPConfigValidator:
    """
    Validates and processes AdvCP configuration parameters.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.errors = []
        self.warnings = []

    def validate(self) -> bool:
        """Validate all configuration parameters."""
        self.errors = []
        self.warnings = []

        self._validate_advcp_enabled()
        self._validate_attack_parameters()
        self._validate_defense_parameters()
        self._validate_config_file()

        if self.errors:
            for error in self.errors:
                logger.error(f"Config Error: {error}")
            return False

        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Config Warning: {warning}")

        return True

    def _validate_advcp_enabled(self) -> None:
        """Validate AdvCP enabled flag."""
        with_advcp = self.config.get("with_advcp", False)
        if not isinstance(with_advcp, bool):
            self.errors.append("with_advcp must be a boolean value")

    def _validate_attack_parameters(self) -> None:
        """Validate attack-related parameters."""
        if not self.config.get("with_advcp", False):
            return  # No need to validate if AdvCP is disabled

        # Validate attackers_ratio
        attackers_ratio = self.config.get("attackers_ratio", 0.2)
        if not isinstance(attackers_ratio, (int, float)) or not (0.0 <= attackers_ratio <= 1.0):
            self.errors.append("attackers_ratio must be a float between 0.0 and 1.0")

        # Validate attack_type
        attack_type = self.config.get("attack_type", "lidar_remove_early")
        valid_attack_types = [
            "lidar_remove_early",
            "lidar_remove_intermediate",
            "lidar_remove_late",
            "lidar_spoof_early",
            "lidar_spoof_intermediate",
            "lidar_spoof_late",
            "adv_shape",
        ]
        if attack_type not in valid_attack_types:
            self.errors.append(f"attack_type must be one of: {valid_attack_types}")

        # Validate attack_target
        attack_target = self.config.get("attack_target", "random")
        valid_targets = ["random", "specific_vehicle", "all_non_attackers"]
        if attack_target not in valid_targets:
            self.errors.append(f"attack_target must be one of: {valid_targets}")

    def _validate_defense_parameters(self) -> None:
        """Validate defense-related parameters."""
        apply_cad_defense = self.config.get("apply_cad_defense", False)
        if not isinstance(apply_cad_defense, bool):
            self.errors.append("apply_cad_defense must be a boolean value")

        if apply_cad_defense:
            defense_threshold = self.config.get("defense_threshold", 0.7)
            if not isinstance(defense_threshold, (int, float)) or not (0.0 <= defense_threshold <= 1.0):
                self.errors.append("defense_threshold must be a float between 0.0 and 1.0")

    def _validate_config_file(self) -> None:
        """Validate the AdvCP configuration file path."""
        config_path = self.config.get("advcp_config", "opencda/core/common/advcp/advcp_config.yaml")

        if not os.path.exists(config_path):
            self.warnings.append(f"AdvCP config file not found at {config_path}. Using default settings.")
        else:
            try:
                with open(config_path, "r") as f:
                    yaml.safe_load(f)
            except Exception as e:
                self.errors.append(f"Invalid YAML in config file {config_path}: {e}")

    def get_processed_config(self) -> Dict[str, Any]:
        """Return processed and validated configuration."""
        if not self.validate():
            raise ValueError("Configuration validation failed")

        # Process and normalize configuration values
        processed_config = self.config.copy()

        # Normalize boolean values
        processed_config["with_advcp"] = bool(processed_config.get("with_advcp", False))
        processed_config["apply_cad_defense"] = bool(processed_config.get("apply_cad_defense", False))

        # Normalize numeric values
        processed_config["attackers_ratio"] = float(processed_config.get("attackers_ratio", 0.2))
        processed_config["defense_threshold"] = float(processed_config.get("defense_threshold", 0.7))

        return processed_config


class AdvCPConfigLoader:
    """
    Loads and processes AdvCP configuration from multiple sources.
    """

    def __init__(self):
        self.default_config = self._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default AdvCP configuration."""
        return {
            "with_advcp": False,
            "advcp_config": "opencda/core/common/advcp/advcp_config.yaml",
            "attackers_ratio": 0.2,
            "attack_type": "lidar_remove_early",
            "attack_target": "random",
            "apply_cad_defense": False,
            "defense_threshold": 0.7,
            "attack_parameters": {"dense": 3, "sync": 1, "advshape": 1},
            "defense_parameters": {"threshold": 0.7, "method": "perception"},
        }

    def load_from_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from command-line arguments."""
        config = self.default_config.copy()

        # Update with provided arguments
        for key in config.keys():
            if key in args:
                config[key] = args[key]

        return config

    def load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(file_path):
            logger.warning(f"Config file not found: {file_path}")
            return self.default_config.copy()

        try:
            with open(file_path, "r") as f:
                file_config = yaml.safe_load(f) or {}

            # Merge with default config
            config = self.default_config.copy()
            config.update(file_config)
            return config

        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return self.default_config.copy()

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        merged_config = self.default_config.copy()

        for config in configs:
            for key, value in config.items():
                if key in merged_config:
                    # For nested dictionaries, merge recursively
                    if isinstance(merged_config[key], dict) and isinstance(value, dict):
                        merged_config[key].update(value)
                    else:
                        merged_config[key] = value
                else:
                    merged_config[key] = value

        return merged_config

    def validate_and_process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and process configuration."""
        validator = AdvCPConfigValidator(config)
        return validator.get_processed_config()


# Convenience functions for configuration management
def load_advcp_config(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load and validate AdvCP configuration from various sources.

    Args:
        args: Optional dictionary of configuration parameters

    Returns:
        Validated and processed configuration dictionary
    """
    loader = AdvCPConfigLoader()

    # Load from different sources in order of priority
    config_sources = []

    # 1. Command-line arguments (highest priority)
    if args:
        config_sources.append(loader.load_from_args(args))

    # 2. Configuration file
    default_config_path = "opencda/core/common/advcp/advcp_config.yaml"
    config_path = args["advcp_config"] if args and "advcp_config" in args else default_config_path
    config_sources.append(loader.load_from_file(config_path))

    # 3. Merge all sources
    merged_config = loader.merge_configs(*config_sources)

    # 4. Validate and process
    return loader.validate_and_process(merged_config)


def save_advcp_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Save AdvCP configuration to a YAML file.

    Args:
        config: Configuration dictionary to save
        file_path: Path to save the configuration file
    """
    try:
        with open(file_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        logger.info(f"Saved AdvCP configuration to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save config to {file_path}: {e}")


def get_advcp_usage_info() -> str:
    """
    Get usage information for AdvCP configuration.

    Returns:
        String with usage information and examples
    """
    usage_info = """
    AdvCP Configuration Usage:

    Command-line arguments:
    --with-advcp               Enable AdvCP module
    --advcp-config PATH        Path to AdvCP config file
    --attackers-ratio FLOAT    Ratio of attackers (0.0-1.0)
    --attack-type TYPE         Attack type (lidar_remove_early, lidar_spoof_early, adv_shape)
    --attack-target TARGET     Attack target (random, specific_vehicle, all_non_attackers)
    --apply-cad-defense        Enable CAD defense
    --defense-threshold FLOAT  Trust threshold for CAD defense (0.0-1.0)

    Example usage:
    python opencda.py --with-advcp --attackers-ratio 0.3 --attack-type lidar_spoof_early

    Configuration file example (advcp_config.yaml):
    with_advcp: true
    attackers_ratio: 0.25
    attack_type: "lidar_remove_intermediate"
    attack_target: "random"
    apply_cad_defense: true
    defense_threshold: 0.8
    """
    return usage_info
