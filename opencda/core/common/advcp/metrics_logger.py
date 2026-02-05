"""
MetricsLogger - Logs attack and defense metrics for analysis.
Tracks performance, success rates, and system behavior.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger("cavise.advcp.metrics_logger")


class MetricsLogger:
    def __init__(self):
        """Initialize metrics logger."""
        self.attack_metrics = []
        self.defense_metrics = []
        self.performance_metrics = []
        self.current_session = self._initialize_session()

        logger.info("MetricsLogger initialized")

    def _initialize_session(self) -> Dict[str, Any]:
        """Initialize a new logging session."""
        return {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "attack_metrics": [],
            "defense_metrics": [],
            "performance_metrics": [],
            "config": {},
        }

    def log_attack_metrics(self, metrics: Dict, tick_number: int):
        """
        Log attack metrics.

        Args:
            metrics: Dictionary of attack metrics
            tick_number: Current simulation tick
        """
        attack_log = {"tick": tick_number, "timestamp": datetime.now().isoformat(), "metrics": metrics.copy()}

        self.attack_metrics.append(attack_log)
        self.current_session["attack_metrics"].append(attack_log)

        logger.debug("Logged attack metrics: %s", metrics)

    def log_defense_metrics(self, metrics: Dict, tick_number: int):
        """
        Log defense metrics.

        Args:
            metrics: Dictionary of defense metrics
            tick_number: Current simulation tick
        """
        defense_log = {"tick": tick_number, "timestamp": datetime.now().isoformat(), "metrics": metrics.copy()}

        self.defense_metrics.append(defense_log)
        self.current_session["defense_metrics"].append(defense_log)

        logger.debug("Logged defense metrics: %s", metrics)

    def log_performance_metrics(self, metrics: Dict, tick_number: int):
        """
        Log performance metrics.

        Args:
            metrics: Dictionary of performance metrics
            tick_number: Current simulation tick
        """
        performance_log = {"tick": tick_number, "timestamp": datetime.now().isoformat(), "metrics": metrics.copy()}

        self.performance_metrics.append(performance_log)
        self.current_session["performance_metrics"].append(performance_log)

        logger.debug("Logged performance metrics: %s", metrics)

    def log_attack_event(self, event_type: str, details: Dict, tick_number: int):
        """
        Log a specific attack event.

        Args:
            event_type: Type of attack event (e.g., "attack_scheduled", "attack_executed")
            details: Event details
            tick_number: Current simulation tick
        """
        event_log = {"tick": tick_number, "timestamp": datetime.now().isoformat(), "event_type": event_type, "details": details.copy()}

        self.attack_metrics.append(event_log)
        self.current_session["attack_metrics"].append(event_log)

        logger.info("Logged attack event: %s", event_type)

    def log_defense_event(self, event_type: str, details: Dict, tick_number: int):
        """
        Log a specific defense event.

        Args:
            event_type: Type of defense event (e.g., "anomaly_detected", "attack_blocked")
            details: Event details
            tick_number: Current simulation tick
        """
        event_log = {"tick": tick_number, "timestamp": datetime.now().isoformat(), "event_type": event_type, "details": details.copy()}

        self.defense_metrics.append(event_log)
        self.current_session["defense_metrics"].append(event_log)

        logger.info("Logged defense event: %s", event_type)

    def get_attack_metrics(self) -> List[Dict]:
        """Get all logged attack metrics."""
        return self.attack_metrics.copy()

    def get_defense_metrics(self) -> List[Dict]:
        """Get all logged defense metrics."""
        return self.defense_metrics.copy()

    def get_performance_metrics(self) -> List[Dict]:
        """Get all logged performance metrics."""
        return self.performance_metrics.copy()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        summary = self.current_session.copy()
        summary["end_time"] = datetime.now().isoformat()
        summary["duration_seconds"] = self._calculate_duration()
        summary["total_attacks"] = len([m for m in self.attack_metrics if m.get("event_type") == "attack_executed"])
        summary["blocked_attacks"] = len([m for m in self.defense_metrics if m.get("event_type") == "attack_blocked"])

        return summary

    def _calculate_duration(self) -> float:
        """Calculate session duration in seconds."""
        start_time = datetime.fromisoformat(self.current_session["start_time"])
        end_time = datetime.now()
        return (end_time - start_time).total_seconds()

    def save_metrics(self, directory: str, session_id: Optional[str] = None):
        """
        Save all metrics to files.

        Args:
            directory: Directory to save metrics
            session_id: Optional session ID (defaults to current session)
        """
        if session_id is None:
            session_id = self.current_session["session_id"]

        # Create directory
        os.makedirs(directory, exist_ok=True)

        # Save attack metrics
        attack_file = os.path.join(directory, f"{session_id}_attack_metrics.json")
        with open(attack_file, "w") as f:
            json.dump(self.attack_metrics, f, indent=2)

        # Save defense metrics
        defense_file = os.path.join(directory, f"{session_id}_defense_metrics.json")
        with open(defense_file, "w") as f:
            json.dump(self.defense_metrics, f, indent=2)

        # Save performance metrics
        performance_file = os.path.join(directory, f"{session_id}_performance_metrics.json")
        with open(performance_file, "w") as f:
            json.dump(self.performance_metrics, f, indent=2)

        # Save session summary
        summary_file = os.path.join(directory, f"{session_id}_session_summary.json")
        with open(summary_file, "w") as f:
            json.dump(self.get_session_summary(), f, indent=2)

        logger.info("Saved metrics to %s", directory)

    def load_metrics(self, directory: str, session_id: str):
        """
        Load metrics from files.

        Args:
            directory: Directory containing metrics files
            session_id: Session ID to load
        """
        # Load attack metrics
        attack_file = os.path.join(directory, f"{session_id}_attack_metrics.json")
        if os.path.exists(attack_file):
            with open(attack_file, "r") as f:
                self.attack_metrics = json.load(f)

        # Load defense metrics
        defense_file = os.path.join(directory, f"{session_id}_defense_metrics.json")
        if os.path.exists(defense_file):
            with open(defense_file, "r") as f:
                self.defense_metrics = json.load(f)

        # Load performance metrics
        performance_file = os.path.join(directory, f"{session_id}_performance_metrics.json")
        if os.path.exists(performance_file):
            with open(performance_file, "r") as f:
                self.performance_metrics = json.load(f)

        # Initialize session
        self.current_session = {
            "session_id": session_id,
            "start_time": self.attack_metrics[0]["timestamp"] if self.attack_metrics else datetime.now().isoformat(),
            "attack_metrics": self.attack_metrics,
            "defense_metrics": self.defense_metrics,
            "performance_metrics": self.performance_metrics,
            "config": {},
        }

        logger.info("Loaded metrics from %s", directory)

    def cleanup(self):
        """Cleanup metrics logger resources."""
        logger.info("Cleaning up MetricsLogger")
        self.attack_metrics.clear()
        self.defense_metrics.clear()
        self.performance_metrics.clear()
        logger.info("MetricsLogger cleanup completed")
