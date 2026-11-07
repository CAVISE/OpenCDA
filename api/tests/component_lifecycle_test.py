"""Regression coverage for interrupted startup rollback."""

import importlib.util
import os
from pathlib import Path
import signal
from argparse import Namespace

import pytest


def test_interrupt_during_rollback_still_stops_sumo(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "opencda/scenario_testing/component_lifecycle.py"
    spec = importlib.util.spec_from_file_location("component_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def request(url, method="POST"):
        calls.append(url)
        if url.endswith("/status"):
            return '{"state": "IDLE"}'
        if url == "carla/v1/run":
            raise RuntimeError("startup failed")
        if url == "carla/v1/stop":
            os.kill(os.getpid(), signal.SIGINT)
        return ""

    monkeypatch.setattr(module, "adapter_request", request)
    previous = signal.getsignal(signal.SIGINT)
    options = Namespace(with_capi=False, cosim=True, test_scenario="demo", sumo_adapter_url="sumo", carla_adapter_url="carla")
    with pytest.raises(RuntimeError, match="startup failed"):
        with module.ManagedComponents(options):
            pytest.fail("Startup must fail")
    assert calls[-2:] == ["carla/v1/stop", "sumo/v1/stop"]
    assert signal.getsignal(signal.SIGINT) == previous
