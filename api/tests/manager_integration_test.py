"""Optional real Go/Python protocol check without Docker or GPU workloads."""

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest


@pytest.mark.skipif(not os.environ.get("CAVISE_MANAGER_TEST_BINARY"), reason="Set CAVISE_MANAGER_TEST_BINARY to a built Go Manager")
def test_manager_enrollment_restart_and_revocation(tmp_path):
    """Registration survives restart without polling; retired credentials remain revoked."""

    def address():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return f"127.0.0.1:{listener.getsockname()[1]}"

    manager_address, runner_address = address(), address()
    manager_url, runner_url = "http://" + manager_address, "http://" + runner_address
    processes = []

    def launch(command, environment):
        process = subprocess.Popen(command, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(process)
        return process

    def wait_for(check):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                result = check()
                if result:
                    return result
            except (httpx.HTTPError, KeyError):
                # Startup and restart temporarily make the requested state unavailable.
                pass
            time.sleep(0.05)
        raise AssertionError("Protocol state did not converge within 15 seconds")

    manager_environment = dict(os.environ, CAVISE_MANAGER_ADMIN_TOKEN="integration-admin")
    manager_command = [os.environ["CAVISE_MANAGER_TEST_BINARY"], "--listen", manager_address, "--database", str(tmp_path / "manager.db")]
    try:
        manager = launch(manager_command, manager_environment)
        with httpx.Client(timeout=2) as client:
            wait_for(lambda: client.get(manager_url + "/healthz").status_code == 200)
            admin = {"Authorization": "Bearer integration-admin"}
            response = client.post(manager_url + "/api/v1/runners", headers=admin, json={"name": "Python integration"})
            assert response.status_code == 201
            registration = response.json()
            token = registration["token"]
            path = manager_url + "/api/v1/runners/" + registration["runner"]["id"]
            environment = dict(
                os.environ,
                CAVISE_RUNNER_STATE_DIR=str(tmp_path / "runner"),
                CAVISE_MANAGER_URL=manager_url,
                CAVISE_RUNNER_API_URL=runner_url,
                CAVISE_REGISTRATION_ID=registration["runner"]["id"],
                CAVISE_REGISTRATION_TOKEN=token,
            )
            environment.pop("CAVISE_REGISTRATION_TOKEN_FILE", None)
            command = [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                runner_address.split(":")[1],
                "--no-access-log",
            ]
            runner = launch(command, environment)

            def registered():
                response = client.get(path, headers=admin)
                record = response.json()
                return record if record.get("registration_state") == "active" else None

            first = wait_for(registered)
            time.sleep(0.3)
            assert registered()["last_seen_at"] == first["last_seen_at"]
            runner.terminate()
            runner.wait(timeout=5)
            manager.terminate()
            manager.wait(timeout=5)
            launch(manager_command, manager_environment)
            runner = launch(command, environment)
            second = wait_for(registered)
            assert second["runner_id"] == first["runner_id"]
            bearer = {"Authorization": "Bearer " + token}
            wait_for(lambda: client.get(runner_url + "/v1/status", headers=bearer).status_code == 200)
            assert second["last_seen_at"] == first["last_seen_at"]
            response = client.delete(path, headers=admin)
            assert response.status_code in (202, 204)
            wait_for(lambda: client.get(path, headers=admin).status_code == 404)
            bearer = {"Authorization": "Bearer " + token}
            assert client.get(runner_url + "/v1/status", headers=bearer).status_code == 401
            response = client.delete(runner_url + "/v1/registration", headers=bearer)
            assert response.status_code == 204
            runner.terminate()
            runner.wait(timeout=5)
            runner = launch(command, environment)
            wait_for(lambda: client.get(runner_url + "/healthz").status_code == 200)
            assert client.get(runner_url + "/v1/status", headers=bearer).status_code == 401
            response = client.delete(runner_url + "/v1/registration", headers=bearer)
            assert response.status_code == 204
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
