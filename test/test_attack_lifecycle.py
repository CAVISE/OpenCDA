"""Regression tests for independently removable service interceptors."""

import unittest
from unittest.mock import patch

from opencda.core.application.behavior.capability import Capability
from opencda.core.attack.adversary_framework import (
    Attack,
    AttackSpec,
    AttackStageResult,
    ConditionSpec,
    RuntimeStatus,
    StageSpec,
    Status,
    TriggerSourceSpec,
    install_output_interceptor,
)
from opencda.core.attack.adversary_framework.interceptor_manager import InterceptorManager
from opencda.core.attack.adversary_framework.stages.dropper.stage import DropperStage
from opencda.scenario_testing.types import SimulationSnapshot


class Service:
    service_type = "test"

    def output(self):
        return (1, 2)

    @property
    def capability_bindings(self):
        return {Capability.RESPONSE_SUBMIT: self.output}


class InterceptorTests(unittest.TestCase):
    def test_removal_orders_do_not_resurrect_interceptors(self):
        for order in ((0, 1), (1, 0)):
            with self.subTest(order=order):
                service = Service()
                manager = InterceptorManager()
                handles = [
                    manager.install(service, service.output, lambda value: value + (3,)),
                    manager.install(service, service.output, lambda value: value + (4,)),
                ]
                self.assertEqual(service.output(), (1, 2, 3, 4))
                handles[order[0]]()
                self.assertEqual(service.output(), (1, 2, 4) if order[0] == 0 else (1, 2, 3))
                handles[order[0]]()  # Repeated removal must be harmless.
                handles[order[1]]()
                self.assertEqual(service.output(), (1, 2))
                self.assertNotIn("output", vars(service))
                self.assertFalse(manager._bindings)

    def test_instance_override_is_restored(self):
        service = Service()

        def output():
            return (9,)

        service.output = output
        remove = InterceptorManager().install(service, service.output, lambda value: ())
        self.assertEqual(service.output(), ())
        remove()
        self.assertIs(service.output, output)

    def test_services_are_independent(self):
        manager = InterceptorManager()
        first, second = Service(), Service()
        a = manager.install(first, first.output, lambda value: ())
        b = manager.install(second, second.output, lambda value: (8,))
        a()
        self.assertEqual(first.output(), (1, 2))
        self.assertEqual(second.output(), (8,))
        b()

    def test_public_helper_handles_dynamic_capability_bindings(self):
        service = Service()
        a = install_output_interceptor(service, Capability.RESPONSE_SUBMIT, rewrite_result=lambda value: value + (3,))
        b = install_output_interceptor(service, Capability.RESPONSE_SUBMIT, rewrite_result=lambda value: value + (4,))
        a()
        self.assertEqual(service.output(), (1, 2, 4))
        b()
        self.assertEqual(service.output(), (1, 2))

    def test_external_replacement_is_not_overwritten_on_cleanup(self):
        service = Service()
        remove = InterceptorManager().install(service, service.output, lambda value: ())

        def replacement():
            return (7,)

        service.output = replacement
        remove()
        self.assertIs(service.output, replacement)


class LifecycleTests(unittest.TestCase):
    def make_attack(self, stop=None):
        stage = DropperStage((Capability.RESPONSE_SUBMIT,))
        spec = AttackSpec(name="test", stages=(StageSpec(id="drop", type="dropper", stage_stop_trigger=stop),))
        attack = Attack(spec, (stage,))
        attack.mark_started()
        self.addCleanup(attack.deactivate)
        return attack

    def run_attack(self, attack, services):
        return attack.run_stage_lifecycle(None, SimulationSnapshot(tick=1), tuple(services))

    def test_persistent_stage_lasts_until_attack_stop(self):
        service = Service()
        attack = self.make_attack()
        self.run_attack(attack, [service])
        self.assertTrue(attack.is_active)
        self.assertEqual(attack.stage_runtimes[0].status, RuntimeStatus.ACTIVE)
        self.assertEqual(service.output(), ())
        attack.mark_stopped()
        self.assertEqual(service.output(), (1, 2))
        attack.mark_stopped()

    def test_all_terminal_transitions_remove_interceptors(self):
        for method in ("mark_succeeded", "mark_failed", "mark_stopped", "reset_runtime", "deactivate"):
            with self.subTest(method=method):
                service = Service()
                attack = self.make_attack()
                self.run_attack(attack, [service])
                getattr(attack, method)()
                self.assertEqual(service.output(), (1, 2))
                self.assertNotIn("output", vars(service))

    def test_stage_stop_completes_attack(self):
        condition = ConditionSpec(source=TriggerSourceSpec(kind="snapshot"), verb="exists")
        attack = self.make_attack(condition)
        service = Service()
        self.run_attack(attack, [service])
        with patch.object(attack, "_evaluate_condition", return_value=True):
            self.run_attack(attack, [service])
        self.assertEqual(attack.status, RuntimeStatus.SUCCESS)
        self.assertEqual(service.output(), (1, 2))

    def test_exception_after_partial_installation_cleans_up(self):
        service = Service()
        attack = self.make_attack()
        execute = attack.stages[0].execute

        def fail(services):
            execute(services)
            raise RuntimeError("installation failed")

        with patch.object(attack.stages[0], "execute", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "installation failed"):
                self.run_attack(attack, [service])
        self.assertEqual(attack.status, RuntimeStatus.FAIL)
        self.assertEqual(attack.previous_status, RuntimeStatus.STARTED)
        self.assertEqual(attack.stage_runtimes[0].status, RuntimeStatus.FAIL)
        self.assertEqual(service.output(), (1, 2))

    def test_overlapping_attacks_can_stop_in_either_order(self):
        for order in ((0, 1), (1, 0)):
            service = Service()
            attacks = [self.make_attack(), self.make_attack()]
            for attack in attacks:
                self.run_attack(attack, [service])
            attacks[order[0]].mark_stopped()
            self.assertEqual(service.output(), ())
            attacks[order[1]].mark_stopped()
            self.assertEqual(service.output(), (1, 2))

    def test_persistent_installation_allows_next_stage_to_start(self):
        stages = (DropperStage((Capability.RESPONSE_SUBMIT,)), DropperStage((Capability.RESPONSE_SUBMIT,)))
        spec = AttackSpec(name="sequence", stages=(StageSpec(id="a", type="dropper"), StageSpec(id="b", type="dropper")))
        attack = Attack(spec, stages)
        self.addCleanup(attack.deactivate)
        attack.mark_started()
        self.assertEqual(len(self.run_attack(attack, [Service()])), 2)
        self.assertTrue(all(runtime.status == RuntimeStatus.ACTIVE for runtime in attack.stage_runtimes))

    def test_failure_result_cleans_preceding_stage(self):
        stages = (DropperStage((Capability.RESPONSE_SUBMIT,)), DropperStage((Capability.RESPONSE_SUBMIT,)))
        spec = AttackSpec(name="failure", stages=(StageSpec(id="a", type="dropper"), StageSpec(id="b", type="dropper")))
        attack = Attack(spec, stages)
        self.addCleanup(attack.deactivate)
        attack.mark_started()
        service = Service()
        with patch.object(stages[1], "execute", return_value=AttackStageResult(stage_name="dropper", status=Status.FAIL)):
            self.run_attack(attack, [service])
        self.assertEqual(attack.status, RuntimeStatus.FAIL)
        self.assertEqual(service.output(), (1, 2))

    def test_restart_installs_fresh_interceptors(self):
        service = Service()
        attack = self.make_attack()
        self.run_attack(attack, [service])
        attack.mark_failed()
        attack.mark_started()
        self.run_attack(attack, [service])
        self.assertEqual(service.output(), ())
        self.assertEqual(attack.stage_runtimes[0].status, RuntimeStatus.ACTIVE)
        attack.mark_stopped()
        self.assertEqual(service.output(), (1, 2))

    def test_one_shot_stage_completes_and_cleans_up(self):
        attack = self.make_attack()
        attack.stages[0].persistent_effect = False
        service = Service()
        self.run_attack(attack, [service])
        self.assertEqual(attack.status, RuntimeStatus.SUCCESS)
        self.assertEqual(service.output(), (1, 2))

    def test_stop_condition_error_cleans_up(self):
        condition = ConditionSpec(source=TriggerSourceSpec(kind="snapshot"), verb="exists")
        attack = self.make_attack(condition)
        service = Service()
        self.run_attack(attack, [service])
        with patch.object(attack, "_evaluate_condition", side_effect=ValueError("bad condition")):
            with self.assertRaises(ValueError):
                self.run_attack(attack, [service])
        self.assertEqual(attack.status, RuntimeStatus.FAIL)
        self.assertEqual(service.output(), (1, 2))


if __name__ == "__main__":
    unittest.main()
