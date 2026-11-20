# Service interceptor lifecycle

All built-in stages install their rewrites through a shared interceptor manager.
A service method has one dispatcher, which applies active rewrites in installation
order. Each registration has an idempotent removal callback. Removing a stage
never removes another stage or restores a previously removed rewrite. After the
last removal, the original instance attribute (or class method lookup) is restored.
Handlers replaced by unrelated code are not overwritten during cleanup.

## Stage lifetime

- Spoofer, dropper, replayer and sniffer declare `persistent_effect = True`.
  A successful `execute()` means installation succeeded; the stage stays ACTIVE.
- A stage stop trigger removes the stage's interceptors and marks it SUCCESS.
- Without a stage stop trigger, a persistent stage lasts until the attack stops.
  Its successful installation allows the next implicit sequential stage to start.
  Explicit status conditions still see ACTIVE, not SUCCESS, during this lifetime.
- One-shot custom stages (without `persistent_effect`) complete after successful
  execution unless they have a stage stop trigger. Their cleanup runs immediately.
- Attack success, failure, cancellation, reset and scenario shutdown deactivate all
  stages. Cleanup attempts every stage even if a custom cleanup raises; errors are
  logged. Stage execution errors also trigger cleanup before being re-raised.
- Restarting an attack resets stage states and results. Replay buffers are cleared
  by deactivation; sniffer observations remain available for inspection.

A persistent stage without either a stage or attack stop trigger intentionally
lasts until scenario shutdown. Give it a stop condition when finite completion is
required. The manager is designed for the synchronous simulation loop; it does
not introduce thread synchronization or message scheduling.

## Validation

`test/test_attack_lifecycle.py` covers removal order, overlapping attacks,
instance overrides, persistent and sequential stages, terminal cleanup,
partial-installation failures and restart. The scenario finalization tests check
that attack cleanup precedes service destruction and continues after errors.
