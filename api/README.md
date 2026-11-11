# OpenCDA Runner API

OpenCDA is the only externally registered Runner on a prepared Linux node.
CARLA and SUMO APIs are internal adapters. Run one API worker per installation;
`python -m api` enforces one Uvicorn worker without reload. Install
`python -m pip install -r api/requirements.txt` from the OpenCDA repository.

## Enrollment

Create a pending registration with Manager's administrative credential. Configure
these variables in the OpenCDA container before `python -m api`:

| Variable | Meaning |
| --- | --- |
| `CAVISE_RUNNER_STATE_DIR` | Persistent private directory; default `.runner-state`. |
| `CAVISE_MANAGER_URL` | Manager base URL, reachable from this container. |
| `CAVISE_REGISTRATION_ID` | UUID returned by Manager, not the installation ID. |
| `CAVISE_REGISTRATION_TOKEN_FILE` | Protected file containing the registration token. |
| `CAVISE_REGISTRATION_TOKEN` | Alternative initial token if no file is configured. |
| `CAVISE_RUNNER_API_URL` | Direct advertised base URL reachable from Manager. |
| `CAVISE_CARLA_API_URL` | Internal adapter URL; default `http://carla:8080`. |
| `CAVISE_SUMO_API_URL` | Internal adapter URL; default `http://sumo:8080`. |

Do not install the Manager administrative token on Runner. State directory mode is
0700; SQLite mode is 0600. Mount the same directory across container replacement.
Do not clone an installation's state to a different node. Enrollment is persisted
before the asynchronous confirmation request. The API serves reverse authenticated
status while enrollment retries transient errors with bounded backoff. Rejected
credentials/configuration require correction and API restart; they do not stop a
scenario. Redirects are not followed.

With no registration variables, startup reads the existing binding from SQLite.
With a registration UUID configured, URLs must be supplied; its stored token can
be reused if the initial secret is omitted. A conflicting binding is rejected.
After confirmed removal, supply a new Manager UUID/token and restart. Installation
identity and retained task results survive; the API instance UUID changes.

## HTTP contract

Every route except `/healthz` requires `Authorization: Bearer <registration-token>`.
This includes `/v1/run`, configuration uploads, `/v1/stop`, `/docs`, and
`/openapi.json`. Native OpenAPI is also checked in as `api/openapi.yaml`.
Errors use `code` and `message`; missing task results return 404.

- `GET /v1/status`: complete registration/installation/instance identity, revision,
  version, capabilities, readiness, occupancy and nullable current task.
- `GET /v1/tasks/{task_id}`: retained task identity and state.
- `POST /v1/run?scenario=demo`: accept one scenario, returning 202 and its task ID.
  Alternatively send multipart `config`; its YAML stem still selects SUMO assets.
- `POST /v1/stop`: request cancellation and wait up to 30 seconds for confirmed
  completion/cleanup. Timeout leaves execution occupied and can be retried.
- `DELETE /v1/registration`: atomically reject busy/uncertain execution or revoke
  the idle binding. It never stops a task. Revocation immediately disables ordinary authenticated requests.

Retired token hashes persist as tombstones. Only repeated unbinding accepts a
retired token with 204. It cannot revoke a subsequent registration or read data.

## On-demand status

There is no event stream or heartbeat. Manager verifies status at registration
and before removal; it does not continuously monitor registered Runners. Runner
captures a fresh snapshot only when requested. Revision changes when the returned
snapshot changes within the same API instance.

## Execution and recovery

The API reserves a task in SQLite before launching `api.supervisor` in an
independent process session. The supervisor owns the scenario process group,
records cancellation intent and persists results even if the API dies. API
shutdown cancels enrollment retries, not execution. Container shutdown still terminates
container processes: a later startup reports uncertainty unless a result was
persisted. Use the deployment's init process to reap orphaned supervisors.

`PREPARING` lasts until the first successful simulator tick. `RUNNING` is reported
by the scenario at that point. A task becomes terminal only after its process
exits and both configured adapters confirm cleanup. The scenario durably records each adapter before attempting its start. After
scenario exit, the supervisor retries stop requests only for those recorded
adapters and releases each responsibility after confirmed idle status. Missing
adapters, surviving components or ambiguous cleanup keep occupancy unknown and
block new work.
Terminal outcomes and release of occupancy are one SQLite transaction.

Recovery verifies the supervisor's Linux boot ID and process start ticks, not
just PID. If ownership cannot be verified, the task remains `UNKNOWN`; it is
never automatically restarted or marked successful. This phase has no force-reset
HTTP operation. Preserve the database for investigation when manual recovery is
required. Stop all relevant containers and verify component cleanup before an
operator repairs unresolved local state; never delete credentials to bypass a
busy check. Back up the private state only with API and supervisors stopped.

## Validation

From this repository, with test dependencies installed:

```sh
PYTHONPATH=. python -m pytest api/tests -q
ruff check api
ruff format --check api
mypy --follow-imports=silent api --exclude tests
```

Tests use fake adapters/processes and a real independent recorder with a harmless
Python child. They do not validate GPU simulations or Docker image startup.

To run the real Go/Python protocol check without simulation, build Manager and set
`CAVISE_MANAGER_TEST_BINARY=/absolute/path/to/cavise-manager` when running pytest.
It checks enrollment, both process restarts and credential revocation.
