"""Local bridge between the Python agent and the Unreal research environment.

* :mod:`.protocol`: binary framing, message schema and validation.
* :mod:`.client`: the localhost-only TCP client used by the agent and runner.
* :mod:`.observation_adapter`: decodes observations into NumPy frames plus
  camera metadata.
* :mod:`.action_adapter`: turns controller intents into validated actions.
* :mod:`.mock_server`: a protocol-identical server backed by the Python
  reference arena (:mod:`chest_hunter.simulation3d`), for development and
  tests where Unreal Engine is not installed.

Only loopback connections are made or accepted. There is no remote console,
no arbitrary command execution and no connection to any game service.
"""
