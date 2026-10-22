"""Job handlers; importing this package registers every kind with `tasks.HANDLERS`."""

from . import inference, monitoring, release, replay, shadow, snapshot, training  # noqa: F401
