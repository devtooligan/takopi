from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from collections.abc import Awaitable, Callable, Hashable

import anyio

from ..logging import get_logger
from .client_api import RetryAfter

logger = get_logger(__name__)

SEND_PRIORITY = 0
DELETE_PRIORITY = 1
EDIT_PRIORITY = 2

# Upper bound on how long a queued op keeps retrying on transient failures
# before it is dropped with a None result. Without this, a sustained Telegram
# outage makes the outbox requeue an op forever, blocking every caller awaiting
# a send/edit/delete (e.g. the initial progress message before a runner starts).
_OP_RETRY_DEADLINE_S = 60.0


@dataclass(slots=True)
class OutboxOp:
    execute: Callable[[], Awaitable[Any]]
    priority: int
    queued_at: float
    chat_id: int | None
    label: str | None = None
    done: anyio.Event = field(default_factory=anyio.Event)
    result: Any = None

    def set_result(self, result: Any) -> None:
        if self.done.is_set():
            return
        self.result = result
        self.done.set()


class TelegramOutbox:
    def __init__(
        self,
        *,
        interval_for_chat: Callable[[int | None], float],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
        on_error: Callable[[OutboxOp, Exception], None] | None = None,
        on_outbox_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._interval_for_chat = interval_for_chat
        self._clock = clock
        self._sleep = sleep
        self._on_error = on_error
        self._on_outbox_error = on_outbox_error
        self._pending: dict[Hashable, OutboxOp] = {}
        self._cond = anyio.Condition()
        self._start_lock = anyio.Lock()
        self._closed = False
        self._tg: TaskGroup | None = None
        self.next_at = 0.0
        self.retry_at = 0.0

    async def ensure_worker(self) -> None:
        async with self._start_lock:
            if self._tg is not None or self._closed:
                return
            self._tg = await anyio.create_task_group().__aenter__()
            self._tg.start_soon(self.run)

    async def enqueue(self, *, key: Hashable, op: OutboxOp, wait: bool = True) -> Any:
        await self.ensure_worker()
        async with self._cond:
            if self._closed:
                op.set_result(None)
                return op.result
            previous = self._pending.get(key)
            if previous is not None:
                op.queued_at = previous.queued_at
                previous.set_result(None)
            self._pending[key] = op
            self._cond.notify()
        if not wait:
            return None
        await op.done.wait()
        return op.result

    async def drop_pending(self, *, key: Hashable) -> None:
        async with self._cond:
            pending = self._pending.pop(key, None)
            if pending is not None:
                pending.set_result(None)
            self._cond.notify()

    async def close(self) -> None:
        async with self._cond:
            self._closed = True
            self.fail_pending()
            self._cond.notify_all()
        if self._tg is not None:
            await self._tg.__aexit__(None, None, None)
            self._tg = None

    def fail_pending(self) -> None:
        for pending in list(self._pending.values()):
            pending.set_result(None)
        self._pending.clear()

    def pick_locked(self) -> tuple[Hashable, OutboxOp] | None:
        if not self._pending:
            return None
        return min(
            self._pending.items(),
            key=lambda item: (item[1].priority, item[1].queued_at),
        )

    async def execute_op(self, op: OutboxOp) -> Any:
        try:
            return await op.execute()
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, RetryAfter):
                raise
            if self._on_error is not None:
                self._on_error(op, exc)
            return None

    async def sleep_until(self, deadline: float) -> None:
        delay = deadline - self._clock()
        if delay > 0:
            await self._sleep(delay)

    async def run(self) -> None:
        cancel_exc = anyio.get_cancelled_exc_class()
        try:
            while True:
                async with self._cond:
                    while not self._pending and not self._closed:
                        await self._cond.wait()
                    if self._closed and not self._pending:
                        return
                blocked_until = max(self.next_at, self.retry_at)
                if self._clock() < blocked_until:
                    await self.sleep_until(blocked_until)
                    continue
                async with self._cond:
                    if self._closed and not self._pending:
                        return
                    picked = self.pick_locked()
                    if picked is None:
                        continue
                    key, op = picked
                    self._pending.pop(key, None)
                started_at = self._clock()
                try:
                    result = await self.execute_op(op)
                except RetryAfter as exc:
                    self.retry_at = max(self.retry_at, self._clock() + exc.retry_after)
                    # Give up if the next retry would fall past the op's deadline,
                    # so a caller awaiting the result is never blocked much beyond
                    # it. A large server-supplied retry_after (flood-wait) must not
                    # stall the pipeline waiting for a send that will be dropped.
                    expired = self.retry_at - op.queued_at >= _OP_RETRY_DEADLINE_S
                    async with self._cond:
                        if self._closed or expired:
                            op.set_result(None)
                        elif key not in self._pending:
                            self._pending[key] = op
                            self._cond.notify()
                        else:
                            op.set_result(None)
                    if expired and not self._closed:
                        logger.warning(
                            "telegram.outbox.giving_up",
                            method=op.label,
                            retry_after=exc.retry_after,
                            error=exc.description,
                        )
                    continue
                self.next_at = started_at + self._interval_for_chat(op.chat_id)
                op.set_result(result)
        except cancel_exc:
            return
        except Exception as exc:  # noqa: BLE001
            async with self._cond:
                self._closed = True
                self.fail_pending()
                self._cond.notify_all()
            if self._on_outbox_error is not None:
                self._on_outbox_error(exc)
            return


if TYPE_CHECKING:
    from anyio.abc import TaskGroup
else:
    TaskGroup = object
