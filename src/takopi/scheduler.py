from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Protocol
from collections.abc import Awaitable, Callable

import anyio

from .context import RunContext
from .logging import get_logger
from .model import ResumeToken
from .transport import ChannelId, MessageId, MessageRef, ThreadId

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ThreadJob:
    chat_id: ChannelId
    user_msg_id: MessageId
    text: str
    resume_token: ResumeToken
    context: RunContext | None = None
    thread_id: ThreadId | None = None
    session_key: tuple[int, int | None] | None = None
    progress_ref: MessageRef | None = None


@dataclass(frozen=True, slots=True)
class SquashResult:
    job: ThreadJob
    dropped_progress_refs: list[MessageRef]
    merged_count: int


RunJob = Callable[[ThreadJob], Awaitable[None]]


class TaskGroup(Protocol):
    def start_soon(
        self, func: Callable[..., Awaitable[object]], *args: Any
    ) -> None: ...


class ThreadScheduler:
    def __init__(self, *, task_group: TaskGroup, run_job: RunJob) -> None:
        self._task_group = task_group
        self._run_job = run_job
        self._lock = anyio.Lock()
        self._pending_by_thread: dict[str, deque[ThreadJob]] = {}
        self._queued_by_progress: dict[tuple[ChannelId, MessageId], ThreadJob] = {}
        self._active_threads: set[str] = set()
        self._busy_until: dict[str, anyio.Event] = {}

    @staticmethod
    def thread_key(token: ResumeToken) -> str:
        return f"{token.engine}:{token.value}"

    async def note_thread_known(self, token: ResumeToken, done: anyio.Event) -> None:
        key = self.thread_key(token)
        async with self._lock:
            current = self._busy_until.get(key)
            if current is None or current.is_set():
                self._busy_until[key] = done
        self._task_group.start_soon(self._clear_busy, key, done)

    async def enqueue(
        self, job: ThreadJob, *, combine: bool = False
    ) -> SquashResult | None:
        key = self.thread_key(job.resume_token)
        squash_result: SquashResult | None = None
        start_worker = False
        async with self._lock:
            queue = self._pending_by_thread.get(key)
            if queue is None:
                queue = deque()
                self._pending_by_thread[key] = queue
            queue.append(job)
            if job.progress_ref is not None:
                progress_key = (job.chat_id, job.progress_ref.message_id)
                self._queued_by_progress[progress_key] = job
            if combine and len(queue) > 1:
                squash_result = self._squash_queued_locked(job.resume_token)
            if key in self._active_threads:
                return squash_result
            self._active_threads.add(key)
            start_worker = True
        if start_worker:
            self._task_group.start_soon(self._thread_worker, key)
        return squash_result

    async def enqueue_resume(
        self,
        chat_id: ChannelId,
        user_msg_id: MessageId,
        text: str,
        resume_token: ResumeToken,
        context: RunContext | None = None,
        thread_id: ThreadId | None = None,
        session_key: tuple[int, int | None] | None = None,
        progress_ref: MessageRef | None = None,
        combine: bool = False,
    ) -> SquashResult | None:
        return await self.enqueue(
            ThreadJob(
                chat_id=chat_id,
                user_msg_id=user_msg_id,
                text=text,
                resume_token=resume_token,
                context=context,
                thread_id=thread_id,
                session_key=session_key,
                progress_ref=progress_ref,
            ),
            combine=combine,
        )

    async def cancel_queued(
        self, chat_id: ChannelId, progress_msg_id: MessageId
    ) -> ThreadJob | None:
        async with self._lock:
            return self._pop_queued_locked(chat_id, progress_msg_id)

    async def claim_queued(
        self, chat_id: ChannelId, progress_msg_id: MessageId
    ) -> ThreadJob | None:
        async with self._lock:
            return self._pop_queued_locked(chat_id, progress_msg_id)

    async def squash_queued(
        self,
        resume_token: ResumeToken,
        *,
        separator: str = "\n\n",
    ) -> SquashResult | None:
        async with self._lock:
            return self._squash_queued_locked(resume_token, separator=separator)

    async def squash_queued_by_progress(
        self,
        chat_id: ChannelId,
        progress_msg_id: MessageId,
        *,
        separator: str = "\n\n",
    ) -> SquashResult | None:
        async with self._lock:
            job = self._queued_by_progress.get((chat_id, progress_msg_id))
            if job is None:
                return None
            return self._squash_queued_locked(job.resume_token, separator=separator)

    async def queued_threads_for_chat(
        self, chat_id: ChannelId, thread_id: ThreadId | None
    ) -> list[ResumeToken]:
        async with self._lock:
            tokens: list[ResumeToken] = []
            for queue in self._pending_by_thread.values():
                if not queue:
                    continue
                head = queue[0]
                if head.chat_id == chat_id and head.thread_id == thread_id:
                    tokens.append(head.resume_token)
            return tokens

    async def requeue_front(self, job: ThreadJob) -> None:
        key = self.thread_key(job.resume_token)
        async with self._lock:
            queue = self._pending_by_thread.get(key)
            if queue is None:
                queue = deque()
                self._pending_by_thread[key] = queue
            queue.appendleft(job)
            if job.progress_ref is not None:
                progress_key = (job.chat_id, job.progress_ref.message_id)
                self._queued_by_progress[progress_key] = job
            if key in self._active_threads:
                return
            self._active_threads.add(key)
        self._task_group.start_soon(self._thread_worker, key)

    async def get_queued(
        self, chat_id: ChannelId, progress_msg_id: MessageId
    ) -> ThreadJob | None:
        progress_key = (chat_id, progress_msg_id)
        async with self._lock:
            return self._queued_by_progress.get(progress_key)

    async def is_busy(self, token: ResumeToken) -> bool:
        key = self.thread_key(token)
        async with self._lock:
            done = self._busy_until.get(key)
            return done is not None and not done.is_set()

    def _pop_queued_locked(
        self, chat_id: ChannelId, progress_msg_id: MessageId
    ) -> ThreadJob | None:
        progress_key = (chat_id, progress_msg_id)
        job = self._queued_by_progress.get(progress_key)
        if job is None:
            return None
        thread_key = self.thread_key(job.resume_token)
        queue = self._pending_by_thread.get(thread_key)
        if queue is None:
            return None
        try:
            queue.remove(job)
        except ValueError:
            return None
        self._queued_by_progress.pop(progress_key, None)
        if not queue:
            self._pending_by_thread.pop(thread_key, None)
        return job

    def _squash_queued_locked(
        self,
        resume_token: ResumeToken,
        *,
        separator: str = "\n\n",
    ) -> SquashResult | None:
        key = self.thread_key(resume_token)
        queue = self._pending_by_thread.get(key)
        if not queue:
            return None
        jobs = list(queue)
        texts = [job.text for job in jobs if job.text.strip()]
        merged = replace(jobs[0], text=separator.join(texts))
        dropped: list[MessageRef] = []
        for job in jobs:
            if job.progress_ref is not None:
                self._queued_by_progress.pop(
                    (job.chat_id, job.progress_ref.message_id), None
                )
                if job is not jobs[0]:
                    dropped.append(job.progress_ref)
        queue.clear()
        queue.append(merged)
        if merged.progress_ref is not None:
            self._queued_by_progress[
                (merged.chat_id, merged.progress_ref.message_id)
            ] = merged
        return SquashResult(
            job=merged,
            dropped_progress_refs=dropped,
            merged_count=len(jobs),
        )

    async def _clear_busy(self, key: str, done: anyio.Event) -> None:
        await done.wait()
        async with self._lock:
            if self._busy_until.get(key) is done:
                self._busy_until.pop(key, None)

    async def _thread_worker(self, key: str) -> None:
        try:
            while True:
                async with self._lock:
                    done = self._busy_until.get(key)
                    queue = self._pending_by_thread.get(key)
                    if not queue:
                        self._pending_by_thread.pop(key, None)
                        self._active_threads.discard(key)
                        return

                if done is not None and not done.is_set():
                    await done.wait()
                    continue

                async with self._lock:
                    queue = self._pending_by_thread.get(key)
                    if not queue:
                        continue
                    job = queue.popleft()
                    if job.progress_ref is not None:
                        progress_key = (job.chat_id, job.progress_ref.message_id)
                        self._queued_by_progress.pop(progress_key, None)

                try:
                    await self._run_job(job)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "scheduler.job_failed",
                        key=key,
                        tag=job.resume_token.engine,
                        chat_id=job.chat_id,
                        user_msg_id=job.user_msg_id,
                        error=str(exc),
                        error_type=exc.__class__.__name__,
                    )
        finally:
            async with self._lock:
                self._active_threads.discard(key)
