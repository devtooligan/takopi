from __future__ import annotations

from typing import Any

import anyio
import pytest

from takopi.model import ResumeToken
from takopi.scheduler import ThreadScheduler
from takopi.transport import MessageRef

CODEX_ENGINE = "codex"


class _NoopTaskGroup:
    def start_soon(self, func, *args: Any) -> None:
        _ = func, args
        return None


async def _make_scheduler() -> ThreadScheduler:
    async def _noop_run_job(_) -> None:
        return None

    return ThreadScheduler(task_group=_NoopTaskGroup(), run_job=_noop_run_job)


async def _enqueue(
    scheduler: ThreadScheduler,
    resume: ResumeToken,
    *,
    user_msg_id: int,
    text: str,
    progress_id: int,
) -> MessageRef:
    ref = MessageRef(channel_id=123, message_id=progress_id)
    await scheduler.enqueue_resume(
        chat_id=123,
        user_msg_id=user_msg_id,
        text=text,
        resume_token=resume,
        progress_ref=ref,
    )
    return ref


@pytest.mark.anyio
async def test_squash_merges_all_queued_into_one() -> None:
    scheduler = await _make_scheduler()
    resume = ResumeToken(engine=CODEX_ENGINE, value="sid")
    first = await _enqueue(scheduler, resume, user_msg_id=1, text="one", progress_id=51)
    second = await _enqueue(
        scheduler, resume, user_msg_id=2, text="two", progress_id=52
    )
    third = await _enqueue(
        scheduler, resume, user_msg_id=3, text="three", progress_id=53
    )

    result = await scheduler.squash_queued(resume)

    assert result is not None
    assert result.merged_count == 3
    assert result.job.text == "one\n\ntwo\n\nthree"
    assert result.job.user_msg_id == 1
    assert result.job.progress_ref == first
    assert result.dropped_progress_refs == [second, third]
    assert await scheduler.get_queued(123, first.message_id) is result.job
    assert await scheduler.get_queued(123, second.message_id) is None
    assert await scheduler.get_queued(123, third.message_id) is None


@pytest.mark.anyio
async def test_squash_empty_queue_returns_none() -> None:
    scheduler = await _make_scheduler()
    resume = ResumeToken(engine=CODEX_ENGINE, value="sid")

    assert await scheduler.squash_queued(resume) is None


@pytest.mark.anyio
async def test_squash_ignores_blank_text_jobs() -> None:
    scheduler = await _make_scheduler()
    resume = ResumeToken(engine=CODEX_ENGINE, value="sid")
    await _enqueue(scheduler, resume, user_msg_id=1, text="", progress_id=51)
    await _enqueue(scheduler, resume, user_msg_id=2, text="real", progress_id=52)

    result = await scheduler.squash_queued(resume)

    assert result is not None
    assert result.job.text == "real"


@pytest.mark.anyio
async def test_queued_threads_for_chat_resolves_current_thread() -> None:
    scheduler = await _make_scheduler()
    resume = ResumeToken(engine=CODEX_ENGINE, value="sid")
    await _enqueue(scheduler, resume, user_msg_id=1, text="one", progress_id=51)

    tokens = await scheduler.queued_threads_for_chat(123, None)
    assert tokens == [resume]
    assert await scheduler.queued_threads_for_chat(999, None) == []


@pytest.mark.anyio
async def test_squashed_job_is_what_runs() -> None:
    resume = ResumeToken(engine=CODEX_ENGINE, value="sid")
    active_done = anyio.Event()
    ran: list[str] = []

    async def _run_job(job) -> None:
        ran.append(job.text)

    async with anyio.create_task_group() as tg:
        scheduler = ThreadScheduler(task_group=tg, run_job=_run_job)
        # Hold the thread busy so the three follow-ups queue instead of running.
        await scheduler.note_thread_known(resume, active_done)
        for i in (1, 2, 3):
            await _enqueue(
                scheduler, resume, user_msg_id=i, text=f"m{i}", progress_id=50 + i
            )

        result = await scheduler.squash_queued(resume)
        assert result is not None

        # Release the thread; the worker should run exactly the merged job.
        active_done.set()

    assert ran == ["m1\n\nm2\n\nm3"]
