from __future__ import annotations

from typing import TYPE_CHECKING

from ...logging import get_logger
from ...scheduler import SquashResult, ThreadScheduler
from ..types import TelegramIncomingMessage
from .cancel import _edit_labelled_message
from .reply import make_reply

if TYPE_CHECKING:
    from ..bridge import TelegramBridgeConfig

logger = get_logger(__name__)

_AMBIGUOUS = "multiple queued threads here — reply to a queued message to pick one."
_ALREADY_COMBINING = "already in squash mode — queued messages combine automatically."


async def handle_squash(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    scheduler: ThreadScheduler | None = None,
) -> None:
    reply = make_reply(cfg, msg)
    if scheduler is None:
        await reply(text="nothing queued to squash.")
        return
    if cfg.queue.combine:
        await reply(text=_ALREADY_COMBINING)
        return

    reply_id = msg.reply_to_message_id
    if reply_id is not None:
        result = await scheduler.squash_queued_by_progress(msg.chat_id, reply_id)
    else:
        tokens = await scheduler.queued_threads_for_chat(msg.chat_id, msg.thread_id)
        if len(tokens) > 1:
            await reply(text=_AMBIGUOUS)
            return
        result = await scheduler.squash_queued(tokens[0]) if tokens else None

    if result is None:
        await reply(text="nothing queued to squash.")
        return

    logger.info(
        "squash.queued",
        chat_id=msg.chat_id,
        resume=result.job.resume_token.value,
        merged_count=result.merged_count,
    )
    await edit_squash_result(cfg, result)


async def edit_squash_result(
    cfg: TelegramBridgeConfig,
    result: SquashResult,
) -> None:
    if result.job.progress_ref is not None:
        await _edit_labelled_message(
            cfg, result.job.progress_ref, result.job, label="queued, combined"
        )
    for dropped_ref in result.dropped_progress_refs:
        await _edit_labelled_message(cfg, dropped_ref, result.job, label="merged")
