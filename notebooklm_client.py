"""NotebookLM client — generates audio briefings via notebooklm-py API.

No Chrome required. Uses stored Google auth cookies.
"""

import asyncio
import logging
import os
import time
from typing import List, Optional

from notebooklm.client import NotebookLMClient
from notebooklm.auth import AuthTokens
from notebooklm.rpc.types import StudioContentType

logger = logging.getLogger(__name__)

PROMPT_HE = (
    "צרו פודקאסט בעברית בין מנחה ואנליסט מומחה. "
    "המנחה היא אישה שמארחת אנליסט גבר. "
    "היא שואלת שאלות חכמות ומקשה, והוא עונה בעומק עם דוגמאות. "
    "הטון טבעי וזורם, לא אקדמי. "
    "דגש על חדשות, סנטימנט שוק, ודעות אנליסטים. ניתוח טכני קצר ולעניין."
)


async def _generate_podcast_async(
    notebook_id: str,
    text_source: str,
    prompt: str = PROMPT_HE,
    timeout_minutes: int = 20,
    on_step=None,
) -> Optional[bytes]:
    """Async implementation of the full pipeline."""
    _step = on_step or (lambda s: None)

    _step("connect")
    logger.info("Authenticating with NotebookLM API...")
    auth = await AuthTokens.from_storage()

    async with NotebookLMClient(auth) as client:

        # Clear old sources
        _step("clear")
        logger.info("Clearing old sources...")
        sources = await client.sources.list(notebook_id)
        for src in sources:
            try:
                await client.sources.delete(notebook_id, src.id)
                logger.info(f"  Deleted source: {src.title}")
            except Exception as e:
                logger.warning(f"  Could not delete source {src.id}: {e}")

        # Add new text source
        _step("upload_text")
        logger.info("Adding text source...")
        await client.sources.add_text(
            notebook_id,
            title="Portfolio Briefing",
            content=text_source,
            wait=True,
            wait_timeout=120,
        )
        logger.info("Text source added.")

        # Snapshot existing artifact IDs BEFORE triggering generation
        existing_ids = {a.id for a in await client.artifacts.list(notebook_id)
                        if a.artifact_type == StudioContentType.AUDIO.value}
        logger.info(f"Existing audio artifacts before generation: {len(existing_ids)}")

        # Trigger audio generation
        _step("generate")
        logger.info("Starting audio generation...")
        status = await client.artifacts.generate_audio(
            notebook_id,
            language="he",
            instructions=prompt,
        )
        if status.is_failed:
            if status.is_rate_limited:
                raise RuntimeError(
                    "NotebookLM daily audio quota exceeded. "
                    "Audio overview generation is limited per day on free accounts. "
                    "Try again tomorrow."
                )
            raise RuntimeError(f"Audio generation failed: {status.error} (code={status.error_code})")

        task_id = status.task_id
        logger.info(f"Generation started — task_id: {task_id}")

        # Poll until ready
        _step("waiting")
        max_wait = timeout_minutes * 60
        start = time.time()
        artifact_id = None

        while time.time() - start < max_wait:
            try:
                artifacts = await client.artifacts.list(notebook_id)
                audio = [a for a in artifacts if a.artifact_type == StudioContentType.AUDIO.value]

                # NEW artifact = one that didn't exist before
                new_artifacts = [a for a in audio if a.id not in existing_ids]
                new_ready = [a for a in new_artifacts if a.status == 3]
                new_generating = [a for a in new_artifacts if a.status == 1]

                elapsed = int(time.time() - start)
                logger.info(f"[{elapsed}s] new: generating={len(new_generating)}, ready={len(new_ready)}")

                if new_ready:
                    artifact_id = new_ready[0].id
                    logger.info(f"Audio ready: {new_ready[0].title}")
                    break

                await asyncio.sleep(15)

            except Exception as e:
                logger.warning(f"Polling error (retrying): {e}")
                await asyncio.sleep(15)

        if not artifact_id:
            raise RuntimeError("Audio generation timed out or failed")

        # Download
        _step("download")
        logger.info(f"Downloading artifact {artifact_id}...")

        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".m4a")
        try:
            os.close(tmp_fd)
            await client.artifacts.download_audio(notebook_id, tmp_path, artifact_id=artifact_id)
            with open(tmp_path, "rb") as f:
                content = f.read()
            logger.info(f"Downloaded {len(content):,} bytes")
            return content
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def generate_podcast(
    notebook_url: str,
    text_source: str,
    url_sources: List[str],
    port: int = 9222,
    prompt: str = PROMPT_HE,
    timeout_minutes: int = 20,
    on_step=None,
) -> Optional[bytes]:
    """Generate a podcast via NotebookLM API (no Chrome required).

    Args:
        notebook_url: Full NotebookLM notebook URL (notebook ID is extracted).
        text_source: Hebrew briefing text to upload as source.
        url_sources: Ignored — NotebookLM searches the web on its own.
        port: Ignored — no Chrome used.
        prompt: Custom instructions for the podcast hosts.
        timeout_minutes: Max wait for generation to complete.
        on_step: Optional callback called with step name strings.

    Returns:
        Audio bytes (m4a) or None on failure.
    """
    del url_sources, port  # kept in signature for API compatibility; not used
    # Extract notebook ID from URL
    notebook_id = notebook_url.rstrip("/").split("/")[-1].split("?")[0]
    logger.info(f"Notebook ID: {notebook_id}")

    return asyncio.run(
        _generate_podcast_async(
            notebook_id=notebook_id,
            text_source=text_source,
            prompt=prompt,
            timeout_minutes=timeout_minutes,
            on_step=on_step,
        )
    )
