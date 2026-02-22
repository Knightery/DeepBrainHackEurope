from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import chainlit as cl


def _is_finite_number(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _parse_equity_curve_points(raw_curve: Any) -> list[tuple[str, float]]:
    if not isinstance(raw_curve, list):
        return []
    points: list[tuple[str, float]] = []
    for item in raw_curve:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        date = str(item[0]).strip()
        if not date:
            continue
        try:
            value = float(item[1])
        except (TypeError, ValueError):
            continue
        if not _is_finite_number(value):
            continue
        points.append((date, value))
    return points


def _build_equity_curve_figure(metrics: dict[str, Any]) -> Any | None:
    strategy_points = _parse_equity_curve_points(metrics.get("equity_curve"))
    if len(strategy_points) < 2:
        return None

    benchmark_points = _parse_equity_curve_points(metrics.get("benchmark_curve"))

    import plotly.graph_objects as go  # noqa: PLC0415

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[pt[0] for pt in strategy_points],
            y=[pt[1] for pt in strategy_points],
            name="Strategy",
            mode="lines",
            line=dict(color="#00e676", width=2.25),
            fill="tozeroy",
            fillcolor="rgba(0,230,118,0.08)",
            hovertemplate="%{x}<br>Strategy: %{y:.3f}<extra></extra>",
        )
    )

    if len(benchmark_points) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[pt[0] for pt in benchmark_points],
                y=[pt[1] for pt in benchmark_points],
                name="Benchmark (B&H)",
                mode="lines",
                line=dict(color="#9e9e9e", width=1.6, dash="dot"),
                hovertemplate="%{x}<br>Benchmark: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Equity Curve (normalized, start = 1.0)",
        template="plotly_dark",
        height=420,
        margin=dict(l=40, r=20, t=55, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Portfolio Value", tickformat=".2f")
    return fig


async def send_equity_curve_chart(metrics: dict[str, Any], logger: Any) -> bool:
    try:
        fig = _build_equity_curve_figure(metrics)
        if fig is None:
            return False
        await cl.Message(
            content="Pinned equity curve chart on the right panel.",
            author="Backtest Agent",
            elements=[cl.Plotly(name="equity_curve", figure=fig, display="side")],
        ).send()
        return True
    except Exception as exc:
        logger.debug("Equity curve chart failed: %s", exc)
        return False


class CuaLiveStreamer:
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        logger: Any,
        author: str,
        wait_label: str = "[CUA] Waiting for container output...",
        update_interval_seconds: float = 0.7,
        heartbeat_interval_seconds: float = 2.0,
        max_lines: int = 0,
        max_buffer_lines: int = 2000,
        error_tag: str = "CUA live flush failed",
    ) -> None:
        self._loop = loop
        self._logger = logger
        self._author = author
        self._wait_label = wait_label
        self._update_interval = update_interval_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._max_lines = max_lines
        self._max_buffer_lines = max(200, max_buffer_lines)
        self._error_tag = error_tag

        self._lines: list[str] = []
        self._dropped_line_count = 0
        self._lines_lock = threading.Lock()
        self._message: cl.Message | None = None
        self._last_update = 0.0
        self._last_activity = time.time()
        self._started = time.time()
        self._stop_event = asyncio.Event()
        self._flush_lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def _flush(self) -> None:
        async with self._flush_lock:
            with self._lines_lock:
                snapshot = list(self._lines[-self._max_lines :]) if self._max_lines > 0 else list(self._lines)
                dropped = self._dropped_line_count
            body = "\n".join(snapshot) if snapshot else self._wait_label
            if dropped > 0:
                body = f"[CUA] Earlier logs omitted: {dropped} line(s)\n" + body
            idle_secs = int(max(0.0, time.time() - self._last_activity))
            elapsed_secs = int(max(0.0, time.time() - self._started))
            content = (
                "```\n"
                + body
                + "\n```\n\n"
                + f"_Live stream: elapsed {elapsed_secs}s · last log {idle_secs}s ago_"
            )
            if self._message is None:
                self._message = cl.Message(content=content, author=self._author)
                await self._message.send()
            else:
                self._message.content = content
                await self._message.update()

    async def _heartbeat(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(self._heartbeat_interval)
            await self._flush()

    async def start(self) -> None:
        await self._flush()
        self._heartbeat_task = asyncio.create_task(self._heartbeat())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        with self._lines_lock:
            has_lines = bool(self._lines)
        if has_lines:
            await self._flush()

    def callback(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return

        with self._lines_lock:
            self._lines.append(stripped)
            overflow = len(self._lines) - self._max_buffer_lines
            if overflow > 0:
                del self._lines[:overflow]
                self._dropped_line_count += overflow
        now = time.time()
        self._last_activity = now
        if now - self._last_update <= self._update_interval:
            return

        self._last_update = now
        future = asyncio.run_coroutine_threadsafe(self._flush(), self._loop)

        def _on_done(done_future: Any) -> None:
            try:
                done_future.result()
            except Exception as exc:
                self._logger.debug("%s: %s", self._error_tag, exc)

        try:
            future.add_done_callback(_on_done)
        except Exception as exc:
            self._logger.debug("Could not attach live-flush callback: %s", exc)
