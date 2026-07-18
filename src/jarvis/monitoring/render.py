"""Render a :class:`MonitorSnapshot` to a self-contained HTML dashboard.

Pure function of the snapshot — no I/O, no external assets (inline CSS only) so
the output opens straight in a browser. The renderer colors states but never
changes them: an ``UNVERIFIED`` or ``FAILED`` reads as itself, honestly
(development-principles: monitoring must not fabricate success).
"""

from __future__ import annotations

from html import escape

from jarvis.monitoring.snapshot import (
    AdapterView,
    CommandView,
    FusionView,
    GazeView,
    GestureView,
    LatencyView,
    MonitorSnapshot,
    TraceRow,
)

# State/status string → CSS class. Unlisted values fall back to "neutral".
_STATUS_CLASS: dict[str, str] = {
    # positive / confirmed
    "VERIFIED": "ok",
    "ACKNOWLEDGED": "ok",
    "TARGET_LOCKED": "ok",
    "COMMITTED": "ok",
    "on": "ok",
    # in-progress / caution
    "UNVERIFIED": "warn",
    "CANDIDATE": "warn",
    "TARGET_CANDIDATE": "warn",
    "GESTURE_WAIT": "warn",
    "GESTURE_TRACKING": "warn",
    "INTENT_CANDIDATE": "warn",
    "ONSET": "warn",
    "ACTIVE": "warn",
    "COOLDOWN": "warn",
    # failure / rejection
    "FAILED": "bad",
    "REJECTED": "bad",
    "EXPIRED": "bad",
    "UNCONFIGURED": "bad",
    "UNKNOWN": "bad",
    "TRACKING_LOST": "bad",
}


def _status_class(value: str) -> str:
    return _STATUS_CLASS.get(value, "neutral")


def _chip(value: str) -> str:
    return f'<span class="chip {_status_class(value)}">{escape(value)}</span>'


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} ms"


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _bar(fraction: float, klass: str = "neutral") -> str:
    width = max(0.0, min(1.0, fraction)) * 100
    return f'<div class="bar"><div class="bar-fill {klass}" style="width:{width:.0f}%"></div></div>'


def _unavailable_badge(available: bool) -> str:
    if available:
        return ""
    return '<span class="chip mock">미구현 · mock</span>'


def _capture_panel(snapshot: MonitorSnapshot) -> str:
    cap = snapshot.capture
    drops = "".join(
        f"<li>{escape(name)}: <b>{count}</b> dropped</li>"
        for name, count in cap.queue_drops.items()
    )
    return f"""
    <section class="panel">
      <h2>1 · Capture</h2>
      <div class="kv"><span>FPS</span><b>{"—" if cap.fps is None else f"{cap.fps:.1f}"}</b></div>
      <div class="kv"><span>frame_id</span><b>{cap.latest_frame_id if cap.latest_frame_id is not None else "—"}</b></div>
      <div class="kv"><span>timestamp</span><b>{_ms(cap.latest_timestamp_ms)}</b></div>
      <div class="kv"><span>face</span>{_chip("tracked" if cap.face_tracked else "TRACKING_LOST")}</div>
      <div class="kv"><span>hand</span>{_chip("tracked" if cap.hand_tracked else "TRACKING_LOST")}</div>
      <ul class="drops">{drops or "<li>no drops</li>"}</ul>
    </section>
    """


def _gaze_panel(gaze: GazeView) -> str:
    devices = "".join(
        f'<div class="kv"><span>{escape(d.device_id)}</span>'
        f'<div class="grow">{_bar(d.probability, "ok" if d.device_id == gaze.target else "neutral")}</div>'
        f"<b>{_pct(d.probability)}</b></div>"
        for d in gaze.device_probabilities
    )
    return f"""
    <section class="panel">
      <h2>2 · Gaze Targeting {_unavailable_badge(gaze.available)}</h2>
      <div class="kv"><span>lock</span>{_chip(gaze.lock_state)}</div>
      <div class="kv"><span>target</span>{_chip(gaze.target)}</div>
      <div class="kv"><span>stability</span><b>{_pct(gaze.stability)}</b></div>
      <div class="kv"><span>margin</span><b>{_pct(gaze.margin)}</b></div>
      <div class="kv"><span>dwell</span><b>{gaze.dwell_ms:.0f} ms</b></div>
      <div class="kv"><span>lock TTL</span><b>{_ms(gaze.lock_ttl_remaining_ms)}</b></div>
      <h3>device probabilities</h3>
      {devices or "<p class='muted'>no devices</p>"}
    </section>
    """


def _gesture_panel(gesture: GestureView) -> str:
    phases = ["IDLE", "ONSET", "ACTIVE", "ENDING"]
    track = "".join(
        f'<span class="phase {"on" if p == gesture.phase else ""}">{p}</span>'
        for p in phases
    )
    return f"""
    <section class="panel">
      <h2>3 · Gesture Spotter {_unavailable_badge(gesture.available)}</h2>
      <div class="phases">{track}</div>
      <div class="kv"><span>gesture</span>{_chip(gesture.gesture)}</div>
      <div class="kv"><span>confidence</span><b>{_pct(gesture.gesture_confidence)}</b></div>
      <div class="kv"><span>uncertainty</span><b>{_pct(gesture.uncertainty)}</b></div>
    </section>
    """


def _fusion_panel(fusion: FusionView) -> str:
    conditions = "".join(
        f'<li class="{"pass" if c.passed else "fail"}">'
        f'{"✓" if c.passed else "✗"} {escape(c.label)}</li>'
        for c in fusion.conditions
    )
    score_class = "ok" if fusion.fusion_score >= fusion.commit_threshold else "warn"
    return f"""
    <section class="panel wide">
      <h2>4 · Intent Fusion {_unavailable_badge(fusion.available)}</h2>
      <div class="kv"><span>state</span>{_chip(fusion.state)}</div>
      <div class="kv"><span>score</span>
        <div class="grow">{_bar(fusion.fusion_score, score_class)}</div>
        <b>{fusion.fusion_score:.2f} / {fusion.commit_threshold:.2f}</b></div>
      <h3>commit conditions</h3>
      <ul class="conditions">{conditions or "<li>no conditions</li>"}</ul>
    </section>
    """


def _commands_panel(commands: tuple[CommandView, ...]) -> str:
    rows = "".join(
        f"<tr><td>{escape(c.command_id)}</td><td>{escape(c.device_id)}</td>"
        f"<td>{escape(c.capability)}·{escape(c.operation)}={escape(str(c.value))}</td>"
        f"<td>{_chip(c.state)}</td><td>{_ms(c.ttl_remaining_ms)}</td>"
        f"<td class='muted'>{escape(c.detail)}</td></tr>"
        for c in commands
    )
    return f"""
    <section class="panel wide">
      <h2>5 · Commands</h2>
      <table>
        <thead><tr><th>command</th><th>device</th><th>action</th><th>state</th><th>TTL</th><th>detail</th></tr></thead>
        <tbody>{rows or "<tr><td colspan='6' class='muted'>no commands</td></tr>"}</tbody>
      </table>
    </section>
    """


def _adapters_panel(adapters: tuple[AdapterView, ...]) -> str:
    rows = "".join(
        f"<div class='kv'><span>{escape(a.name)}</span>"
        f"{_chip('configured' if a.configured else 'UNCONFIGURED')}"
        f"{_chip(a.last_status)}<span class='muted'>{escape(a.last_detail)}</span></div>"
        for a in adapters
    )
    return f"""
    <section class="panel">
      <h2>6 · Adapters</h2>
      {rows or "<p class='muted'>no adapters</p>"}
    </section>
    """


def _latency_panel(latency: tuple[LatencyView, ...]) -> str:
    rows = "".join(
        f"<tr><td>{escape(lat.stage)}</td><td>{lat.p50:.0f}</td>"
        f"<td class='{'bad' if lat.over_target else ''}'>{lat.p95:.0f}</td>"
        f"<td>{_ms(lat.target_ms)}</td><td>{lat.count}</td></tr>"
        for lat in latency
    )
    return f"""
    <section class="panel">
      <h2>7 · Latency (ms)</h2>
      <table>
        <thead><tr><th>stage</th><th>p50</th><th>p95</th><th>target</th><th>n</th></tr></thead>
        <tbody>{rows or "<tr><td colspan='5' class='muted'>no samples</td></tr>"}</tbody>
      </table>
    </section>
    """


def _trace_panel(trace: tuple[TraceRow, ...]) -> str:
    rows = "".join(
        f"<tr><td>{t.timestamp_ms}</td><td>{_chip(t.kind)}</td>"
        f"<td>{escape(t.correlation_id)}</td><td class='muted'>{escape(t.detail)}</td></tr>"
        for t in trace
    )
    return f"""
    <section class="panel wide">
      <h2>8 · Trace timeline</h2>
      <table>
        <thead><tr><th>t (ms)</th><th>event</th><th>correlation</th><th>detail</th></tr></thead>
        <tbody>{rows or "<tr><td colspan='4' class='muted'>no events</td></tr>"}</tbody>
      </table>
    </section>
    """


_CSS = """
* { box-sizing: border-box; }
body { margin:0; font:14px/1.5 system-ui, sans-serif; background:#0f1419; color:#e6e6e6; }
header { padding:14px 20px; background:#161b22; border-bottom:1px solid #30363d; }
header h1 { margin:0; font-size:18px; }
header .src { color:#8b949e; font-size:12px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; padding:16px; }
.panel { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }
.panel.wide { grid-column:1 / -1; }
h2 { margin:0 0 10px; font-size:14px; color:#58a6ff; }
h3 { margin:12px 0 6px; font-size:12px; color:#8b949e; text-transform:uppercase; letter-spacing:.04em; }
.kv { display:flex; align-items:center; gap:8px; margin:4px 0; }
.kv > span:first-child { color:#8b949e; min-width:78px; }
.kv .grow { flex:1; }
.kv b { margin-left:auto; }
.chip { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; font-weight:600; background:#30363d; }
.chip.ok { background:#193d2b; color:#3fb950; }
.chip.warn { background:#3d3417; color:#d29922; }
.chip.bad { background:#3d1c1c; color:#f85149; }
.chip.neutral { background:#21262d; color:#8b949e; }
.chip.mock { background:#2d2438; color:#bc8cff; margin-left:8px; }
.bar { height:8px; background:#21262d; border-radius:4px; overflow:hidden; }
.bar-fill { height:100%; }
.bar-fill.ok { background:#3fb950; } .bar-fill.warn { background:#d29922; } .bar-fill.neutral { background:#58a6ff; }
.phases { display:flex; gap:6px; margin-bottom:10px; }
.phase { flex:1; text-align:center; padding:4px; border-radius:4px; background:#21262d; color:#8b949e; font-size:12px; }
.phase.on { background:#193d2b; color:#3fb950; }
ul.conditions, ul.drops { list-style:none; margin:0; padding:0; }
ul.conditions li, ul.drops li { padding:2px 0; }
ul.conditions li.pass { color:#3fb950; } ul.conditions li.fail { color:#f85149; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:5px 8px; border-bottom:1px solid #21262d; }
th { color:#8b949e; font-weight:600; }
td.bad { color:#f85149; font-weight:700; }
.muted { color:#6e7681; }
"""


def render_html(snapshot: MonitorSnapshot) -> str:
    """Render a full self-contained HTML dashboard for one snapshot."""
    body = "\n".join(
        [
            _capture_panel(snapshot),
            _gaze_panel(snapshot.gaze),
            _gesture_panel(snapshot.gesture),
            _fusion_panel(snapshot.fusion),
            _commands_panel(snapshot.commands),
            _adapters_panel(snapshot.adapters),
            _latency_panel(snapshot.latency),
            _trace_panel(snapshot.trace),
        ]
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JARVIS Monitor</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>JARVIS Pipeline Monitor</h1>
  <div class="src">source: {escape(snapshot.source_label)} · generated at {snapshot.generated_at_ms} ms</div>
</header>
<div class="grid">
{body}
</div>
</body>
</html>
"""
