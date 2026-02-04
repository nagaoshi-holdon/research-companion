# -*- coding: utf-8 -*-
"""
Research Companion (Streamlit) - monitor build (no subscription prompts)

Goals:
- Solid, low-friction input flow
- Gentle, compassionate feedback wording
- Timer with seconds, pause/resume, and "early finish" reason capture
- Long-term goals require "deliverable definition" + milestones
- Daily close saves a durable record; sidebar dashboard shows accumulation
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Optional: smoother 1-second refresh without full page jump
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None  # type: ignore


# -----------------------------
# Constants / Paths
# -----------------------------
APP_TITLE = "Research Companion"
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "db.json"

DEFAULT_DAY_CUTOVER_HOUR = 4  # "research day" changes at 4:00 by default

# Timer behavior
TIMER_REFRESH_MS = 1000  # 1 sec
MIN_EARLY_FINISH_THRESHOLD_MIN = 1  # if ended early, ask reason


# -----------------------------
# Utilities
# -----------------------------
def now_local() -> datetime:
    return datetime.now()


def fmt_date(d: date) -> str:
    return d.isoformat()


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_db() -> Dict[str, Any]:
    ensure_dir(DATA_DIR)
    if not DB_PATH.exists():
        return {}
    try:
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_db(db: Dict[str, Any]) -> None:
    ensure_dir(DATA_DIR)
    DB_PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")


def toast(msg: str, icon: Optional[str] = None) -> None:
    """Show a transient notification (toast if available, else info)."""
    if hasattr(st, "toast"):
        try:
            st.toast(msg, icon=icon)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    st.info(msg)


def _ensure_toast_queue() -> None:
    if "_toast_queue" not in st.session_state:
        st.session_state["_toast_queue"] = []


def queue_toast(msg: str, icon: str | None = None) -> None:
    """st.rerun() を跨いでも1回だけ出る通知キュー."""
    if "_toast_queue" not in st.session_state:
        st.session_state["_toast_queue"] = []
    st.session_state["_toast_queue"].append({"msg": msg, "icon": icon})


def flush_toasts() -> None:
    """キューを吐いて空にする（毎回は出ない）"""
    q = st.session_state.get("_toast_queue", [])
    if not q:
        return

    # ここで“必ず”空にする（＝出続けるのを防ぐ）
    st.session_state["_toast_queue"] = []

    for item in q:
        msg = (item.get("msg") or "").strip()
        if not msg:
            continue
        icon = item.get("icon")

        # Streamlitがtoast対応ならtoast、なければinfo（ただしこの実行1回だけ）
        if hasattr(st, "toast"):
            st.toast(msg, icon=icon)
        else:
            st.info(msg)



def rerun() -> None:
    """Streamlit rerun helper."""
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()  # type: ignore[attr-defined]


# -----------------------------
# Data model normalization
# -----------------------------
def normalize_db() -> None:
    db = load_db()

    # Root
    db.setdefault("settings", {})
    db.setdefault("days", {})  # day_id -> day_record
    db.setdefault("goals", [])  # list of long-term goals

    # Settings
    settings = db["settings"]
    settings.setdefault("day_cutover_hour", DEFAULT_DAY_CUTOVER_HOUR)

    # Days
    days = db["days"]
    if not isinstance(days, dict):
        db["days"] = {}
        days = db["days"]

    # Goals
    if not isinstance(db.get("goals"), list):
        db["goals"] = []

    save_db(db)


def research_day_id(dt: datetime, cutover_hour: int) -> str:
    """Return a stable day id for "research day" which changes at cutover_hour."""
    if dt.time() < time(cutover_hour, 0, 0):
        d = (dt.date() - timedelta(days=1))
    else:
        d = dt.date()
    return d.isoformat()


def ensure_current_day() -> str:
    db = load_db()
    cutover_hour = int(db.get("settings", {}).get("day_cutover_hour", DEFAULT_DAY_CUTOVER_HOUR))
    day_id = research_day_id(now_local(), cutover_hour)
    days = db.get("days", {})

    if day_id not in days:
        days[day_id] = new_day_record(day_id)
        db["days"] = days
        save_db(db)
    return day_id


def new_day_record(day_id: str) -> Dict[str, Any]:
    return {
        "id": day_id,
        "created_at": now_local().isoformat(timespec="seconds"),
        "status": "planning",  # planning -> active -> closed
        "plan": {
            "title": "",
            "tasks": [],  # list of task dicts
            "from_goals": [],  # referenced long-term goals
            "suggestions": [],  # suggestion entries
        },
        "today": {
            "active_task_id": None,
            "task_sessions": [],  # list of session dicts
        },
        "close": {
            "feel": None,  # 1-7
            "done_text": "",
            "note_next": "",
            "closed_at": None,
        },
    }


def get_day(day_id: str) -> Dict[str, Any]:
    db = load_db()
    days = db.get("days", {})
    if day_id not in days:
        days[day_id] = new_day_record(day_id)
        db["days"] = days
        save_db(db)
    return days[day_id]


def save_day(day_id: str, record: Dict[str, Any]) -> None:
    db = load_db()
    days = db.get("days", {})
    days[day_id] = record
    db["days"] = days
    save_db(db)


def set_day_status(day_id: str, status: str) -> None:
    d = get_day(day_id)
    d["status"] = status
    if status == "closed":
        d["close"]["closed_at"] = now_local().isoformat(timespec="seconds")
    save_day(day_id, d)


# -----------------------------
# Goals (long-term)
# -----------------------------
def load_goals() -> List[Dict[str, Any]]:
    db = load_db()
    goals = db.get("goals", [])
    if not isinstance(goals, list):
        return []
    return goals


def save_goals(goals: List[Dict[str, Any]]) -> None:
    db = load_db()
    db["goals"] = goals
    save_db(db)


def goal_due_sort_key(g: Dict[str, Any]) -> Tuple[int, str]:
    archived = 1 if bool(g.get("archived", False)) else 0
    due = (g.get("due_date") or "9999-12-31").strip()
    return (archived, due)


def add_goal(title: str, due: date, deliverable: str) -> None:
    goals = load_goals()
    goals.append(
        {
            "id": str(uuid.uuid4()),
            "title": (title or "").strip() or "（無題）",
            "due_date": fmt_date(due),
            "deliverable": (deliverable or "").strip(),
            "created_at": now_local().isoformat(timespec="seconds"),
            "archived": False,
            "milestones": [],  # list of {"id","title","due_date","done","created_at"}
        }
    )
    save_goals(goals)


def update_goal(goal_id: str, patch: Dict[str, Any]) -> None:
    goals = load_goals()
    for g in goals:
        if g.get("id") == goal_id:
            g.update(patch)
            break
    save_goals(goals)


def add_milestone(goal_id: str, title: str, due: date) -> None:
    goals = load_goals()
    for g in goals:
        if g.get("id") == goal_id:
            ms = g.get("milestones", [])
            if not isinstance(ms, list):
                ms = []
            ms.append(
                {
                    "id": str(uuid.uuid4()),
                    "title": (title or "").strip() or "（無題）",
                    "due_date": fmt_date(due),
                    "done": False,
                    "created_at": now_local().isoformat(timespec="seconds"),
                }
            )
            g["milestones"] = ms
            break
    save_goals(goals)


def toggle_milestone(goal_id: str, ms_id: str, done: bool) -> None:
    goals = load_goals()
    for g in goals:
        if g.get("id") == goal_id:
            ms = g.get("milestones", [])
            for m in ms:
                if m.get("id") == ms_id:
                    m["done"] = bool(done)
                    break
            break
    save_goals(goals)


# -----------------------------
# Planning (today's tasks)
# -----------------------------
def ensure_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    task = dict(task)
    if not task.get("id"):
        task["id"] = str(uuid.uuid4())
    task.setdefault("title", "")
    task.setdefault("est_min", 30)
    task.setdefault("done", False)
    task.setdefault("done_at", None)
    task.setdefault("done_reason", "")  # early finish reason/notes
    task.setdefault("created_at", now_local().isoformat(timespec="seconds"))
    task.setdefault("from_goal_id", None)
    return task


def plan_add_task(day_id: str, title: str, est_min: int, from_goal_id: Optional[str] = None) -> None:
    d = get_day(day_id)
    t = ensure_task_id(
        {
            "title": (title or "").strip(),
            "est_min": int(est_min),
            "from_goal_id": from_goal_id,
        }
    )
    d["plan"]["tasks"].append(t)
    save_day(day_id, d)


def plan_update_task(day_id: str, task_id: str, patch: Dict[str, Any]) -> None:
    d = get_day(day_id)
    tasks = d["plan"].get("tasks", [])
    for t in tasks:
        if t.get("id") == task_id:
            t.update(patch)
            break
    d["plan"]["tasks"] = tasks
    save_day(day_id, d)


def plan_remove_task(day_id: str, task_id: str) -> None:
    d = get_day(day_id)
    tasks = [t for t in d["plan"].get("tasks", []) if t.get("id") != task_id]
    d["plan"]["tasks"] = tasks
    # If removing the active task, clear active_task_id
    if d["today"].get("active_task_id") == task_id:
        d["today"]["active_task_id"] = None
    save_day(day_id, d)


def plan_apply_to_today(day_id: str) -> None:
    """Move day status to active and clear stale toasts; keeps tasks as-is."""
    set_day_status(day_id, "active")
    queue_toast("今日へ反映しました。", icon="✅")
    rerun()


# -----------------------------
# Timer sessions
# -----------------------------
def _get_active_session(day: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    sessions = day.get("today", {}).get("task_sessions", [])
    for s in reversed(sessions):
        if s.get("task_id") == task_id and s.get("ended_at") is None:
            return s
    return None


def start_task_session(day_id: str, task_id: str) -> None:
    d = get_day(day_id)
    d["today"]["active_task_id"] = task_id

    s = _get_active_session(d, task_id)
    if s is None:
        d["today"]["task_sessions"].append(
            {
                "id": str(uuid.uuid4()),
                "task_id": task_id,
                "started_at": now_local().isoformat(timespec="seconds"),
                "paused_at": None,
                "pause_total_sec": 0,
                "last_pause_started_at": None,
                "ended_at": None,
                "ended_type": None,  # "done" or "stop"
                "ended_reason": "",
            }
        )
    save_day(day_id, d)


def pause_task_session(day_id: str, task_id: str) -> None:
    d = get_day(day_id)
    s = _get_active_session(d, task_id)
    if not s:
        return
    if s.get("last_pause_started_at") is None:
        s["last_pause_started_at"] = now_local().isoformat(timespec="seconds")
    save_day(day_id, d)


def resume_task_session(day_id: str, task_id: str) -> None:
    d = get_day(day_id)
    s = _get_active_session(d, task_id)
    if not s:
        return
    lp = s.get("last_pause_started_at")
    if lp:
        try:
            lp_dt = datetime.fromisoformat(lp)
            delta = (now_local() - lp_dt).total_seconds()
            s["pause_total_sec"] = int(s.get("pause_total_sec", 0) + max(0, delta))
        except Exception:
            pass
        s["last_pause_started_at"] = None
    save_day(day_id, d)


def end_task_session(day_id: str, task_id: str, ended_type: str, reason: str = "") -> None:
    d = get_day(day_id)
    s = _get_active_session(d, task_id)
    if not s:
        return

    # If currently paused, resume first to account pause time
    if s.get("last_pause_started_at"):
        resume_task_session(day_id, task_id)
        d = get_day(day_id)
        s = _get_active_session(d, task_id) or s

    s["ended_at"] = now_local().isoformat(timespec="seconds")
    s["ended_type"] = ended_type
    s["ended_reason"] = (reason or "").strip()

    # When done, mark task done and save reason if needed
    if ended_type == "done":
        mark_task_done(day_id, task_id, reason_for_early_finish=reason)

    # Clear active task if it matches
    if d["today"].get("active_task_id") == task_id:
        d["today"]["active_task_id"] = None

    save_day(day_id, d)


def session_elapsed_seconds(s: Dict[str, Any]) -> int:
    try:
        start = datetime.fromisoformat(s.get("started_at"))
    except Exception:
        return 0
    end = now_local()
    try:
        if s.get("ended_at"):
            end = datetime.fromisoformat(s.get("ended_at"))
    except Exception:
        pass

    total = (end - start).total_seconds()

    # Subtract pause total and current pause ongoing
    pause_total = int(s.get("pause_total_sec", 0))
    lp = s.get("last_pause_started_at")
    if lp:
        try:
            lp_dt = datetime.fromisoformat(lp)
            pause_total += int(max(0, (now_local() - lp_dt).total_seconds()))
        except Exception:
            pass

    return int(max(0, total - pause_total))


# -----------------------------
# Mark done & early-finish reason
# -----------------------------
def mark_task_done(day_id: str, task_id: str, reason_for_early_finish: str = "") -> None:
    d = get_day(day_id)
    tasks = d["plan"].get("tasks", [])
    for t in tasks:
        if t.get("id") == task_id:
            t["done"] = True
            t["done_at"] = now_local().isoformat(timespec="seconds")
            if reason_for_early_finish:
                t["done_reason"] = reason_for_early_finish.strip()
            break
    d["plan"]["tasks"] = tasks
    save_day(day_id, d)


# -----------------------------
# Reports / Dashboard summaries
# -----------------------------
@dataclass
class WeekSummary:
    week_start: date
    week_end: date
    days: int
    total_focus_min: int
    tasks_done: int
    tasks_total: int
    top_tasks: List[Tuple[str, int]]


def iter_day_ids_sorted(db: Dict[str, Any]) -> List[str]:
    days = db.get("days", {})
    if not isinstance(days, dict):
        return []
    return sorted(days.keys())


def day_total_focus_seconds(day: Dict[str, Any]) -> int:
    secs = 0
    for s in day.get("today", {}).get("task_sessions", []):
        secs += session_elapsed_seconds(s)
    return secs


def summarize_recent_days(n: int = 7) -> List[Dict[str, Any]]:
    db = load_db()
    ids = iter_day_ids_sorted(db)
    ids = ids[-n:]
    out = []
    for did in ids:
        d = db["days"][did]
        tasks = d.get("plan", {}).get("tasks", [])
        done = [t for t in tasks if t.get("done")]
        out.append(
            {
                "day_id": did,
                "status": d.get("status"),
                "focus_min": int(day_total_focus_seconds(d) // 60),
                "tasks_done": len(done),
                "tasks_total": len(tasks),
                "feel": d.get("close", {}).get("feel"),
            }
        )
    return list(reversed(out))


def weekly_report(db: Dict[str, Any], end_day: date) -> WeekSummary:
    """
    Build a 7-day report ending at end_day (inclusive).
    """
    start_day = end_day - timedelta(days=6)
    days_dict = db.get("days", {})
    tasks_done = 0
    tasks_total = 0
    total_focus_min = 0

    task_focus: Dict[str, int] = {}

    for i in range(7):
        did = (start_day + timedelta(days=i)).isoformat()
        d = days_dict.get(did)
        if not d:
            continue
        total_focus_min += int(day_total_focus_seconds(d) // 60)

        tasks = d.get("plan", {}).get("tasks", [])
        tasks_total += len(tasks)
        tasks_done += len([t for t in tasks if t.get("done")])

        # allocate focus minutes to active task in sessions
        for s in d.get("today", {}).get("task_sessions", []):
            sec = session_elapsed_seconds(s)
            tid = s.get("task_id")
            if not tid:
                continue
            # find title
            title = ""
            for t in tasks:
                if t.get("id") == tid:
                    title = (t.get("title") or "").strip()
                    break
            title = title or "（不明タスク）"
            task_focus[title] = task_focus.get(title, 0) + int(sec // 60)

    top = sorted(task_focus.items(), key=lambda x: x[1], reverse=True)[:10]
    return WeekSummary(
        week_start=start_day,
        week_end=end_day,
        days=7,
        total_focus_min=total_focus_min,
        tasks_done=tasks_done,
        tasks_total=tasks_total,
        top_tasks=top,
    )


def render_weekly_report_download() -> None:
    db = load_db()
    cutover = int(db.get("settings", {}).get("day_cutover_hour", DEFAULT_DAY_CUTOVER_HOUR))
    # Align report end_day to research day of "now"
    end_id = research_day_id(now_local(), cutover)
    end_day = parse_date(end_id)
    rep = weekly_report(db, end_day)

    md = []
    md.append(f"# 週間レポート（{rep.week_start.isoformat()}〜{rep.week_end.isoformat()}）")
    md.append("")
    md.append(f"- 合計集中時間: **{rep.total_focus_min} 分**")
    md.append(f"- タスク完了: **{rep.tasks_done}/{rep.tasks_total}**")
    md.append("")
    if rep.top_tasks:
        md.append("## よく取り組んだこと（上位）")
        for title, mins in rep.top_tasks:
            md.append(f"- {title}: {mins} 分")
    else:
        md.append("## よく取り組んだこと（上位）")
        md.append("- （まだ記録がありません）")

    md_text = "\n".join(md)
    st.download_button(
        "週間レポートをダウンロード（Markdown）",
        data=md_text.encode("utf-8"),
        file_name=f"weekly_report_{rep.week_end.isoformat()}.md",
        mime="text/markdown",
        use_container_width=True,
    )


# -----------------------------
# UX text helpers (compassionate)
# -----------------------------
def gentle_feedback_for_plan(day: Dict[str, Any], goals: List[Dict[str, Any]]) -> str:
    """
    Provide gentle suggestions based on yesterday/today.
    No "AI judgement"; keep human-friendly and optional.
    """
    db = load_db()
    ids = iter_day_ids_sorted(db)
    if not ids:
        return "まずは今日を軽く整えるだけで十分です。ここにいる時点で、もう前に進んでいます。"
    # find yesterday
    try:
        idx = ids.index(day.get("id"))
    except Exception:
        idx = len(ids) - 1
    y = None
    if idx - 1 >= 0:
        y = db.get("days", {}).get(ids[idx - 1])

    pieces = []
    if y:
        y_tasks = y.get("plan", {}).get("tasks", [])
        y_done = [t for t in y_tasks if t.get("done")]
        y_focus = int(day_total_focus_seconds(y) // 60)
        if y_focus > 0:
            pieces.append(f"昨日の集中時間は **{y_focus}分**。積み上げがちゃんと残っています。")
        if y_tasks:
            pieces.append(f"昨日のタスク完了は **{len(y_done)}/{len(y_tasks)}**。小さくても前進です。")

    # If there are goals with deliverables, suggest connecting
    active_goals = [g for g in goals if not g.get("archived")]
    if active_goals and not day.get("plan", {}).get("from_goals"):
        pieces.append("もしよければ、遠い目標から“今日やること”へ 1つだけ繋げてみましょう。継続の意味が毎日出ます。")

    if not pieces:
        return "今日は、無理なく進める形に整えるだけでOKです。やることを小さくしても、進捗は進捗です。"
    return " ".join(pieces)


# -----------------------------
# CSS
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
<style>
/* Keep UI solid & simple */
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
h1,h2,h3 { letter-spacing: -0.02em; }
div[data-testid="stMetric"] { border-radius: 12px; padding: 10px 12px; background: rgba(0,0,0,0.03); }
.small-muted { color: rgba(0,0,0,0.55); font-size: 0.92rem; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(0,0,0,0.06); font-size: 0.85rem; margin-left: 6px; }
hr { margin: 1.0rem 0; }
</style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Sidebar dashboard
# -----------------------------
def sidebar_dashboard(current_day_id: str) -> None:
    st.sidebar.markdown(f"## {APP_TITLE}")
    st.sidebar.caption(f"研究日: {current_day_id}")

    # Settings - renamed for clarity
    db = load_db()
    settings = db.get("settings", {})
    cutover_hour = int(settings.get("day_cutover_hour", DEFAULT_DAY_CUTOVER_HOUR))

    with st.sidebar.expander("日付変更時間の設定", expanded=False):
        rh = st.number_input("研究日の区切り（時）", min_value=0, max_value=8, step=1, value=cutover_hour, key="day_cutover_hour_widget")
        if st.button("適用", key="apply_cutover", use_container_width=True):
            settings["day_cutover_hour"] = int(rh)
            db["settings"] = settings
            save_db(db)
            queue_toast("区切り設定を適用しました。", icon="🕒")
            rerun()

    # Recent records
    st.sidebar.markdown("### 最近の記録")
    rec = summarize_recent_days(7)
    if not rec:
        st.sidebar.caption("まだ記録がありません。")
    else:
        for r in rec:
            feel = r.get("feel")
            feel_text = f" 手応え:{feel}" if feel else ""
            st.sidebar.write(f"- {r['day_id']}  {r['tasks_done']}/{r['tasks_total']}  {r['focus_min']}分{feel_text}")

    st.sidebar.divider()

    # Goals quick view (with created_at shown clearly)
    st.sidebar.markdown("### 遠い目標（β）")
    goals = load_goals()
    active_goals = [g for g in goals if not g.get("archived")]
    if not active_goals:
        st.sidebar.caption("まだ目標がありません。今日することをまとめるで追加できます。")
    else:
        for g in sorted(active_goals, key=goal_due_sort_key)[:8]:
            title = (g.get("title") or "（無題）").strip()
            due = (g.get("due_date") or "").strip()
            created_at = (g.get("created_at") or "").strip()
            st.sidebar.write(f"- {title}")
            if due:
                st.sidebar.caption(f"期限: {due}")
            if created_at:
                st.sidebar.caption(f"作成: {created_at}")

    with st.sidebar.expander("目標一覧（詳細）", expanded=False):
        if not goals:
            st.caption("まだ目標がありません。")
        else:
            for g in sorted(goals, key=goal_due_sort_key):
                badge = "（アーカイブ）" if g.get("archived") else ""
                title = (g.get("title") or "（無題）").strip()
                st.markdown(f"**{title} {badge}**")
                st.caption(f"期限: {(g.get('due_date') or '').strip()} / 作成: {(g.get('created_at') or '').strip()}")
                st.caption(f"成果物: {(g.get('deliverable') or '').strip()}")
                ms = g.get("milestones", [])
                if ms:
                    done = len([m for m in ms if m.get("done")])
                    st.caption(f"マイルストーン: {done}/{len(ms)}")

    st.sidebar.divider()

    # Weekly report download
    st.sidebar.markdown("### 無料2週間の自動レポート（試作）")
    st.sidebar.caption("モニター向け：継続の根拠づくりとして、まずはレポート品質を上げます。")
    render_weekly_report_download()



# -----------------------------
# Top navigation (tabs)
# -----------------------------
def top_nav() -> str:
    """
    Use segmented control style with radio for compatibility.
    """
    opts = ["今日することをまとめる", "今日", "今日を終える"]
    default = 0  # show "today plan" on app start
    page = st.radio("画面", opts, index=default, horizontal=True, label_visibility="collapsed")
    return page


# -----------------------------
# Page: Plan (今日することをまとめる)
# -----------------------------
def page_plan(day_id: str) -> None:
    d = get_day(day_id)
    goals = load_goals()

    st.markdown("## 今日することをまとめる")
    st.caption("入力はソリッドに。提案や言葉はやさしく。")

    # Gentle feedback (optional)
    with st.expander("やさしい振り返り（任意）", expanded=False):
        st.write(gentle_feedback_for_plan(d, [g for g in goals if not g.get("archived")]))
        st.caption("要らない日は閉じてOKです。")

    st.divider()

    # ---- Long-term goals editor (in-plan) ----
    st.markdown("### 遠い目標（β）")
    st.caption("締切だけでは弱いので、「成果物（達成条件）」を必須にしています。複数の目標を設定できます。")

    with st.expander("＋ 目標を追加", expanded=False):
        default_due = parse_date(day_id) + timedelta(days=14)
        with st.form("add_goal_form", clear_on_submit=True):
            title = st.text_input("目標タイトル", key="new_goal_title", placeholder="例：学会発表 / 資格試験 / レポート提出")
            due = st.date_input("期限", value=default_due, key="new_goal_due")
            deliverable = st.text_area(
                "成果物（達成条件）※必須",
                key="new_goal_deliverable",
                placeholder="例：スライド20枚＋要旨最終版＋予行2回\n例：第3章の結果まで書き切る",
                height=80,
            )
            submitted = st.form_submit_button("追加", use_container_width=True)
        if submitted:
            if not (deliverable or "").strip():
                st.error("成果物（達成条件）は必須です。短くて大丈夫です。")
            else:
                add_goal(title=title, due=due, deliverable=deliverable)
                queue_toast("目標を追加しました。", icon="🎯")
                rerun()

    active_goals = [g for g in goals if not g.get("archived")]
    if active_goals:
        with st.expander("目標の編集・マイルストーン", expanded=False):
            for g in sorted(active_goals, key=goal_due_sort_key):
                gid = g.get("id")
                title = (g.get("title") or "（無題）").strip()
                st.markdown(f"#### {title}")
                st.caption(f"期限: {g.get('due_date')} / 作成: {g.get('created_at')}")
                st.write(f"成果物: {g.get('deliverable')}")
                cols = st.columns([1, 1, 2])
                with cols[0]:
                    if st.button("アーカイブ", key=f"arch_{gid}", use_container_width=True):
                        update_goal(gid, {"archived": True})
                        queue_toast("アーカイブしました。", icon="🗂️")
                        rerun()
                with cols[1]:
                    if st.button("今日に紐づけ", key=f"link_{gid}", use_container_width=True):
                        # store only ids in from_goals list (no duplicates)
                        d = get_day(day_id)
                        fg = d["plan"].get("from_goals", [])
                        if gid not in fg:
                            fg.append(gid)
                        d["plan"]["from_goals"] = fg
                        save_day(day_id, d)
                        queue_toast("今日に紐づけました。", icon="🔗")
                        rerun()

                # milestones
                ms = g.get("milestones", [])
                if ms:
                    st.caption("マイルストーン")
                    for m in ms:
                        mid = m.get("id")
                        done = bool(m.get("done"))
                        c = st.checkbox(
                            f"{m.get('title')}（期限: {m.get('due_date')}）",
                            value=done,
                            key=f"ms_{gid}_{mid}",
                        )
                        if c != done:
                            toggle_milestone(gid, mid, c)
                            queue_toast("更新しました。", icon="✅")
                            rerun()
                with st.expander("＋ マイルストーン追加", expanded=False):
                    default_ms_due = parse_date(g.get("due_date"))
                    with st.form(f"add_ms_form_{gid}", clear_on_submit=True):
                        ms_title = st.text_input("中間締切（マイルストーン）", key=f"new_ms_title_{gid}", placeholder="例：構成案を確定 / 図表を作る")
                        ms_due = st.date_input("期限", value=default_ms_due, key=f"new_ms_due_{gid}")
                        ms_submit = st.form_submit_button("追加", use_container_width=True)
                    if ms_submit:
                        add_milestone(gid, ms_title, ms_due)
                        queue_toast("マイルストーンを追加しました。", icon="📍")
                        rerun()

    st.divider()

    # ---- Today's tasks ----
    st.markdown("### 今日のタスク")
    st.caption("「成果物」ではなく、**やること（短く）** で入力します。")

    tasks = d.get("plan", {}).get("tasks", [])
    tasks = [ensure_task_id(t) for t in tasks]
    d["plan"]["tasks"] = tasks
    save_day(day_id, d)

    with st.form("add_task_form", clear_on_submit=True):
        title = st.text_input("やること（短く）", key="new_task_title", placeholder="例：導入1ページを書く / 図1の作成 / 先行研究1本読む")
        est = st.number_input("目標時間（分）", min_value=1, max_value=600, value=30, step=5, key="new_task_est")
        # Optional link from goal
        goal_opts = ["（紐づけなし）"] + [f"{g.get('title')}（期限:{g.get('due_date')}）" for g in active_goals]
        goal_sel = st.selectbox("遠い目標と紐づけ（任意）", goal_opts, index=0, key="new_task_goal")
        submitted = st.form_submit_button("追加", use_container_width=True)

    if submitted:
        if not (title or "").strip():
            st.warning("やることを短く1つだけ入れてみましょう。")
        else:
            gid = None
            if goal_sel != "（紐づけなし）":
                # find by title match
                for g in active_goals:
                    label = f"{g.get('title')}（期限:{g.get('due_date')}）"
                    if label == goal_sel:
                        gid = g.get("id")
                        break
            plan_add_task(day_id, title=title, est_min=int(est), from_goal_id=gid)
            queue_toast("タスクを追加しました。", icon="🧩")
            rerun()

    if tasks:
        st.markdown("#### 一覧")
        for t in tasks:
            tid = t.get("id")
            title = (t.get("title") or "").strip()
            est_min = int(t.get("est_min") or 0)
            done = bool(t.get("done"))
            badge = "✅" if done else "⏳"
            cols = st.columns([6, 2, 2, 2])
            with cols[0]:
                st.write(f"{badge} {title}")
                if t.get("from_goal_id"):
                    # show goal title
                    gtitle = ""
                    for g in goals:
                        if g.get("id") == t.get("from_goal_id"):
                            gtitle = (g.get("title") or "").strip()
                            break
                    if gtitle:
                        st.caption(f"紐づけ: {gtitle}")
            with cols[1]:
                new_est = st.number_input("分", min_value=1, max_value=600, value=est_min, step=5, key=f"est_{tid}", label_visibility="collapsed")
                if new_est != est_min:
                    plan_update_task(day_id, tid, {"est_min": int(new_est)})
            with cols[2]:
                if st.button("削除", key=f"del_{tid}", use_container_width=True):
                    plan_remove_task(day_id, tid)
                    queue_toast("削除しました。", icon="🗑️")
                    rerun()
            with cols[3]:
                # quick done toggle
                if not done:
                    if st.button("完了", key=f"done_{tid}", use_container_width=True):
                        mark_task_done(day_id, tid)
                        queue_toast("完了にしました。", icon="✅")
                        rerun()
                else:
                    if st.button("未完了に戻す", key=f"undone_{tid}", use_container_width=True):
                        plan_update_task(day_id, tid, {"done": False, "done_at": None, "done_reason": ""})
                        queue_toast("未完了に戻しました。", icon="↩️")
                        rerun()
    else:
        st.info("まずはタスクを1つだけ追加してみましょう。")

    st.divider()

    # Apply button
    st.markdown("### 今日へ反映")
    st.caption("押した時点のタスクに固定されず、あとから追加した分も保存時に積み上がります。")
    if st.button("今日へ反映する", use_container_width=True):
        plan_apply_to_today(day_id)


# -----------------------------
# Page: Today (今日)
# -----------------------------
def page_today(day_id: str) -> None:
    d = get_day(day_id)
    st.markdown("## 今日")
    st.caption("タスクを開始するとタイマーが動きます。中断・再開できます。")

    tasks = d.get("plan", {}).get("tasks", [])
    if not tasks:
        st.info("今日のタスクがありません。まず「今日することをまとめる」で追加してください。")
        return

    # Show status summary
    done = [t for t in tasks if t.get("done")]
    st.metric("完了", f"{len(done)}/{len(tasks)}")

    active_task_id = d.get("today", {}).get("active_task_id")

    # List tasks with Start/Pause/Resume/Done logic
    for t in tasks:
        tid = t.get("id")
        title = (t.get("title") or "").strip()
        est_min = int(t.get("est_min") or 0)
        done_flag = bool(t.get("done"))

        st.markdown("---")
        st.markdown(f"### {title} {'<span class=\"badge\">完了</span>' if done_flag else ''}", unsafe_allow_html=True)
        st.caption(f"目標: {est_min}分")

        # Find active session for this task
        s = _get_active_session(d, tid)

        # Timer display
        if s:
            elapsed = session_elapsed_seconds(s)
            st.write(f"⏱️ 経過: **{elapsed}秒**（{elapsed//60}分）")
            remaining = max(0, est_min * 60 - elapsed)
            st.write(f"🎯 残り: **{remaining}秒**（{remaining//60}分）")

            # Auto refresh for seconds
            if st_autorefresh is not None:
                st_autorefresh(interval=TIMER_REFRESH_MS, key=f"refresh_{tid}")
            else:
                # fallback: gentle note
                st.caption("※ 秒更新をより滑らかにするには streamlit-autorefresh を使えます。")

            paused = s.get("last_pause_started_at") is not None
            cols = st.columns([1, 1, 2])
            with cols[0]:
                if not paused:
                    if st.button("中断", key=f"pause_{tid}", use_container_width=True):
                        pause_task_session(day_id, tid)
                        rerun()
                else:
                    if st.button("再開", key=f"resume_{tid}", use_container_width=True):
                        resume_task_session(day_id, tid)
                        rerun()

            with cols[1]:
                # Done flow: if ended early, ask reason
                if st.button("できた", key=f"done_btn_{tid}", use_container_width=True, disabled=done_flag):
                    # if elapsed < target and difference >= threshold, prompt reason
                    diff_min = (est_min * 60 - elapsed) / 60.0
                    if diff_min >= MIN_EARLY_FINISH_THRESHOLD_MIN:
                        st.session_state[f"need_reason_{tid}"] = True
                    else:
                        end_task_session(day_id, tid, ended_type="done", reason="")
                        queue_toast("おつかれさま。完了として記録しました。", icon="✅")
                        rerun()

            with cols[2]:
                if st.button("停止（中断のまま終える）", key=f"stop_{tid}", use_container_width=True):
                    end_task_session(day_id, tid, ended_type="stop", reason="")
                    queue_toast("停止しました。必要ならあとで再開できます。", icon="⏸️")
                    rerun()

            # Early finish reason UI
            if st.session_state.get(f"need_reason_{tid}", False):
                st.warning("目標時間より早く終えました。差し支えなければ理由を選ぶか、短く書いてください（あなたを責める意図はゼロです）。")
                reason_opt = st.selectbox(
                    "理由（選択）",
                    [
                        "思ったより早く終わった（良い意味で）",
                        "途中で方針変更した（別タスクに繋げるため）",
                        "体調・用事などやむを得ない",
                        "未完了だが一旦区切った（続きは後で）",
                        "その他",
                    ],
                    key=f"reason_opt_{tid}",
                )
                reason_text = st.text_input("補足（任意）", key=f"reason_txt_{tid}", placeholder="例：集中が切れたので15分だけやった / 先に図を作る方が良いと判断")
                rcols = st.columns([1, 1, 2])
                with rcols[0]:
                    if st.button("理由付きで完了にする", key=f"confirm_done_{tid}", use_container_width=True):
                        reason = reason_opt
                        if reason_text.strip():
                            reason += f" / {reason_text.strip()}"
                        end_task_session(day_id, tid, ended_type="done", reason=reason)
                        st.session_state[f"need_reason_{tid}"] = False
                        queue_toast("理由も含めて完了として記録しました。", icon="✅")
                        rerun()
                with rcols[1]:
                    if st.button("やっぱり続ける", key=f"cancel_reason_{tid}", use_container_width=True):
                        st.session_state[f"need_reason_{tid}"] = False
                        rerun()
                with rcols[2]:
                    st.caption("※ ここでの入力は「あなたの記録」を丁寧にするためだけに使います。")

        else:
            # Not running
            cols = st.columns([1, 1, 3])
            with cols[0]:
                if st.button("開始", key=f"start_{tid}", use_container_width=True, disabled=done_flag):
                    start_task_session(day_id, tid)
                    queue_toast("開始しました。ゆっくりで大丈夫。", icon="▶️")
                    rerun()
            with cols[1]:
                if done_flag:
                    st.write("✅ 完了")
                else:
                    st.write("")
            with cols[2]:
                # Show last session summary if exists
                sessions = d.get("today", {}).get("task_sessions", [])
                last = None
                for s2 in reversed(sessions):
                    if s2.get("task_id") == tid and s2.get("ended_at") is not None:
                        last = s2
                        break
                if last:
                    sec = session_elapsed_seconds(last)
                    st.caption(f"前回: {sec//60}分 {sec%60}秒 / 終了: {last.get('ended_type')}")
                    # If task marked done with reason, show it
                    if t.get("done_reason"):
                        st.caption(f"メモ: {t.get('done_reason')}")

    st.divider()

    # CTA to close day
    if st.button("今日を終えるへ移動", use_container_width=True):
        # set radio page by session_state? easiest: just inform
        queue_toast("上のタブから「今日を終える」に移動してください。", icon="➡️")
        rerun()


# -----------------------------
# Page: Close (今日を終える)
# -----------------------------
def page_close(day_id: str) -> None:
    d = get_day(day_id)
    st.markdown("## 今日を終える")
    st.caption("最後に保存すると、記録として積み上がります。")

    tasks = d.get("plan", {}).get("tasks", [])
    sessions = d.get("today", {}).get("task_sessions", [])

    # Summary table
    st.markdown("### 今日のまとめ")
    total_focus_min = int(day_total_focus_seconds(d) // 60)
    done = [t for t in tasks if t.get("done")]
    st.write(f"- 集中時間: **{total_focus_min}分**")
    st.write(f"- タスク完了: **{len(done)}/{len(tasks)}**")

    if tasks:
        st.markdown("#### タスク別（完了/時間）")
        for t in tasks:
            tid = t.get("id")
            title = (t.get("title") or "").strip()
            est_min = int(t.get("est_min") or 0)
            done_flag = bool(t.get("done"))
            # sum sessions for this task
            sec = 0
            for s in sessions:
                if s.get("task_id") == tid:
                    sec += session_elapsed_seconds(s)
            actual_min = int(sec // 60)
            st.write(f"- {'✅' if done_flag else '⬜'} {title} / 目標:{est_min}分 / 実績:{actual_min}分")
            if done_flag and (t.get("done_reason") or "").strip():
                st.caption(f"  メモ: {t.get('done_reason')}")

    st.divider()

    # Reflections
    st.markdown("### 手応え（1〜7）")
    feel = st.slider("今日の手応え", min_value=1, max_value=7, value=int(d.get("close", {}).get("feel") or 4))

    st.markdown("### 今日できたこと（短く）")
    done_text = st.text_area(
        "自由記述（任意）",
        value=(d.get("close", {}).get("done_text") or ""),
        placeholder="例：導入の骨子を作れた / 図の方針が決まった / 手を動かせた",
        height=90,
    )

    st.markdown("### 明日につなぐメモ（任意）")
    note_next = st.text_area(
        "明日の自分へ",
        value=(d.get("close", {}).get("note_next") or ""),
        placeholder="例：次はこの論文の結果だけ読む / 図2を先に作る / 30分だけ着手",
        height=90,
    )

    st.markdown("### 保存")
    st.caption("保存すると、サイドバーのダッシュボードに積み上がります。")

    if st.button("今日を終える（保存）", use_container_width=True):
        d["close"]["feel"] = int(feel)
        d["close"]["done_text"] = done_text.strip()
        d["close"]["note_next"] = note_next.strip()
        d["close"]["prompt_log"] = ""
        set_day_status(day_id, "closed")
        if hasattr(st, "balloons"):
            st.balloons()
        st.success("保存しました。今日の分は確実に積み上がっています。")
        rerun()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    normalize_db()
    inject_css()

    current_day_id = ensure_current_day()
    sidebar_dashboard(current_day_id)

    st.markdown(f"# {APP_TITLE}")
    st.caption("モニター版：サブスク誘導は停止中。まずは体験と改善に集中します。")

    page = top_nav()

    # ←この直後に追加
    flush_toasts()

    if page == "今日することをまとめる":
        page_plan(current_day_id)
    elif page == "今日":
        page_today(current_day_id)
    else:
        page_close(current_day_id)


if __name__ == "__main__":
    main()
