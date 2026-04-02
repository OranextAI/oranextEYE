"""
demo1.py — Dynamic multi-model runner.

Reads enabled models from the database (camera_ai table) every POLL_INTERVAL
seconds. Starts new model threads when enabled, stops them when disabled.
No restart needed when toggling models in the web app.
"""

import asyncio
import threading
import time
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from oureyes.puller_bis import pull_stream
from oureyes.debug_timing import mark_stage

from models.fire_detection.fire_detection import fire_detection
from models.ppe_detection.ppe_detection import ppe_detection
from models.zone_detection.zone_detection import zone_detection
from models.zone_analysis.zone_analysis import zone_analysis

# ── Config ────────────────────────────────────────────────────────────────
FPS          = 25
POLL_INTERVAL = 5   # seconds between DB checks
RESTART_DELAY = 5   # seconds before restarting a crashed thread

# Map model_name → function
MODEL_FN = {
    "fire_detection":  fire_detection,
    "ppe_detection":   ppe_detection,
    "zone_detection":  zone_detection,
    "zone_analysis":   zone_analysis,
}

# ── DB connection (persistent, reconnects on error) ───────────────────────
_db_conn = None

def get_db():
    global _db_conn
    try:
        if _db_conn is None or _db_conn.closed:
            _db_conn = psycopg2.connect(
                host=os.getenv("PGHOST",       "4.233.212.118"),
                port=int(os.getenv("PGPORT",   "5432")),
                dbname=os.getenv("PGDATABASE", "oranextdb"),
                user=os.getenv("PGUSER",       "postgres"),
                password=os.getenv("PGPASSWORD", "smartiCPS@2025"),
            )
        return _db_conn
    except Exception as e:
        print(f"[db] Connection error: {e}")
        _db_conn = None
        raise

def fetch_enabled_models():
    """Return enabled rows from camera_ai with stream info and zone points."""
    try:
        conn = get_db()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT ca.id, ca.model_name, ca.stream_id, ca.enabled,
                   s.streamname AS camera_stream
            FROM camera_ai ca
            JOIN device d ON ca.camera_id = d.id
            JOIN stream  s ON s.iddevice  = d.id
            WHERE ca.enabled = true
              AND ca.model_name = ANY(%s)
        """, (list(MODEL_FN.keys()),))
        rows = cur.fetchall()

        # Attach zone points for zone-based models
        result = []
        for row in rows:
            cur.execute("""
                SELECT points FROM camera_zones
                WHERE camera_ai_id = %s ORDER BY zone_index
            """, (row["id"],))
            zone_rows = cur.fetchall()
            row = dict(row)
            row["zone_points"] = [z["points"] for z in zone_rows]
            result.append(row)

        cur.close()
        return result
    except Exception as e:
        print(f"[db] Error fetching models: {e}")
        global _db_conn
        _db_conn = None
        return []

# ── Thread management ─────────────────────────────────────────────────────
# Key: (model_name, camera_stream)  Value: {"thread": Thread, "stop": Event}
_running: dict = {}
_lock = threading.Lock()

def sync_frames(queue, loop, label, stop_event):
    """Yield frames from asyncio queue, stopping when stop_event is set."""
    while not stop_event.is_set():
        try:
            frame = asyncio.run_coroutine_threadsafe(
                queue.get(), loop
            ).result(timeout=5.0)
            mark_stage("model_input", label, frame, pop=False)
            yield frame
        except Exception:
            if stop_event.is_set():
                return
            time.sleep(0.2)

def run_model_thread(model_fn, queue, loop, dest_cam, label, stop_event, zone_points=None):
    """Run model in a loop, respecting stop_event."""
    # Models that accept zone_points
    ZONE_MODELS = {"zone_analysis", "zone_detection", "surveillance_zones"}
    model_name = label.split("[")[0]

    while not stop_event.is_set():
        try:
            print(f"🚀 [{label}] → {dest_cam}")
            kwargs = {"dest_cam": dest_cam, "fps": FPS}
            if model_name in ZONE_MODELS and zone_points is not None:
                kwargs["zone_points"] = zone_points
            model_fn(sync_frames(queue, loop, label, stop_event), **kwargs)
        except Exception as e:
            if stop_event.is_set():
                break
            import traceback
            print(f"❌ [{label}] crashed: {e}")
            traceback.print_exc()
            time.sleep(RESTART_DELAY)

async def start_model(row, loop):
    """Start a model thread for a DB row."""
    key = (row["model_name"], row["camera_stream"])
    with _lock:
        if key in _running:
            return  # already running

    model_fn = MODEL_FN.get(row["model_name"])
    if not model_fn:
        print(f"⚠️  No function for model: {row['model_name']}")
        return

    queue     = await pull_stream(row["camera_stream"])
    stop_evt  = threading.Event()
    label     = f"{row['model_name']}[{row['camera_stream']}]"
    dest_cam  = row["stream_id"]

    t = threading.Thread(
        target=run_model_thread,
        args=(model_fn, queue, loop, dest_cam, label, stop_evt, row.get("zone_points", [])),
        name=label,
        daemon=True,
    )
    t.start()

    with _lock:
        _running[key] = {"thread": t, "stop": stop_evt, "queue": queue}

    print(f"✅ Started  {label} → {dest_cam}")

def stop_model(key):
    """Stop a running model thread."""
    with _lock:
        entry = _running.pop(key, None)
    if entry:
        entry["stop"].set()
        print(f"🛑 Stopped  {key[0]}[{key[1]}]")

# ── Main loop ─────────────────────────────────────────────────────────────
async def main():
    loop = asyncio.get_running_loop()
    print("=" * 60)
    print("OranextEYE — dynamic model runner")
    print(f"  Polling DB every {POLL_INTERVAL}s for enabled models")
    print(f"  Supported models: {', '.join(MODEL_FN.keys())}")
    print("=" * 60)

    try:
        while True:
            enabled_rows = fetch_enabled_models()
            enabled_keys = {(r["model_name"], r["camera_stream"]) for r in enabled_rows}

            # Start newly enabled models
            for row in enabled_rows:
                key = (row["model_name"], row["camera_stream"])
                with _lock:
                    already = key in _running
                if not already:
                    await start_model(row, loop)

            # Stop disabled models
            with _lock:
                running_keys = set(_running.keys())
            for key in running_keys - enabled_keys:
                stop_model(key)

            # Report status
            with _lock:
                active = list(_running.keys())
            if active:
                print(f"[{time.strftime('%H:%M:%S')}] Running: {[f'{k[0]}[{k[1]}]' for k in active]}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] No models enabled — waiting…")

            await asyncio.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down…")
        with _lock:
            keys = list(_running.keys())
        for key in keys:
            stop_model(key)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("✅ Done")
