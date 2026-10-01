"""Real-world pilot study on the laptop webcam: a guided, keyboard-driven test window.

Runs the production engine, liveness session (VerificationTask) and presence guard on the live camera.
Nothing is locked (the guard runs in log-only mode) and no image is saved: each trial appends one JSON line
to results/realworld/trials.jsonl; owner and phone-attack trials also store a per-frame numeric trace
(yaw, parallax, blink/smile scores, 5 landmarks) in results/realworld/traces/. Volunteer trials store only
decisions, scores and timings.

    ..\\.venv\\Scripts\\python.exe exp_realworld.py        (close the FaceGuard app first: it holds the camera)
"""

from __future__ import annotations

import json
import random
import time
import uuid
from datetime import datetime

import cv2
import numpy as np

from common import RESULTS

from faceguard.config import Settings
from faceguard.recognition.database import FaceDatabase
from faceguard.recognition.engine import EngineOptions, FaceEngine
from faceguard.security import winsys
from faceguard.security.guard import PresenceGuard, guard_faces
from faceguard.security.liveness import LivenessConfig, VerificationTask
from faceguard.security.runner import config_from_settings
from faceguard.vision.camera import open_source
from faceguard.vision.geometry import parallax_yaw

try:
    import winsound
except ImportError:  # pragma: no cover - Windows only
    winsound = None

OUT = RESULTS / "realworld"
TRACES = OUT / "traces"
LOG = OUT / "trials.jsonl"
WIN = "FaceGuard real-world test"
PROTOCOL = "v2"  # v1 = pilot (results/realworld/pilot_trials_protocol_v1.jsonl); v2 = continuity check <=20 deg, blink 0.30/0.20
CAM_W, CAM_H = 640, 480  # this webcam: 10 fps at 1280x720 but 20 fps at 640x480 (blinks need the frame rate)
W, H = 1280, 720

PHONE_HELP = [
    "Hold the phone with YOUR photo in front of the webcam, about where your face would be.",
    "Hide your own face (e.g. behind the phone or out of view).",
    "Follow the prompts as well as a photo can: TILT/TURN the phone towards the requested side,",
    "move it, bend your wrist - try your best to fool it.",
]
UNLOCK = {
    "1": dict(key="owner_live", title="You: live face unlock (full protocol)", subject="owner", n=30,
              cfg=LivenessConfig(), expr=True, trace=True,
              help=["Sit normally and follow the prompts on screen.",
                    "Vary things a little between trials: distance, posture, glasses, room light."]),
    "t": dict(key="owner_turns", title="You: live face unlock, head turns only", subject="owner", n=15,
              cfg=LivenessConfig(), expr=False, trace=True,
              help=["Sit normally and follow the head-turn prompts."]),
    "2": dict(key="phone_full", title="Attack: phone photo vs full protocol", subject="owner", n=20,
              cfg=LivenessConfig(), expr=True, trace=True, help=PHONE_HELP),
    "3": dict(key="phone_turns", title="Attack: phone photo vs turn-only protocol (parallax gate ON)", subject="owner",
              n=20, cfg=LivenessConfig(), expr=False, trace=True, help=PHONE_HELP),
    "4": dict(key="phone_turns_noparallax", title="Attack: phone photo vs turn-only protocol (parallax gate OFF)",
              subject="owner", n=20, cfg=LivenessConfig(parallax_fraction=None), expr=False, trace=True, help=PHONE_HELP),
    "5": dict(key="volunteer_live", title="Volunteer: face unlock (not enrolled = impostor)", subject="volunteer", n=10,
              cfg=LivenessConfig(), expr=True, trace=False,
              help=["The VOLUNTEER sits in front of the laptop and follows the prompts honestly.",
                    "The owner stays out of the camera view."]),
}
GUARD = {
    "6": dict(key="guard_desk", title="Guard: you work at the desk (10 min)", subject="owner", n=1, minutes=10.0,
              help=["Work normally at the PC for 10 minutes (type, read, look around).",
                    "Stay in your seat. Esc ends the block early."]),
    "7": dict(key="guard_walkaway", title="Guard: walk away", subject="owner", n=5,
              help=["Sit normally and press SPACE.", "At the BEEP get up and walk out of the camera view.",
                    "Stay away until you hear TWO beeps (about 35-45 s), then come back."]),
    "8": dict(key="guard_intruder", title="Guard: stranger sits down", subject="volunteer", n=5,
              help=["Owner sits and presses SPACE; at the BEEP the owner leaves.",
                    "About 5 s later the VOLUNTEER sits down and looks at the screen until two beeps."]),
    "9": dict(key="guard_shoulder", title="Guard: shoulder-surfer", subject="volunteer", n=5,
              help=["Owner sits and presses SPACE and keeps working.",
                    "At the BEEP the VOLUNTEER stands behind the owner, looking at the screen, until two beeps."]),
    "0": dict(key="guard_lockwatch", title="Guard: lock-screen watch (simulated lock)", subject="volunteer", n=3,
              help=["Nothing is really locked. Owner presses SPACE; at the BEEP the owner leaves.",
                    "About 5 s later the VOLUNTEER sits down and looks at the screen until two beeps."]),
}


# ------------------------------------------------------------------ small UI helpers
def beep(n=1):
    for _ in range(n):
        if winsound:
            winsound.Beep(1200, 180)
        time.sleep(0.12)


def canvas(frame=None, mirror=True):
    if frame is None:
        img = np.full((H, W, 3), 32, np.uint8)
    else:
        img = cv2.resize(cv2.flip(frame, 1) if mirror else frame, (W, H))
    return img


def text(img, s, y, scale=0.8, color=(255, 255, 255), x=30, bold=2, band=False):
    if band:
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, bold)
        cv2.rectangle(img, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)


LAST_KEY = {"k": None}


def dbg(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "debug.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}" + "\n")


class WindowClosed(Exception):
    pass


def show(img):
    if LAST_KEY["k"] is not None:
        text(img, f"last key: {LAST_KEY['k']}", H - 12, 0.5, (160, 160, 160), x=W - 200, bold=1)
    cv2.imshow(WIN, img)
    k = cv2.waitKey(1) & 0xFF
    if k != 255:
        LAST_KEY["k"] = {32: "SPACE", 27: "Esc"}.get(k, chr(k) if 32 < k < 127 else str(k))
        dbg(f"key {LAST_KEY['k']}")
    try:
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            raise WindowClosed
    except cv2.error:
        raise WindowClosed
    return k


def counts():
    c = {}
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            k = (r["block"], r.get("subject", "owner"))
            c[k] = c.get(k, 0) + 1
    return c


def log_trial(rec: dict, trace: list | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    rec["id"] = uuid.uuid4().hex[:10]
    rec["when"] = datetime.now().isoformat(timespec="seconds")
    if trace is not None:
        TRACES.mkdir(parents=True, exist_ok=True)
        (TRACES / f"{rec['id']}.json").write_text(json.dumps(trace), encoding="utf-8")
        rec["trace"] = f"traces/{rec['id']}.json"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ test app
class RealWorldTest:
    def __init__(self):
        self.s = Settings.load()
        self.db = FaceDatabase.open()
        owner = self.db.owner()
        if owner is None or owner.count() == 0:
            raise SystemExit("No owner is enrolled in FaceGuard - enroll first.")
        self.owner_name = owner.name
        self.engine = FaceEngine.create(self.s)
        self.engine.set_gallery(self.db.snapshot())
        self.source = open_source(self.s.camera_index, CAM_W, CAM_H).start()
        self.mirror = bool(self.s.mirror_preview)
        self.consented: set[str] = set()
        f, _ = self.frame()
        self.cam_info = f"{getattr(self.source, 'backend', '') or self.source.name}, {f.shape[1]}x{f.shape[0]}"
        dbg(f"start: owner={self.owner_name} camera={self.cam_info} threshold={self.engine.matcher.threshold:.2f}")

    def frame(self):
        f, ts = self.source.read(1.0)
        while f is None:
            f, ts = self.source.read(1.0)
        return f, ts

    def wait_key(self, lines, keys, frame=True, title=None):
        """Show the live view with instruction lines until one of `keys` is pressed; returns the key."""
        while True:
            f = self.frame()[0] if frame else None
            img = canvas(f, self.mirror)
            if f is not None:
                img = (img * 0.45).astype(np.uint8)
            y = 60
            if title:
                text(img, title, y, 1.0, (120, 200, 255), bold=2)
                y += 50
            for ln in lines:
                text(img, ln, y, 0.72)
                y += 36
            k = show(img)
            if k in keys:
                return k

    def pick_volunteer(self):
        k = self.wait_key(["Which volunteer is taking part? Press 1, 2 or 3.", "(Esc = back)"], {ord("1"), ord("2"), ord("3"), 27},
                          title="Volunteer")
        if k == 27:
            return None
        vid = f"V{chr(k)}"
        if vid not in self.consented:
            k = self.wait_key([
                f"Volunteer {vid}: consent", "",
                "This test checks whether FaceGuard rejects people who are not enrolled.",
                "Your face is processed only on this laptop. NO image or video is saved.",
                "Saved: pass/fail decisions, similarity scores and timings, labelled only as " + vid + ".",
                "The anonymous results may be reported in a research paper. You can stop at any time.", "",
                "Press Y if you agree to take part, N if not.",
            ], {ord("y"), ord("Y"), ord("n"), ord("N")}, frame=False, title="Consent")
            if k in (ord("n"), ord("N")):
                return None
            self.consented.add(vid)
        return vid

    # -------------------------------------------------------------- face-unlock trials
    def unlock_block(self, b):
        subject = "owner"
        if b["subject"] == "volunteer":
            subject = self.pick_volunteer()
            if subject is None:
                return
        expr = b["expr"] and self.engine.ensure_mesh()
        done = counts().get((b["key"], subject), 0)
        while True:
            k = self.wait_key(b["help"] + ["", f"Trials done: {done}/{b['n']}.   SPACE = start next trial,   Esc = back to menu"],
                              {32, 27}, title=b["title"])
            if k == 27:
                return
            rec, trace = self.unlock_trial(b, expr, subject)
            if rec is not None:
                log_trial(rec, trace if b["trace"] and subject == "owner" else None)
                done += 1

    def unlock_trial(self, b, expr, subject):
        seed = random.SystemRandom().randrange(1 << 30)
        task = VerificationTask(self.engine.gallery, self.engine.matcher, b["cfg"], expr, False, random.Random(seed))
        self.engine.reset()
        trace, bright, faces_px, max_par, n = [], [], [], 0.0, 0
        t0 = None
        state = None
        left_align = False
        while True:
            f, ts = self.frame()
            t0 = ts if t0 is None else t0
            res = self.engine.process(f, ts, EngineOptions(mesh=expr, embed_interval=0.35))
            state = task.update(res)
            n += 1
            left_align = left_align or state.phase not in ("align", "failed")
            p = res.primary
            par = None
            if p is not None and task.session.base_landmarks is not None:
                par = float(parallax_yaw(task.session.base_landmarks, p.landmarks))
                max_par = max(max_par, abs(par))
            if p is not None:
                faces_px.append(float(p.face_px))
            bright.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
            if b["trace"]:
                trace.append({"t": round(ts - t0, 3), "faces": len(res.faces), "phase": state.phase,
                              "challenge": state.challenge.value if state.challenge else None,
                              "yaw": None if p is None else round(float(p.pose.yaw), 2),
                              "parallax": None if par is None else round(par, 2),
                              "blink": None if (p is None or p.mesh is None) else round(float(p.mesh.blink), 3),
                              "smile": None if (p is None or p.mesh is None) else round(float(p.mesh.smile), 3),
                              "meter": round(float(state.meter), 3), "fresh": bool(p.fresh) if p else False,
                              "lm": None if p is None else np.round(p.landmarks, 1).tolist()})
            img = canvas(f, self.mirror)
            sx, sy = W / f.shape[1], H / f.shape[0]
            for face in res.faces:
                x, y, w, h = (float(v) for v in face.bbox)
                if self.mirror:
                    x = f.shape[1] - x - w
                cv2.rectangle(img, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)), (90, 200, 90), 2)
            if p is None:
                diag = "no face detected - sit in front of the camera, light on your face"
            elif state.phase == "align":
                diag = f"waiting for a frontal face: yaw {p.pose.yaw:+.0f} deg (needs within +-{b['cfg'].center_degrees:.0f})"
            else:
                diag = f"phase {state.phase}  yaw {p.pose.yaw:+.0f}  parallax {0.0 if par is None else par:+.0f}  faces {len(res.faces)}"
            text(img, diag, H - 110, 0.6, (200, 230, 255), band=True)
            text(img, state.instruction or "", 70, 1.15, (255, 255, 255), band=True)
            if b["key"].startswith("phone"):
                text(img, "ATTACK TRIAL: only the PHONE may be visible - keep your real face hidden", H - 150, 0.65,
                     (60, 160, 255), band=True)
            if state.hint:
                text(img, state.hint, 130, 0.9, (80, 220, 255), band=True)
            cv2.rectangle(img, (30, H - 60), (30 + int((W - 60) * state.meter), H - 35), (90, 200, 90), -1)
            cv2.rectangle(img, (30, H - 60), (W - 30, H - 35), (220, 220, 220), 2)
            text(img, f"{b['title']}  |  step {state.step}/{state.total}  |  {state.time_left:4.1f} s   (Esc = abort trial)",
                 H - 80, 0.6, (230, 230, 230))
            if show(img) == 27:
                dbg(f"trial {b['key']} aborted by Esc in phase {state.phase} after {ts - t0:.1f}s")
                return None, None
            if state.finished:
                break
        r = task.result
        m = r.match if r else None
        dbg(f"trial {b['key']} {subject}: {state.phase} ({state.reason}) after {state.elapsed:.1f}s")
        stamps = [x["t"] for x in trace] if trace else []
        rec = {"block": b["key"], "subject": subject, "protocol": PROTOCOL, "camera": f"{CAM_W}x{CAM_H}",
               "fps": round(1.0 / float(np.median(np.diff(stamps))), 1) if len(stamps) > 2 else None,
               "seed": seed, "expressions": bool(expr),
               "parallax_gate": b["cfg"].parallax_fraction is not None,
               "sequence": [c.value for c in task.session.sequence], "completed": [c.value for c in state.completed],
               "phase": state.phase, "reason": state.reason, "reached_challenge": left_align, "live": bool(r and r.live), "verified": bool(r and r.verified),
               "similarity": None if (m is None or m.similarity is None) else round(float(m.similarity), 4),
               "matched_owner": bool(m and m.is_match and m.person and m.person.is_owner),
               "duration": round(float(state.elapsed), 2), "frames": n, "strikes": int(task.session.strikes),
               "max_abs_parallax": round(max_par, 2),
               "brightness_mean": round(float(np.mean(bright)), 1),
               "face_px_median": round(float(np.median(faces_px)), 1) if faces_px else None}
        ok = rec["verified"] if b["subject"] == "owner" and not b["key"].startswith("phone") else not rec["verified"]
        msg = ("PASSED" if rec["verified"] else ("LIVE, NOT RECOGNISED" if rec["live"] else "FAILED")) + f"  ({state.reason or 'ok'})"
        img = canvas(None)
        text(img, msg, 200, 1.0, (120, 255, 120) if ok else (90, 90, 255))
        text(img, f"duration {rec['duration']:.1f} s   similarity {rec['similarity']}", 260, 0.75)
        show(img)
        cv2.waitKey(1500)
        if b["key"].startswith("phone"):
            k = self.wait_key(["Was ONLY the phone visible during the whole trial (your real face hidden)?", "",
                               "Y = yes, it was a clean attack       N = no, my face was visible (trial is discarded)"],
                              {ord("y"), ord("Y"), ord("n"), ord("N")}, frame=False, title="Check the attack")
            rec["clean_presentation"] = k in (ord("y"), ord("Y"))
            if not rec["clean_presentation"]:
                dbg("attack trial marked NOT clean by the user - discarded")
                return None, None
        return rec, trace

    # -------------------------------------------------------------- guard trials (log-only)
    def guard_block(self, b):
        subject = "owner"
        if b["subject"] == "volunteer":
            subject = self.pick_volunteer()
            if subject is None:
                return
        done = counts().get((b["key"], subject), 0)
        while True:
            k = self.wait_key(b["help"] + ["", f"Trials done: {done}/{b['n']}.   SPACE = start,   Esc = back to menu",
                                           "(Log-only: the PC is NOT locked; decisions are only recorded.)"],
                              {32, 27}, title=b["title"])
            if k == 27:
                return
            rec = self.guard_trial(b, subject)
            if rec is not None:
                log_trial(rec)
                done += 1

    def guard_trial(self, b, subject):
        kind = b["key"]
        guard = PresenceGuard(config_from_settings(self.s))
        self.engine.reset()
        period = 1.0 / max(0.5, self.s.guard_fps)
        t_start = self.frame()[1]
        guard.start(t_start)
        locked_sim = False
        t_event = None  # beep: owner leaves / volunteer enters
        if kind == "guard_desk":
            beep(1)
            t_event = t_start
        frames, labels, actions = 0, {}, []
        t_auth, longest_gap = t_start, 0.0
        first_stranger = first_pair = None
        target = {"guard_walkaway": ("lock", "absence"), "guard_intruder": ("lock", "intruder"),
                  "guard_shoulder": ("notify", "shoulder"), "guard_lockwatch": ("snapshot", "locked")}.get(kind)
        t_hit = None
        last = -1e9
        limit = b.get("minutes", 0) * 60 if kind == "guard_desk" else 90.0
        while True:
            f, ts = self.frame()
            if t_event is None and ts - t_start >= 3.0:
                t_event = ts
                beep(1)
                if kind == "guard_lockwatch":
                    guard.on_session_locked(ts)
                    locked_sim = True
            img = canvas(f, self.mirror)
            if ts - last >= period:
                last = ts
                res = self.engine.process(f, ts, EngineOptions(mesh=False, embed_interval=0.2))
                faces = guard_faces(res)
                frames += 1
                lab = res.primary.match.status if res.primary else "no_face"
                labels[lab] = labels.get(lab, 0) + 1
                if any(g.authorized for g in faces):
                    longest_gap = max(longest_gap, ts - t_auth) if t_auth is not None else longest_gap
                    t_auth = ts
                if t_event is not None and first_stranger is None and any(not g.authorized for g in faces):
                    first_stranger = ts
                if t_event is not None and first_pair is None and any(g.authorized for g in faces) and \
                        any(not g.authorized for g in faces):
                    first_pair = ts
                if locked_sim:
                    acts = guard.update_locked(ts, faces)
                else:
                    acts = guard.update(ts, faces, winsys.idle_seconds(), True)
                for a in acts:
                    actions.append({"t": round(ts - t_start, 2), "kind": a.kind, "reason": a.reason})
                    if target and t_hit is None and (a.kind, a.reason) == target and t_event is not None:
                        t_hit = ts
                        beep(2)
            elapsed = ts - t_start
            text(img, b["title"], 50, 0.9, (120, 200, 255), band=True)
            text(img, f"guard: {guard.status_text(ts)}", 100, 0.7, band=True)
            text(img, f"{elapsed:5.1f} s" + ("" if t_event is None else f"   since beep {ts - t_event:5.1f} s"), 145, 0.7, band=True)
            if t_hit is not None:
                text(img, "Decision recorded - you can come back / stop.", 190, 0.75, (120, 255, 120), band=True)
            k = show(img)
            if k == 27 and kind != "guard_desk":
                return None
            done = (k == 27) or elapsed >= limit or (t_hit is not None and ts - t_hit > 2.0)
            if done:
                break
        if kind == "guard_desk":
            beep(2)
        ref = {"guard_walkaway": t_event, "guard_intruder": first_stranger, "guard_lockwatch": first_stranger,
               "guard_shoulder": first_pair}.get(kind)
        return {"block": kind, "subject": subject, "protocol": PROTOCOL, "camera": f"{CAM_W}x{CAM_H}", "duration_s": round(elapsed, 1), "frames": frames, "labels": labels,
                "actions": actions, "longest_gap_s": round(longest_gap, 2),
                "t_event_s": None if t_event is None else round(t_event - t_start, 2),
                "t_reference_s": None if ref is None else round(ref - t_start, 2),
                "latency_s": None if (t_hit is None or ref is None) else round(t_hit - ref, 2),
                "detected": t_hit is not None,
                "intruder_locks": sum(a["kind"] == "lock" and a["reason"] == "intruder" for a in actions),
                "absence_locks": sum(a["kind"] == "lock" and a["reason"] == "absence" for a in actions),
                "shoulder_alerts": sum(a["kind"] == "notify" and a["reason"] == "shoulder" for a in actions)}

    # -------------------------------------------------------------- menu
    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1100, 620)
        try:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        while True:
            c = counts()
            lines = [f"Owner: {self.owner_name}.  Pick a block (number/letter key), Q = quit.", ""]
            for key, b in list(UNLOCK.items()) + list(GUARD.items()):
                if b["subject"] == "owner":
                    got = c.get((b["key"], "owner"), 0)
                    prog = f"{got}/{b['n']}"
                else:
                    prog = ", ".join(f"{v}: {c.get((b['key'], v), 0)}/{b['n']}" for v in ("V1", "V2", "V3")
                                     if c.get((b["key"], v), 0)) or f"0/{b['n']} per volunteer"
                lines.append(f"[{key}]  {b['title']}   -   {prog}")
            lines += ["", f"Camera: {self.cam_info}.  Click this window once if the keys do nothing."]
            k = self.wait_key(lines, {ord(x) for x in "1234567890tTqQ"}, frame=True, title="FaceGuard real-world test")
            dbg(f"menu -> {chr(k)}")
            ch = chr(k).lower()
            if ch == "q":
                break
            if ch in UNLOCK:
                self.unlock_block(UNLOCK[ch])
            elif ch in GUARD:
                self.guard_block(GUARD[ch])
        self.source.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import traceback
    try:
        RealWorldTest().run()
    except WindowClosed:
        dbg("window closed")
    except BaseException:
        dbg("CRASH " + traceback.format_exc())
        raise
