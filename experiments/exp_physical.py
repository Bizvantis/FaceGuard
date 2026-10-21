"""Physical study on the laptop webcam: bona fide sessions of several users, presentation attacks with several
instruments, passive PAD baselines, tracked-feature parallax and a continuous guard session with scripted events.

Everything is log-only (nothing is locked) and no image or video is stored: each trial appends one JSON line to
results/physical/trials.jsonl and a numeric per-frame trace (head angles, the 5 landmarks, tracked-feature
planarity, PAD scores) to results/physical/traces/. Volunteers are enrolled in memory only; their template is
discarded when the program ends or another user is selected.

    ..\\.venv\\Scripts\\python.exe exp_physical.py [--camera N]      (close the FaceGuard app first)

Order of the study (see results/physical/README in the paper's reproducibility notes):
  1. owner: bona fide full + turn-only sessions and a static hold        -> calibrates the flow gate (owner only)
  2. exp_physical_analysis.py --freeze                                    -> freezes the flow-gate threshold
  3. attacks (gate condition randomised per trial), volunteers' bona fide sessions (held-out users), guard sessions
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import uuid
from datetime import datetime

import cv2
import numpy as np

from common import RESULTS
from pad_models import heavy_models, light_models

from faceguard.config import Settings
from faceguard.recognition.database import FaceDatabase
from faceguard.recognition.engine import EngineOptions, FaceEngine
from faceguard.recognition.enrollment import EnrollmentSession
from faceguard.recognition.gallery import Gallery, PersonInfo
from faceguard.security import winsys
from faceguard.security.guard import PresenceGuard, guard_faces
from faceguard.security.liveness import LivenessConfig, VerificationTask
from faceguard.security.runner import config_from_settings
from faceguard.vision.camera import open_source
from faceguard.vision.flow import PlanarityTracker
from faceguard.vision.geometry import parallax_yaw

try:
    import winsound
except ImportError:  # pragma: no cover - Windows only
    winsound = None

OUT = RESULTS / "physical"
TRACES = OUT / "traces"
LOG = OUT / "trials.jsonl"
FROZEN = OUT / "frozen_thresholds.json"
WIN = "FaceGuard physical study"
PROTOCOL = "v2"
W, H = 1280, 720

PAI = {
    "p": ("phone_photo", "PHONE showing a PHOTO of the user (tilt / turn / bend the wrist as the prompts ask)"),
    "s": ("screen_photo", "SECOND SCREEN (tablet / laptop / monitor) showing a PHOTO of the user"),
    "v": ("phone_video", "PHONE playing VIDEO clips of the user: play the clip that matches each prompt"),
    "w": ("screen_video", "SECOND SCREEN playing VIDEO clips of the user: play the clip matching each prompt"),
    "u": ("phone_video_loop", "PHONE looping ONE video in which the user does all actions (no clip switching)"),
    "f": ("print_flat", "PRINTED photo held FLAT (tilt / turn it as the prompts ask)"),
    "c": ("print_curved", "PRINTED photo CURVED around a bottle or can (face bulging towards the camera)"),
    "b": ("print_folded", "PRINTED photo FOLDED vertically along the nose line (a shallow V)"),
    "x": ("other", "Another instrument - note what it was"),
}
GATES = ("none", "landmark", "flow")


# ------------------------------------------------------------------ small UI helpers
def beep(n=1, hz=1200):
    for _ in range(n):
        if winsound:
            winsound.Beep(hz, 180)
        time.sleep(0.12)


def canvas(frame=None, mirror=True):
    if frame is None:
        return np.full((H, W, 3), 32, np.uint8)
    return cv2.resize(cv2.flip(frame, 1) if mirror else frame, (W, H))


def text(img, s, y, scale=0.8, color=(255, 255, 255), x=30, bold=2, band=False):
    if band:
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, bold)
        cv2.rectangle(img, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)


def dbg(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "debug.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}\n")


class WindowClosed(Exception):
    pass


def show(img):
    cv2.imshow(WIN, img)
    k = cv2.waitKey(1) & 0xFF
    if k != 255:
        dbg(f"key {k}")
    try:
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            raise WindowClosed
    except cv2.error:
        raise WindowClosed
    return k


def records():
    if not LOG.exists():
        return []
    return [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines() if x.strip()]


def log_trial(rec: dict, trace: dict | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    rec["id"] = uuid.uuid4().hex[:10]
    rec["when"] = datetime.now().isoformat(timespec="seconds")
    if trace is not None:
        TRACES.mkdir(parents=True, exist_ok=True)
        (TRACES / f"{rec['id']}.json").write_text(json.dumps(trace), encoding="utf-8")
        rec["trace"] = f"traces/{rec['id']}.json"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def frozen():
    return json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else None


# ------------------------------------------------------------------ PAD scoring in background threads
class PadWorker(threading.Thread):
    """Scores the most recent submitted frame with each of its models, as fast as it can (frames may be skipped)."""

    def __init__(self, models):
        super().__init__(daemon=True)
        self.models = models
        self.cv = threading.Condition()
        self.item = None
        self.seq = self.done = 0
        self.out: list = []
        self.stopped = False

    def submit(self, frame, bbox, t):
        with self.cv:
            self.item = (frame, np.array(bbox, np.float32), t)
            self.seq += 1
            self.cv.notify()

    def take(self):
        with self.cv:
            out, self.out = self.out, []
            return out

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify()

    def run(self):
        while True:
            with self.cv:
                while (self.item is None or self.seq == self.done) and not self.stopped:
                    self.cv.wait(0.5)
                if self.stopped:
                    return
                frame, bbox, t = self.item
                self.done = self.seq
            for m in self.models:
                try:
                    p, ms = m.timed(frame, bbox)
                except Exception as e:  # pragma: no cover - logged, never fatal during a session
                    dbg(f"PAD {m.name} failed: {e!r}")
                    p, ms = float("nan"), 0.0
                with self.cv:
                    self.out.append([round(float(t), 3), m.name, round(float(p), 4), round(float(ms), 1)])


# ------------------------------------------------------------------ study app
class PhysicalStudy:
    def __init__(self, camera: int | None):
        self.s = Settings.load()
        self.db = FaceDatabase.open()
        owner = self.db.owner()
        if owner is None or owner.count() == 0:
            raise SystemExit("No owner is enrolled in FaceGuard - enroll first.")
        self.owner_name = owner.name
        self.engine = FaceEngine.create(self.s)
        self.cam_index = self.s.camera_index if camera is None else camera
        self.source = open_source(self.cam_index, 640, 480).start()
        self.mirror = bool(self.s.mirror_preview)
        f, _ = self.frame()
        self.cam_info = f"camera {self.cam_index}: {getattr(self.source, 'backend', '') or self.source.name}, {f.shape[1]}x{f.shape[0]}"
        self.pads = [PadWorker(light_models()), PadWorker(heavy_models(threads=2))]
        for w in self.pads:
            w.start()
        self.user = None  # "owner" or "V1".."V9"
        self.gallery = None
        self.consented: set[str] = set()
        dbg(f"start: owner={self.owner_name} {self.cam_info} threshold={self.engine.matcher.threshold:.2f} "
            f"pad={[m.name for w in self.pads for m in w.models]}")

    # -------------------------------------------------------------- basics
    def frame(self):
        f, ts = self.source.read(1.0)
        while f is None:
            f, ts = self.source.read(1.0)
        return f, ts

    def wait_key(self, lines, keys, frame=True, title=None):
        while True:
            f = self.frame()[0] if frame else None
            img = canvas(f, self.mirror)
            if f is not None:
                img = (img * 0.45).astype(np.uint8)
            y = 60
            if title:
                text(img, title, y, 1.0, (120, 200, 255))
                y += 50
            for ln in lines:
                text(img, ln, y, 0.68)
                y += 33
            k = show(img)
            if k in keys:
                return k

    def pick(self, prompt, title, allow_owner=True):
        keys = {ord(str(i)) for i in range(1, 10)} | {27} | ({ord("o"), ord("O")} if allow_owner else set())
        k = self.wait_key(prompt + ["", "(Esc = back)"], keys, title=title)
        if k == 27:
            return None
        return "owner" if chr(k).lower() == "o" else f"V{chr(k)}"

    def pad_flush(self):
        for w in self.pads:
            w.take()

    def pad_collect(self):
        out = []
        for w in self.pads:
            out += w.take()
        return out

    # -------------------------------------------------------------- users
    def select_user(self):
        who = self.pick(["Who is the USER now (the enrolled person)?",
                         "O = the owner (" + self.owner_name + ", already enrolled)",
                         "1-9 = volunteer V1..V9 (enrolled now, in memory only)"], "Select the user")
        if who is None:
            return
        if who == "owner":
            self.gallery = self.db.snapshot()
        else:
            if not self.consent(who):
                return
            g = self.enroll(who)
            if g is None:
                return
            self.gallery = g
        self.user = who
        self.engine.set_gallery(self.gallery)
        dbg(f"user -> {who}")

    def consent(self, vid):
        if vid in self.consented:
            return True
        k = self.wait_key([
            f"Volunteer {vid}: consent", "",
            "This study tests whether FaceGuard recognises its user, rejects photos and videos, and locks the PC",
            "when the user leaves. Your face is processed only on this laptop. NO image or video is saved.",
            "Your face template is kept in memory only and erased when this program ends or the user changes.",
            f"Saved, labelled only as {vid}: decisions, similarity scores, timings, head angles, 5 facial",
            "landmark positions and motion measurements per frame (numbers only).",
            "The anonymous results may be reported in a research paper. You can stop at any time.", "",
            "Press Y if you agree to take part, N if not.",
        ], {ord("y"), ord("Y"), ord("n"), ord("N")}, frame=False, title="Consent")
        if k in (ord("y"), ord("Y")):
            self.consented.add(vid)
            dbg(f"consent {vid}")
            return True
        return False

    def enroll(self, vid):
        self.engine.set_gallery(Gallery([], [], version=-int(time.time())))
        self.engine.reset()
        sess = EnrollmentSession(vid)
        while True:
            f, ts = self.frame()
            res = self.engine.process(f, ts)
            st = sess.update(f, res, self.engine, ts)
            img = canvas(f, self.mirror)
            text(img, f"Enrolling {vid} (in memory only): {st.instruction}", 60, 0.95, band=True)
            if st.hint:
                text(img, st.hint, 115, 0.8, (80, 220, 255), band=True)
            cv2.rectangle(img, (30, H - 60), (30 + int((W - 60) * st.progress), H - 35), (90, 200, 90), -1)
            cv2.rectangle(img, (30, H - 60), (W - 30, H - 35), (220, 220, 220), 2)
            text(img, "S = skip this pose,  Esc = cancel", H - 80, 0.6)
            k = show(img)
            if k == 27:
                return None
            if k in (ord("s"), ord("S")):
                sess.skip_current(ts)
            if st.phase == "done":
                break
        r = sess.finish()
        emb = np.stack([s.embedding for s in r.samples])
        dbg(f"enrolled {vid}: {len(r.samples)} samples, consistency {r.consistency:.3f}")
        return Gallery([PersonInfo(f"study-{vid}", vid, is_owner=True)], [emb], version=int(time.time()))

    # -------------------------------------------------------------- liveness trials
    def counts(self, **match):
        return sum(all(r.get(k) == v for k, v in match.items()) for r in records())

    def next_gate(self, block, pai=None):
        fz = frozen()
        options = [g for g in GATES if g != "flow" or fz]
        if block == "genuine_turns":
            options = ["landmark", "flow"] if fz and self.user != "owner" else ["landmark"]
        done = {g: self.counts(block=block, user=self.user, pai=pai, gate=g) for g in options}
        low = min(done.values())
        return random.choice([g for g in options if done[g] == low])

    def config_for(self, gate):
        fz = frozen()
        if gate == "none":
            return LivenessConfig(parallax_fraction=None)
        if gate == "flow":
            return LivenessConfig(parallax_fraction=None, flow_threshold=float(fz["flow_threshold"]))
        return LivenessConfig()

    def genuine_block(self, turns_only):
        block = "genuine_turns" if turns_only else "genuine_full"
        target = 10
        while True:
            done = self.counts(block=block, user=self.user)
            k = self.wait_key([f"USER {self.user}: sit normally and follow the prompts honestly.",
                               "Vary a little between trials: distance, posture, glasses, room light.", "",
                               f"Done: {done}/{target}.   SPACE = next trial,   Esc = back"], {32, 27},
                              title="Bona fide: " + ("head turns only" if turns_only else "full protocol"))
            if k == 27:
                return
            gate = self.next_gate(block) if turns_only else "landmark"
            cfg = self.config_for(gate)
            expr = (not turns_only) and self.engine.ensure_mesh()
            rec, trace = self.unlock_trial(cfg, expr, dict(block=block, gate=gate, pai=None, operator=self.user,
                                                            media=None, attack=False))
            if rec is not None:
                log_trial(rec, trace)

    def attack_block(self):
        k = self.wait_key([f"Attack on USER {self.user}. Which instrument?"] +
                          [f"{key.upper()} = {desc}" for key, (name, desc) in PAI.items()] + ["", "(Esc = back)"],
                          {ord(c) for c in "psvwufcbxPSVWUFCBX"} | {27}, title="Attack instrument")
        if k == 27:
            return
        pai = PAI[chr(k).lower()][0]
        media = self.pick(["Which photo / video of the user is shown? Press its number 1-9",
                           "(use different photos and videos across trials if you have them)"], "Media", False)
        if media is None:
            return
        operator = self.pick(["Who holds the instrument (the ATTACKER)?", "O = owner, 1-9 = volunteer"], "Operator")
        if operator is None:
            return
        while True:
            k = self.wait_key([
                f"Instrument: {pai}, media {media}, operator {operator}.",
                "The USER's real face must NOT be visible. Only the instrument faces the webcam.",
                "Follow the prompts as well as the instrument allows - tilt, turn, bend, switch clips - try hard.", "",
                f"Trials so far with this instrument: turn-only {self.counts(block='attack_turns', user=self.user, pai=pai)}, "
                f"full {self.counts(block='attack_full', user=self.user, pai=pai)}",
                "T = next trial vs the TURN-ONLY protocol (gate chosen at random, not shown)",
                "F = next trial vs the FULL protocol (turns + blink/smile)", "Esc = back"],
                {ord("t"), ord("T"), ord("f"), ord("F"), 27}, title="Presentation attack")
            if k == 27:
                return
            turns_only = chr(k).lower() == "t"
            block = "attack_turns" if turns_only else "attack_full"
            gate = self.next_gate(block, pai) if turns_only else "landmark"
            cfg = self.config_for(gate)
            expr = (not turns_only) and self.engine.ensure_mesh()
            rec, trace = self.unlock_trial(cfg, expr, dict(block=block, gate=gate, pai=pai, operator=operator,
                                                            media=media, attack=True))
            if rec is not None:
                log_trial(rec, trace)

    def unlock_trial(self, cfg, expr, meta):
        seed = random.SystemRandom().randrange(1 << 30)
        task = VerificationTask(self.gallery, self.engine.matcher, cfg, expr, False, random.Random(seed))
        self.engine.reset()
        tracker = PlanarityTracker()
        fz = frozen()
        stat = fz["flow_statistic"] if fz else "rms"
        frames, n, t0, key, max_par = [], 0, None, None, 0.0
        self.pad_flush()
        while True:
            f, ts = self.frame()
            t0 = ts if t0 is None else t0
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            res = self.engine.process(f, ts, EngineOptions(mesh=expr, embed_interval=0.35))
            p = res.primary
            pl = tracker.update(gray) if tracker.active else None
            flow_val = None if pl is None else getattr(pl, stat)
            state = task.update(res, planarity=flow_val)
            n += 1
            new_key = (state.phase, state.step)
            if state.phase == "challenge" and new_key != key and p is not None:
                tracker.start(gray, p.bbox, p.landmarks)
            elif state.phase != "challenge":
                tracker.reset()
            key = new_key
            par = None
            if p is not None and task.session.base_landmarks is not None:
                par = float(parallax_yaw(task.session.base_landmarks, p.landmarks))
                max_par = max(max_par, abs(par))
            if p is not None:
                self.pads[0].submit(f, p.bbox, ts - t0)
                self.pads[1].submit(f, p.bbox, ts - t0)
            frames.append({
                "t": round(ts - t0, 3), "faces": len(res.faces), "phase": state.phase, "step": state.step,
                "challenge": state.challenge.value if state.challenge else None,
                "yaw": None if p is None else round(float(p.pose.yaw), 2),
                "parallax": None if par is None else round(par, 2),
                "blink": None if (p is None or p.mesh is None) else round(float(p.mesh.blink), 3),
                "smile": None if (p is None or p.mesh is None) else round(float(p.mesh.smile), 3),
                "meter": round(float(state.meter), 3), "fresh": bool(p.fresh) if p else False,
                "sim": None if (p is None or p.match.similarity is None) else round(float(p.match.similarity), 4),
                "px": None if p is None else round(p.face_px, 1),
                "bright": round(float(gray.mean()), 1),
                "lm": None if p is None else np.round(p.landmarks, 1).tolist(),
                "flow": None if pl is None else [pl.n, round(pl.rms, 5), round(pl.p75, 5),
                                                 round(pl.outlier_frac, 4), round(pl.shift, 4)]})
            img = canvas(f, self.mirror)
            text(img, state.instruction or "", 70, 1.15, band=True)
            if state.hint:
                text(img, state.hint, 130, 0.9, (80, 220, 255), band=True)
            if meta["attack"]:
                text(img, "ATTACK TRIAL: only the instrument may be visible", H - 150, 0.65, (60, 160, 255), band=True)
            cv2.rectangle(img, (30, H - 60), (30 + int((W - 60) * state.meter), H - 35), (90, 200, 90), -1)
            cv2.rectangle(img, (30, H - 60), (W - 30, H - 35), (220, 220, 220), 2)
            text(img, f"{meta['block']}  |  step {state.step}/{state.total}  |  {state.time_left:4.1f} s   (Esc = abort trial)",
                 H - 80, 0.6, (230, 230, 230))
            if show(img) == 27:
                dbg(f"trial {meta['block']} aborted in phase {state.phase}")
                return None, None
            if state.finished:
                break
        time.sleep(0.4)  # let the PAD threads finish the last frames
        pad = self.pad_collect()
        r = task.result
        m = r.match if r else None
        stamps = [x["t"] for x in frames]
        rec = {**meta, "user": self.user, "protocol": PROTOCOL, "camera": self.cam_info, "seed": seed,
               "expressions": bool(expr), "flow_threshold": cfg.flow_threshold, "flow_statistic": stat,
               "fps": round(1.0 / float(np.median(np.diff(stamps))), 1) if len(stamps) > 2 else None,
               "sequence": [c.value for c in task.session.sequence], "completed": [c.value for c in state.completed],
               "phase": state.phase, "reason": state.reason, "live": bool(r and r.live),
               "verified": bool(r and r.verified),
               "similarity": None if (m is None or m.similarity is None) else round(float(m.similarity), 4),
               "duration": round(float(state.elapsed), 2), "frames": n, "strikes": int(task.session.strikes),
               "max_abs_parallax": round(max_par, 2)}
        dbg(f"trial {meta['block']} {self.user} gate={meta['gate']} pai={meta['pai']}: {state.phase} ({state.reason})")
        ok = (not rec["verified"]) if meta["attack"] else rec["verified"]
        msg = ("PASSED" if rec["verified"] else ("LIVE, NOT RECOGNISED" if rec["live"] else "FAILED")) + f"  ({state.reason or 'ok'})"
        img = canvas(None)
        text(img, msg, 200, 1.0, (120, 255, 120) if ok else (90, 90, 255))
        text(img, f"duration {rec['duration']:.1f} s   similarity {rec['similarity']}", 260, 0.75)
        show(img)
        cv2.waitKey(1200)
        if meta["attack"]:
            k = self.wait_key(["Was ONLY the instrument visible during the whole trial (no real face)?", "",
                               "Y = yes, clean attack       N = no, a real face was visible (trial is discarded)"],
                              {ord("y"), ord("Y"), ord("n"), ord("N")}, frame=False, title="Check the attack")
            if k in (ord("n"), ord("N")):
                dbg("attack marked NOT clean - discarded")
                return None, None
            rec["clean_presentation"] = True
        return rec, {"frames": frames, "pad": pad}

    # -------------------------------------------------------------- static hold (camera characterisation)
    def static_block(self):
        k = self.wait_key([f"USER {self.user}: sit still and look at the camera for 10 seconds.", "",
                           "SPACE = start,  Esc = back"], {32, 27}, title="Static hold (landmark and flow noise)")
        if k == 27:
            return
        self.engine.reset()
        self.pad_flush()
        tracker = PlanarityTracker()
        frames, t0 = [], None
        while True:
            f, ts = self.frame()
            t0 = ts if t0 is None else t0
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            res = self.engine.process(f, ts, EngineOptions(embed_interval=0.35))
            p = res.primary
            pl = tracker.update(gray) if tracker.active else None
            if p is not None and not tracker.active and ts - t0 > 1.0:
                tracker.start(gray, p.bbox, p.landmarks)
            if p is not None:
                self.pads[0].submit(f, p.bbox, ts - t0)
                self.pads[1].submit(f, p.bbox, ts - t0)
            frames.append({"t": round(ts - t0, 3), "faces": len(res.faces),
                           "yaw": None if p is None else round(float(p.pose.yaw), 2),
                           "lm": None if p is None else np.round(p.landmarks, 2).tolist(),
                           "px": None if p is None else round(p.face_px, 1), "bright": round(float(gray.mean()), 1),
                           "flow": None if pl is None else [pl.n, round(pl.rms, 5), round(pl.p75, 5),
                                                            round(pl.outlier_frac, 4), round(pl.shift, 4)]})
            img = canvas(f, self.mirror)
            text(img, f"Hold still ... {max(0.0, 11.0 - (ts - t0)):4.1f} s", 70, 1.1, band=True)
            if show(img) == 27:
                return
            if ts - t0 >= 11.0:
                break
        time.sleep(0.4)
        log_trial({"block": "static", "user": self.user, "camera": self.cam_info, "attack": False,
                   "frames": len(frames)}, {"frames": frames, "pad": self.pad_collect()})
        beep(2)

    # -------------------------------------------------------------- continuous guard session
    def guard_script(self, helper: bool):
        rng = random.Random()
        ev = [("walk_left", "Stand up and walk away to your LEFT, out of the camera view.", 40, False),
              ("walk_right", "Stand up and walk away to your RIGHT, out of the camera view.", 40, False),
              ("walk_behind", "Stand up and walk away BEHIND you, out of the room or area.", 40, False),
              ("walk_far", "Stand up and stand about 3 METRES behind the chair, facing the laptop.", 40, False),
              ("early_return", "Stand up and walk away. Come back at the double beep (short break).", 15, False),
              ("occlude_hand", "Stay seated and COVER YOUR FACE with your hands.", 15, False),
              ("look_down", "Stay seated and look DOWN at your phone (do not touch the laptop).", 20, False),
              ("lights", "Switch the ROOM LIGHT OFF (or ask the helper). Switch it on at the double beep.", 20, False)]
        if helper:
            ev += [("bystander_pass", "HELPER: walk slowly past BEHIND the user, without stopping.", 10, True),
                   ("shoulder", "HELPER: stand behind the user and LOOK AT THE SCREEN.", 15, True),
                   ("replacement", "USER: leave. HELPER: sit down at once and use the laptop.", 30, True)]
        rng.shuffle(ev)
        return ev

    def guard_session(self):
        helper = self.wait_key(["A ~15-minute session of normal work with scripted events (log-only, nothing locks).",
                                "Is a HELPER (second person) here for the bystander, shoulder-surfer and",
                                "replacement events?   Y = yes,  N = no (those events are skipped)", "", "Esc = back"],
                               {ord("y"), ord("Y"), ord("n"), ord("N"), 27}, title="Continuous guard session")
        if helper == 27:
            return
        helper = helper in (ord("y"), ord("Y"))
        helper_id = self.pick(["Who is the HELPER?", "O = owner, 1-9 = volunteer"], "Helper") if helper else None
        script = self.guard_script(helper)
        guard = PresenceGuard(config_from_settings(self.s))
        self.engine.reset()
        period = 1.0 / max(0.5, self.s.guard_fps)
        t_start = self.frame()[1]
        guard.start(t_start)
        sim_locked = False
        frames, actions, events = [], [], []
        last = -1e9
        rng = random.Random()
        # timeline: work 30-60 s, event, return/recover 15 s, ...
        plan, t = [], 45.0
        for name, cue, dur, needs_helper in script:
            plan.append({"event": name, "cue": cue, "start": t, "end": t + dur, "helper": needs_helper})
            t += dur + 15.0 + rng.uniform(30.0, 60.0)
        total = t
        i_ev, cued, ended = 0, False, False
        beep(1)
        while True:
            f, ts = self.frame()
            el = ts - t_start
            cur = plan[i_ev] if i_ev < len(plan) else None
            if cur and not cued and el >= cur["start"]:
                cued = True
                cur["t_cue"] = round(el, 2)
                beep(1)
                dbg(f"guard event {cur['event']} cue at {el:.1f}")
            if cur and cued and not ended and el >= cur["end"]:
                ended = True
                cur["t_end"] = round(el, 2)
                beep(2)
            if cur and ended and el >= cur["end"] + 15.0:
                events.append(cur)
                i_ev, cued, ended = i_ev + 1, False, False
            if ts - last >= period:
                last = ts
                res = self.engine.process(f, ts, EngineOptions(mesh=False, embed_interval=0.2))
                faces = guard_faces(res)
                auth = any(g.authorized for g in faces)
                if sim_locked:
                    acts = guard.update_locked(ts, faces)
                    if auth:  # the user is back: they would type their PIN now
                        guard.on_session_unlocked(ts)
                        sim_locked = False
                        actions.append({"t": round(el, 2), "kind": "unlock", "reason": "user_back"})
                else:
                    acts = guard.update(ts, faces, winsys.idle_seconds(), True)
                for a in acts:
                    actions.append({"t": round(el, 2), "kind": a.kind, "reason": a.reason})
                    dbg(f"guard action {a.kind}/{a.reason} at {el:.1f}")
                    if a.kind == "lock" and not sim_locked:
                        guard.on_session_locked(ts)
                        sim_locked = True
                        beep(3, 800)
                frames.append([round(el, 2), len(res.faces), int(auth),
                               sum(g.stranger for g in faces), sum((not g.authorized and not g.stranger) for g in faces),
                               None if res.primary is None else round(float(res.primary.match.similarity or 0), 3),
                               None if res.primary is None else round(res.primary.face_px, 1),
                               round(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()), 1),
                               round(float(winsys.idle_seconds()), 1), int(sim_locked)])
            img = canvas(f, self.mirror)
            if cur and cued and not ended:
                text(img, cur["cue"], 70, 0.85, (80, 220, 255), band=True)
                text(img, f"Event {i_ev + 1}/{len(plan)}: {cur['event']}  ({cur['end'] - el:4.0f} s)", 120, 0.7, band=True)
            elif cur and ended:
                text(img, "Come back / return to normal work now.", 70, 0.9, (120, 255, 120), band=True)
            else:
                text(img, "Work normally at the laptop (read, type, browse).", 70, 0.9, band=True)
            text(img, f"guard: {guard.status_text(ts)}   {'(SIMULATED LOCK)' if sim_locked else ''}", H - 110, 0.7, band=True)
            text(img, f"{el:6.0f} / {total:4.0f} s     S = skip this event,  Esc = end session (saved)", H - 60, 0.6, band=True)
            k = show(img)
            if k in (ord("s"), ord("S")) and cur and not ended:
                cur["skipped"] = True
                cur["end"] = el
            if k == 27 or i_ev >= len(plan):
                break
        beep(2)
        log_trial({"block": "guard_session", "user": self.user, "helper": helper_id, "camera": self.cam_info,
                   "attack": False, "duration_s": round(el, 1), "events": events, "actions": actions,
                   "guard_fps": self.s.guard_fps, "absence_timeout": guard.cfg.absence_timeout},
                  {"columns": ["t", "faces", "authorized", "strangers", "uncertain", "sim", "px", "bright",
                               "idle", "locked"], "frames": frames})

    # -------------------------------------------------------------- menu
    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1100, 620)
        try:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        try:
            while True:
                fz = frozen()
                u = self.user or "none selected"
                c = lambda b: self.counts(block=b, user=self.user)  # noqa: E731
                lines = [f"USER: {u}.   Flow gate: " + (f"frozen ({fz['flow_statistic']} >= {fz['flow_threshold']:.4f})"
                                                       if fz else "not frozen yet (owner calibration first)"), "",
                         "[U]  select / enroll the user",
                         f"[1]  bona fide: full protocol            - {c('genuine_full')}/10",
                         f"[2]  bona fide: head turns only          - {c('genuine_turns')}/10",
                         f"[3]  static hold, 10 s                    - {c('static')}/2",
                         f"[A]  presentation attacks on this user   - turn-only {c('attack_turns')}, full {c('attack_full')}",
                         f"[G]  continuous guard session (~15 min)   - {c('guard_session')}/1",
                         "[Q]  quit", "", self.cam_info + ".  Click this window once if keys do nothing."]
                k = self.wait_key(lines, {ord(x) for x in "uU123aAgGqQ"}, title="FaceGuard physical study")
                ch = chr(k).lower()
                if ch == "q":
                    break
                if ch == "u":
                    self.select_user()
                    continue
                if self.user is None:
                    self.select_user()
                    if self.user is None:
                        continue
                if ch == "1":
                    self.genuine_block(False)
                elif ch == "2":
                    self.genuine_block(True)
                elif ch == "3":
                    self.static_block()
                elif ch == "a":
                    self.attack_block()
                elif ch == "g":
                    self.guard_session()
        finally:
            for w in self.pads:
                w.stop()
            self.source.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import traceback

    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args()
    try:
        PhysicalStudy(args.camera).run()
    except WindowClosed:
        dbg("window closed")
    except BaseException:
        dbg("CRASH " + traceback.format_exc())
        raise
