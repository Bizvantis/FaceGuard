import json
import time

import numpy as np

from faceguard.security.guard import GuardFace
from faceguard.security.intruders import IntruderLog

FACE = GuardFace(1, False, True, None, 0.12, 150, True)
FRAME = np.full((480, 640, 3), 90, np.uint8)
BOX = [(np.array([100, 100, 150, 150]), "Unknown", False)]


def test_record_list_delete(tmp_path):
    log = IntruderLog(tmp_path)
    ev = log.record(FRAME, "intruder", [FACE], BOX)
    assert log.path_of(ev).exists() and ev.label == "Intruder at the PC"
    events = log.events()
    assert len(events) == 1 and events[0].faces[0]["similarity"] == 0.12
    log.delete(events)
    assert log.events() == [] and not log.path_of(ev).exists()


def test_cleanup_removes_only_old_snapshots(tmp_path):
    log = IntruderLog(tmp_path)
    old = log.record(FRAME, "intruder", [FACE], BOX)
    new = log.record(FRAME, "shoulder", [FACE], BOX)
    lines = [json.loads(line) for line in log.index.read_text().splitlines()]
    lines[0]["time"] = time.time() - 40 * 86400  # pretend the first one is 40 days old
    log.index.write_text("".join(json.dumps(d) + "\n" for d in lines))
    assert log.cleanup(30) == 1
    assert [e.file for e in log.events()] == [new.file]
    assert not (tmp_path / old.file).exists()
    assert log.cleanup(0) == 0  # 0 = keep forever
