"""Unit tests for the folder watcher's scan logic.

Uses a fake detector so the loop's behaviour (settle-skip, report writing, and
success/failure archiving) is tested without loading a real model or weights.
"""
import time
from pathlib import Path

import numpy as np
import cv2
import pytest

from src.watcher import watch


class _FakeResult:
    def __init__(self):
        self.verdict = "PASS"
        self.n_defects = 0
        self.inference_time_ms = 1.0
        self.annotated_image = np.zeros((4, 4, 3), dtype=np.uint8)
        self.image_path = None

    def to_dict(self):
        return {"verdict": self.verdict, "n_defects": self.n_defects,
                "inference_time_ms": self.inference_time_ms,
                "image_path": self.image_path}


class _FakeDetector:
    def __init__(self, boom=False):
        self.boom = boom

    def detect(self, img, annotate=False):
        if self.boom:
            raise RuntimeError("inference blew up")
        return _FakeResult()


def _make_dirs(tmp_path):
    inbox = tmp_path / "inbox"
    output = tmp_path / "out"
    inbox.mkdir()
    return inbox, output, tmp_path / "processed", tmp_path / "failed"


def _drop_image(inbox: Path, name="board.png"):
    p = inbox / name
    cv2.imwrite(str(p), np.zeros((8, 8, 3), dtype=np.uint8))
    # backdate mtime so the settle check treats it as fully written
    old = time.time() - 10
    import os
    os.utime(p, (old, old))
    return p


def test_success_writes_report_and_archives(tmp_path):
    inbox, output, processed, failed = _make_dirs(tmp_path)
    _drop_image(inbox)
    ok, fail = watch.scan_once(_FakeDetector(), inbox, output, processed, failed)
    assert (ok, fail) == (1, 0)
    assert (output / "board_report.json").exists()
    assert (output / "board_annotated.png").exists()
    assert (processed / "board.png").exists()      # original archived on success
    assert not (inbox / "board.png").exists()       # and removed from the inbox


def test_failure_routes_to_failed_folder(tmp_path):
    inbox, output, processed, failed = _make_dirs(tmp_path)
    _drop_image(inbox, "bad.png")
    ok, fail = watch.scan_once(_FakeDetector(boom=True), inbox, output, processed, failed)
    assert (ok, fail) == (0, 1)
    assert (failed / "bad.png").exists()            # routed to failed/, not lost
    assert not (output / "bad_report.json").exists()


def test_unsettled_file_is_skipped(tmp_path):
    inbox, output, processed, failed = _make_dirs(tmp_path)
    p = inbox / "fresh.png"
    cv2.imwrite(str(p), np.zeros((8, 8, 3), dtype=np.uint8))  # just written -> "settling"
    ok, fail = watch.scan_once(_FakeDetector(), inbox, output, processed, failed,
                               settle_seconds=5.0)
    assert (ok, fail) == (0, 0)
    assert (inbox / "fresh.png").exists()           # left in place for the next pass


def test_non_image_files_ignored(tmp_path):
    inbox, output, processed, failed = _make_dirs(tmp_path)
    (inbox / "notes.txt").write_text("not an image")
    ok, fail = watch.scan_once(_FakeDetector(), inbox, output, processed, failed)
    assert (ok, fail) == (0, 0)
    assert (inbox / "notes.txt").exists()


def test_archive_never_clobbers(tmp_path):
    inbox, output, processed, failed = _make_dirs(tmp_path)
    processed.mkdir()
    (processed / "board.png").write_bytes(b"existing")   # a prior archive with same name
    _drop_image(inbox)
    watch.scan_once(_FakeDetector(), inbox, output, processed, failed)
    names = sorted(p.name for p in processed.iterdir())
    assert names == ["board.png", "board_1.png"]         # both kept, no overwrite
