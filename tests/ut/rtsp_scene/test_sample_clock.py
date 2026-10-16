"""Host-only checks for the clock shared with the hardware decoder."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class SampleClockTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "C++ compiler unavailable")
    def test_missing_pts_slots_are_counted_without_catch_up_samples(self):
        source = r"""
#include "sample_clock.h"
#include <cassert>
#include <limits>
int main() {
    rtsp_scene::SampleClock clock(1);
    assert(clock.select(0));
    assert(!clock.select(.04));
    assert(clock.select(1.04));
    assert(clock.select(4.01));
    assert(clock.missed() == 2);
    assert(!clock.select(4.05));
    assert(clock.select(5.0));
    bool rejected = false;
    try { clock.select(4.9); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { rtsp_scene::SampleClock invalid(std::numeric_limits<double>::infinity()); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
"""
        include = Path(__file__).resolve().parents[3] / "tools/rtsp_scene"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test.cpp").write_text(source)
            subprocess.run(
                ["g++", "-std=c++17", "-I", str(include), str(root / "test.cpp"), "-o", str(root / "test")],
                check=True,
                capture_output=True,
            )
            subprocess.run([str(root / "test")], check=True, capture_output=True)
