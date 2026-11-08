from dsh_vts_avatar.performance.mouth_monitor import MouthMonitor


def test_mouth_monitor_detects_frozen_speaking_mouth():
    monitor = MouthMonitor(window_s=0.3, min_range=0.01)
    for index in range(10):
        monitor.observe(index * 0.05, {"MouthOpen": 0.4}, speaking=True, audible=True)
    assert monitor.stuck is True


def test_mouth_monitor_ignores_idle_and_variation():
    monitor = MouthMonitor(window_s=0.3, min_range=0.01)
    for index in range(10):
        monitor.observe(index * 0.05, {"MouthOpen": 0.4 + (index % 2) * 0.2}, speaking=True, audible=True)
    assert monitor.stuck is False
    monitor.observe(1.0, {"MouthOpen": 0.4}, speaking=False, audible=False)
    assert monitor.stuck is False
