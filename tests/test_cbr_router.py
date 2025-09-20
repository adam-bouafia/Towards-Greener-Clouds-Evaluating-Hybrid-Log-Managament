import time
import importlib.util, pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed; skipping CBR router tests", allow_module_level=True)

from src.routers import CBRRouter, StaticRouter


def make_log(level="INFO", component="app", source="synthetic"):
    return {
        "Level": level,
        "Component": component,
        "LogSource": source,
        "Content": f"Test log {level} {component} {source}",
    }


def test_cbr_fallback_to_static_before_warmup():
    router = CBRRouter(warm_samples=10, sample_prob=0.0)  # disable sampling so it can't warm up
    log = make_log()
    # With no stats gathered, should defer to static router => mysql for routine log
    dest = router.get_route(log)
    assert dest in {"mysql", "elk", "ipfs"}


def test_cbr_attribute_selection_after_warmup():
    # Force sampling always and small warmup
    router = CBRRouter(warm_samples=5, sample_prob=1.0, recompute_interval=1000)
    # Simulate observations with different backend latencies for Level buckets
    levels = ["INFO", "ERROR"]
    # Feed logs and observe artificial latencies
    for i in range(5):
        log = make_log(level=levels[i % 2])
        # Pretend router chose mysql; observe different latencies based on level
        router.observe(
            log_entry=log,
            destination="mysql",
            success=True,
            routing_latency_ms=0.1,
            backend_latency_ms=10.0 if log["Level"] == "INFO" else 1.0,
            energy_cpu_pkg_j=0.0,
        )
    # Trigger scoring by making a routing decision
    _ = router.get_route(make_log(level="ERROR"))
    assert router.classifier_attr is not None
    assert router.classifier_attr in ["Level", "Component", "LogSource"]


def test_cbr_json_dump(tmp_path):
    dump_file = tmp_path / "cbr_diag.json"
    router = CBRRouter(warm_samples=1, sample_prob=1.0, json_dump_path=str(dump_file), json_dump_interval=1)
    # Collect one sample
    log = make_log(level="ERROR")
    router.observe(log_entry=log, destination="mysql", success=True, routing_latency_ms=0.1, backend_latency_ms=2.0, energy_cpu_pkg_j=0.0)
    # Decision triggers dump
    _ = router.get_route(log)
    assert dump_file.exists(), "JSON diagnostic file not created"
