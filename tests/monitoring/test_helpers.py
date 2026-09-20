from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


HELPERS_PATH = Path(__file__).resolve().parents[2] / "monitoring" / "helpers.py"
HELPERS_SPEC = spec_from_file_location("monitoring_helpers", HELPERS_PATH)
helpers = module_from_spec(HELPERS_SPEC)
assert HELPERS_SPEC.loader is not None
HELPERS_SPEC.loader.exec_module(helpers)

infer_counter_flag = helpers.infer_counter_flag
normalize_endpoint = helpers.normalize_endpoint


def test_normalize_endpoint_maps_legacy_routes_to_current_route():
    assert normalize_endpoint("/object-count") == "/object-detection"
    assert normalize_endpoint("/object-list") == "/object-detection"
    assert normalize_endpoint("/object-detection") == "/object-detection"


def test_infer_counter_flag_uses_legacy_endpoint_when_counter_missing():
    assert infer_counter_flag({"counter": None, "endpoint": "/object-count"}) is True
    assert infer_counter_flag({"counter": None, "endpoint": "/object-list"}) is False