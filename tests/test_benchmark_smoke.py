import pytest

pytestmark = pytest.mark.smoke


def test_load_bboxes_benchmark_targets_smoke(via_tracks_csv):
    """Smoke check: benchmark targets should execute once with small inputs.
    This catches API/signature drift without running timing benchmarks in CI.
    """
    from movement.io import load_bboxes

    # calling the same functions used by benchmark tests once (no timing).
    load_bboxes.from_via_tracks_file(via_tracks_csv)
    load_bboxes._df_from_via_tracks_file(via_tracks_csv)
