"""Integration tests involving RoI workflows."""

import pytest
import shapely

from movement.napari.convert_roi import napari_shapes_layer_to_rois
from movement.napari.regions_widget import REGIONS_LAYER_KEY, RegionsWidget
from movement.roi.io import load_rois

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Previous color_by key.*:UserWarning"
)


def test_draw_save_analyse_load(make_napari_viewer_proxy, mocker, tmp_path):
    """Test the full napari-to-Python RoI workflow.

    1. Draw shapes in napari (polygon + path on a RegionsWidget layer).
    2. Save the layer to a GeoJSON file via the widget.
    3. Load the GeoJSON in Python and verify geometry/names.
    4. Use the loaded RoIs for spatial analysis (contains_point).
    5. Load the GeoJSON back into napari and verify the roundtrip.
    """
    # 1. Set up widget and draw shapes in napari
    viewer = make_napari_viewer_proxy()
    widget = RegionsWidget(viewer)

    polygon_data = [[0, 0], [0, 10], [10, 10], [10, 0]]
    path_data = [[0, 0], [5, 5], [10, 0]]
    layer = viewer.add_shapes(
        [polygon_data, path_data],
        shape_type=["polygon", "path"],
        name="test_regions",
        metadata={REGIONS_LAYER_KEY: True},
    )
    layer.properties = {"name": ["arena", "boundary"]}
    widget.layer_dropdown.setCurrentText("test_regions")

    # 2. Save regions via widget (mock only file dialog)
    geojson_path = tmp_path / "regions.geojson"
    mocker.patch(
        "movement.napari.regions_widget.QFileDialog.getSaveFileName",
        return_value=(str(geojson_path), None),
    )
    widget._save_region_layer()
    assert geojson_path.exists()

    # 3. Load in Python and verify geometry/names
    rois = load_rois(geojson_path)
    assert len(rois) == 2
    assert rois[0].name == "arena"
    assert rois[1].name == "boundary"

    original_rois = napari_shapes_layer_to_rois(layer)
    for orig, loaded in zip(original_rois, rois, strict=True):
        assert shapely.normalize(orig.region) == shapely.normalize(
            loaded.region
        )

    # 4. Use loaded RoIs in analysis
    arena = rois[0]  # the polygon
    assert arena.contains_point((5, 5))  # inside
    assert not arena.contains_point((15, 15))  # outside

    # 5. Load back into napari via widget
    mocker.patch(
        "movement.napari.regions_widget.QFileDialog.getOpenFileName",
        return_value=(str(geojson_path), None),
    )
    widget._load_region_layer()
    loaded_layer = viewer.layers["regions"]  # named after file stem
    assert len(loaded_layer.data) == 2

    # Verify geometry survives the full roundtrip
    reloaded_rois = napari_shapes_layer_to_rois(loaded_layer)
    for orig, reloaded in zip(original_rois, reloaded_rois, strict=True):
        assert shapely.normalize(orig.region) == shapely.normalize(
            reloaded.region
        )
