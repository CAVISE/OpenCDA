"""Tests for distributed intermediate-model execution boundaries."""

import sys

import pytest

mocked_opencood = sys.modules.get("opencood")
if mocked_opencood is not None and not hasattr(mocked_opencood, "__path__"):
    pytest.skip(
        "full OpenCOOD tests are disabled when the lightweight OpenCDA test doubles are active",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")

from opencood.models.fuse_modules.where2comm_fuse import Communication  # noqa: E402
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone  # noqa: E402


@pytest.mark.parametrize("compression", [0, 1])
def test_multiscale_backbone_split_matches_centralized_path(compression):
    model_config = {
        "layer_nums": [0, 0],
        "layer_strides": [1, 2],
        "num_filters": [4, 8],
        "upsample_strides": [1, 2],
        "num_upsample_filter": [4, 4],
        "compression": compression,
    }
    backbone = AttBEVBackbone(model_config, input_channels=4).eval()
    spatial_features = (
        torch.arange(
            2 * 4 * 8 * 8,
            dtype=torch.float32,
        ).reshape(2, 4, 8, 8)
        / 100
    )
    record_len = torch.tensor([2], dtype=torch.long)

    with torch.no_grad():
        centralized = backbone(
            {
                "spatial_features": spatial_features.clone(),
                "record_len": record_len,
            }
        )["spatial_features_2d"]
        encoded_features = backbone.encode_agent(spatial_features.clone())
        distributed = backbone.fuse_agents(encoded_features, record_len)

    torch.testing.assert_close(distributed, centralized)
    if compression:
        assert encoded_features[0].shape == (2, 2, 4, 4)
        assert encoded_features[0].numel() < 2 * 4 * 8 * 8


def test_where2comm_rate_is_measured_before_local_feature_is_forced():
    communication = Communication({"threshold": 0.5}).eval()
    confidence_map = torch.tensor(
        [
            [[[10.0, -10.0]]],
            [[[-10.0, 10.0]]],
        ],
        dtype=torch.float32,
    )

    transmitted_mask, transmitted_rate = communication.build_mask(
        confidence_map,
        force_first=False,
    )
    receiver_mask, receiver_rate = communication.build_mask(
        confidence_map,
        force_first=True,
    )

    torch.testing.assert_close(
        transmitted_mask,
        torch.tensor(
            [
                [[[1.0, 0.0]]],
                [[[0.0, 1.0]]],
            ]
        ),
    )
    torch.testing.assert_close(
        receiver_mask,
        torch.tensor(
            [
                [[[1.0, 1.0]]],
                [[[0.0, 1.0]]],
            ]
        ),
    )
    assert transmitted_rate.item() == pytest.approx(0.5)
    assert receiver_rate.item() == pytest.approx(0.5)
