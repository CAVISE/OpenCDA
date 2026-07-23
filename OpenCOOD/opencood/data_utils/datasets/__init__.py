from importlib import import_module


_DATASET_REGISTRY = {
    "LateFusionDataset": (
        "opencood.data_utils.datasets.late_fusion_dataset",
        "LateFusionDataset",
    ),
    "EarlyFusionDataset": (
        "opencood.data_utils.datasets.early_fusion_dataset",
        "EarlyFusionDataset",
    ),
    "IntermediateFusionDataset": (
        "opencood.data_utils.datasets.intermediate_fusion_dataset",
        "IntermediateFusionDataset",
    ),
    "IntermediateFusionDatasetV2": (
        "opencood.data_utils.datasets.intermediate_fusion_dataset_v2",
        "IntermediateFusionDatasetV2",
    ),
}

__all__ = tuple(_DATASET_REGISTRY)

# the final range for evaluation
GT_RANGE = [-140.8, -41.6, -10.6, 140.8, 41.6, 10.6]
# The communication range for cavs
COM_RANGE = 70


def _load_dataset_class(dataset_name):
    module_name, class_name = _DATASET_REGISTRY[dataset_name]
    dataset_class = getattr(import_module(module_name), class_name)
    globals()[dataset_name] = dataset_class
    return dataset_class


def __getattr__(name):
    if name not in _DATASET_REGISTRY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _load_dataset_class(name)


def build_dataset(dataset_cfg, visualize=False, train=True, payload_handler=None):
    dataset_name = dataset_cfg["fusion"]["core_method"]
    error_message = f"{dataset_name} is not found. Please add the dataset to opencood.data_utils.datasets"
    assert dataset_name in _DATASET_REGISTRY, error_message

    dataset_class = _load_dataset_class(dataset_name)
    return dataset_class(params=dataset_cfg, visualize=visualize, train=train, payload_handler=payload_handler)
