from functools import partial
from multimae.output_adapters import SpatialOutputAdapter
from multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.criterion import MaskedCrossEntropyLoss, MaskedL1Loss, MaskedMSELoss

CONFIG = {
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3),
        'loss': MaskedMSELoss,
    },
    'ired': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=3),
        'loss': MaskedMSELoss,
    },
    'sired': {
        'channels': 2,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=2),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=2),
        'loss': MaskedMSELoss,
    },
    'ebands': {
        'channels': 2,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=2),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=2),
        'loss': MaskedMSELoss,
    },
    'depth': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
        'loss': MaskedL1Loss,
    },
    'semseg': {
        'num_classes': 12,
        'stride_level': 4,
        'input_adapter': partial(SemSegInputAdapter, num_classes=12,
                        dim_class_emb=64, interpolate_class_emb=False),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=12),
        'loss': partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
    }
}
