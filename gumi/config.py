""" The GumiConfig class definition. """

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from torch.utils.data.dataset import Dataset


@dataclass
class GumiConfig:
    """
    For the meaning of these fields, please refer to model_runner.parser
    """
    dataset: Union[str, Dict[str, Dataset]] = "cifar10"
    arch: str = "resnet110"

    # Group configurations
    num_groups: int = 0
    mcpg: int = 0
    group_cfg: str = ""
    ind: Optional[str] = None

    # Mask configuration
    perm: str = "GRPS"
    num_sort_iters: int = 10

    # Regularization
    reg_scale: float = 0.0
    reg_epochs: int = 0
    reg_lr: float = 1e-3

    # Control flags
    no_weight: bool = False
    no_perm: bool = False
    no_data_loader: bool = False
    no_bar: bool = False
    scratch: bool = False

    # Paths
    checkpoint: Optional[str] = None
    dataset_dir: str = "data"

    # Training
    workers: int = 8
    epochs: int = 300
    start_epoch: int = 0
    train_batch: int = 128
    test_batch: int = 100
    lr: float = 0.1
    lr_type: str = "multistep"
    drop: float = 0
    schedule: List[int] = field(default_factory=[150, 225])
    gamma: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    gpu_id: str = '0'
    resume: str = ""
    resume_from_best: bool = False
    pretrained: bool = False

    # Misc
    print_freq: int = 50
