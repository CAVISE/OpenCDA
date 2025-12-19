import glob
import importlib
import yaml
import sys
import os
import re
from datetime import datetime

import torch
import torch.optim as optim

from typing import Dict, Any, Tuple, Optional, TypeVar, Type
from torch.optim import Optimizer
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

def load_saved_model(saved_path: str, model: nn.Module) -> Tuple[int, nn.Module]:
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), "{} not found".format(saved_path)

    def findLastCheckpoint(save_dir):
        if os.path.exists(os.path.join(saved_path, "latest.pth")):
            return 10000
        file_list = glob.glob(os.path.join(save_dir, "*epoch*.pth"))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        model_file = os.path.join(saved_path, "net_epoch%d.pth" % initial_epoch) if initial_epoch != 10000 else os.path.join(saved_path, "latest.pth")
        print("resuming by loading epoch %d" % initial_epoch)
        checkpoint = torch.load(model_file, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)

        del checkpoint

    return initial_epoch, model


def setup_train(hypes: Dict[str, Any]) -> str:
    """
    Create folder for saved model based on current timestamp and model name.
    Args:
        hypes (Dict[str, Any]): Configuration dictionary containing at least:
            - name (str): The name of the model.
    Returns:
        str: The full path to the created directory where logs will be saved.
    The function creates a directory structure: logs/{model_name}_{timestamp}/
    and saves the configuration as config.yaml in that directory.
    """
    model_name = hypes["name"]
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, "../logs")

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        # save the yaml file
        save_name = os.path.join(full_path, "config.yaml")
        with open(save_name, "w") as outfile:
            yaml.dump(hypes, outfile)

    return full_path


def create_model(hypes: Dict[str, Any]) -> nn.Module:
    """
    Dynamically import and instantiate a model based on the configuration.
    Args:
        hypes (Dict[str, Any]): Configuration dictionary containing:
            - model.core_method (str): The name of the model class.
            - model.args (Dict[str, Any]): Arguments to pass to the model constructor.
    Returns:
        nn.Module: An instance of the specified model.
    """
    backbone_name = hypes["model"]["core_method"]
    backbone_config = hypes["model"]["args"]

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace("_", "")

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print(
            "backbone not found in models folder. Please make sure you "
            "have a python file named %s and has a class "
            "called %s ignoring upper/lower case" % (model_filename, target_model_name)
        )
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes: Dict[str, Any]) -> nn.Module:
    """
    Create a loss function based on the configuration.
    Args:
        hypes (Dict[str, Any]): Configuration dictionary containing:
            - loss.core_method (str): The name of the loss class.
            - loss.args (Dict[str, Any]): Arguments for the loss constructor.
    Returns:
        nn.Module: An instance of the specified loss function.
    """
    loss_func_name = hypes["loss"]["core_method"]
    loss_func_config = hypes["loss"]["args"]

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace("_", "")

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print(
            "loss function not found in loss folder. Please make sure you "
            "have a python file named %s and has a class "
            "called %s ignoring upper/lower case" % (loss_filename, target_loss_name)
        )
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes: Dict[str, Any], model: nn.Module) -> Optimizer:
    """
    Create and configure an optimizer based on the configuration.
    Args:
        hypes (Dict[str, Any]): Configuration dictionary containing:
            - optimizer.core_method (str): The name of the optimizer class.
            - optimizer.lr (float): Learning rate.
            - optimizer.args (Dict[str, Any], optional): Additional optimizer arguments.
    Returns:
        Optimizer: Configured optimizer instance.
    """
    method_dict = hypes["optimizer"]
    optimizer_method = getattr(optim, method_dict["core_method"], None)
    print("optimizer method is: %s" % optimizer_method)

    if not optimizer_method:
        raise ValueError("{} is not supported".format(method_dict["name"]))
    if "args" in method_dict:
        return optimizer_method(filter(lambda p: p.requires_grad, model.parameters()), lr=method_dict["lr"], **method_dict["args"])
    else:
        return optimizer_method(filter(lambda p: p.requires_grad, model.parameters()), lr=method_dict["lr"])


def setup_lr_schedular(hypes: Dict[str, Any], optimizer: Optimizer, 
                      n_iter_per_epoch: int) -> Optional[_LRScheduler]:
    """
    Set up a learning rate scheduler based on the configuration.
    Args:
        hypes (Dict[str, Any]): Configuration dictionary containing:
            - lr_scheduler (Dict[str, Any]): Scheduler configuration.
        optimizer (Optimizer): The optimizer whose learning rate will be scheduled.
        n_iter_per_epoch (int): Number of iterations per training epoch.
    Returns:
        Optional[_LRScheduler]: Configured learning rate scheduler, or None if not configured.
    """
    lr_schedule_config = hypes["lr_scheduler"]

    if lr_schedule_config["core_method"] == "step":
        from torch.optim.lr_scheduler import StepLR

        step_size = lr_schedule_config["step_size"]
        gamma = lr_schedule_config["gamma"]
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config["core_method"] == "multistep":
        from torch.optim.lr_scheduler import MultiStepLR

        milestones = lr_schedule_config["step_size"]
        gamma = lr_schedule_config["gamma"]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif lr_schedule_config["core_method"] == "exponential":
        print("ExponentialLR is chosen for lr scheduler")
        from torch.optim.lr_scheduler import ExponentialLR

        gamma = lr_schedule_config["gamma"]
        scheduler = ExponentialLR(optimizer, gamma)

    elif lr_schedule_config["core_method"] == "cosineannealwarm":
        print("cosine annealing is chosen for lr scheduler")
        from timm.scheduler.cosine_lr import CosineLRScheduler

        num_steps = lr_schedule_config["epoches"] * n_iter_per_epoch
        warmup_lr = lr_schedule_config["warmup_lr"]
        warmup_steps = lr_schedule_config["warmup_epoches"] * n_iter_per_epoch
        lr_min = lr_schedule_config["lr_min"]

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        sys.exit("not supported lr schedular")

    return scheduler


def to_device(inputs: Any, device: torch.device) -> Any:
    """
    Move input tensors to the specified device.
    Args:
        inputs: Input data (tensor, list, or dict of tensors).
        device: Target device (e.g., 'cuda' or 'cpu').
    Returns:
        The input data with all tensors moved to the specified device.
    """
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) or isinstance(inputs, str):
            return inputs
        return inputs.to(device)
