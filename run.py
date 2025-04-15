"""Training task and network selection, main loop"""

import os
import random
import numpy as np
from tqdm import tqdm
import torch
import utils
import wandb
import hydra
from wp import WPLinear
from net import PerturbNet, BPNet
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base="1.3", config_path="config/", config_name="config")
def run(config) -> None:
    cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    wandb.init(
        config=cfg,
        entity=config.entity,
        project=config.project,
        name=config.name,
        mode=config.mode,
    )
    print(cfg)
    torch.set_printoptions(precision=10)

    if isinstance(config.learning_rate, float) or isinstance(config.learning_rate, int):
        lr = config.learning_rate
    else:
        lr = eval(config.learning_rate)

    if isinstance(config.sigma, float) or isinstance(config.sigma, int):
        sigma = config.sigma
    else:
        sigma = eval(config.sigma)

    if isinstance(config.meta_learning_rate, float) or isinstance(
        config.meta_learning_rate, int
    ):
        meta_learning_rate = config.meta_learning_rate
    else:
        meta_learning_rate = eval(config.meta_learning_rate)

    if isinstance(config.mu_scaling_factor, float) or isinstance(
        config.mu_scaling_factor, int
    ):
        mu_scaling_factor = config.mu_scaling_factor
    else:
        mu_scaling_factor = eval(config.mu_scaling_factor)

    if config.mu_scaling_factor == 0 and "meta" in config.algorithm:
        mu_scaling_factor = lr
        print("here")

    # Initializing random seeding
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    device = torch.device(config.device)

    # Load dataset
    train_loader, test_loader, in_shape, out_shape = utils.construct_dataloaders(
        config.dataset, config.batch_size, device, validation=config.validation
    )
    in_shape = np.prod(in_shape)  # for linear networks

    # Define network
    network = None
    if "ffd" in config.algorithm.lower() or "cfd" in config.algorithm.lower():
        dist_sampler = utils.make_dist_sampler(
            config.distribution,
            device,
        )

        if config.num_layers == 1:
            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                WPLinear(
                    in_shape,
                    out_shape,
                    bias=config.bias,
                    pert_type=config.algorithm,
                    dist_sampler=dist_sampler,
                    sigma=sigma,
                    num_perts=config.num_perts,
                    device=config.device,
                    zero_masking=config.zero_masking,
                    orthogonal_perts=config.orthogonal_perts,
                    mu_scaling_factor=mu_scaling_factor,
                    meta_lr=meta_learning_rate,
                ),
            ).to(device)
            model_bp = (
                torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(in_shape, out_shape),
                ).to(device)
                if config.comp_angles
                else None
            )
        elif config.num_layers == 3:
            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                WPLinear(
                    in_shape,
                    500,
                    bias=config.bias,
                    pert_type=config.algorithm,
                    dist_sampler=dist_sampler,
                    sigma=sigma,
                    num_perts=config.num_perts,
                    device=config.device,
                    zero_masking=config.zero_masking,
                    orthogonal_perts=config.orthogonal_perts,
                    mu_scaling_factor=mu_scaling_factor,
                    meta_lr=meta_learning_rate,
                ),
                torch.nn.ReLU(),
                WPLinear(
                    500,
                    500,
                    bias=config.bias,
                    pert_type=config.algorithm,
                    dist_sampler=dist_sampler,
                    sigma=sigma,
                    num_perts=config.num_perts,
                    device=config.device,
                    zero_masking=config.zero_masking,
                    orthogonal_perts=config.orthogonal_perts,
                    mu_scaling_factor=mu_scaling_factor,
                    meta_lr=meta_learning_rate,
                ),
                torch.nn.ReLU(),
                WPLinear(
                    500,
                    out_shape,
                    bias=config.bias,
                    pert_type=config.algorithm,
                    dist_sampler=dist_sampler,
                    sigma=sigma,
                    num_perts=config.num_perts,
                    device=config.device,
                    zero_masking=config.zero_masking,
                    orthogonal_perts=config.orthogonal_perts,
                    mu_scaling_factor=mu_scaling_factor,
                    meta_lr=meta_learning_rate,
                ),
            ).to(device)

            model_bp = (
                torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(in_shape, 500),
                    torch.nn.ReLU(),
                    torch.nn.Linear(500, 500),
                    torch.nn.ReLU(),
                    torch.nn.Linear(500, out_shape),
                ).to(device)
                if config.comp_angles
                else None
            )

        network = PerturbNet(
            network=model,
            num_perts=config.num_perts,
            pert_type=config.algorithm,
            BP_network=model_bp,
        )

    elif config.algorithm.lower() == "bp":
        config.comp_angles = (
            False  # BP networks do not need to compare angles with BP updates
        )
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_shape, out_shape),
        ).to(device)
        network = BPNet(model)

    network.to(torch.double)
    # Initialize metric storage
    metrics = utils.init_metric(config.comp_angles)

    regular_weights = []
    meta_weights = []

    # If comp angles, optimizer should update both BP and WP model, so we need to iterate through network.named_params

    param_list = (
        network.named_parameters() if config.comp_angles else model.named_parameters()
    )

    for name, param in param_list:
        if name.endswith("mu") or name.endswith("sigma"):
            meta_weights.append(param)
        else:
            regular_weights.append(param)

    # Define optimizers
    fwd_optimizer = None

    if config.optimizer_type.lower() == "adam":
        fwd_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config.optimizer_type.lower() == "sgd":
        fwd_optimizer = torch.optim.SGD(
            regular_weights,
            lr,
            momentum=config.momentum,
            dampening=config.momentum if config.dampening else 0,
            nesterov=config.nesterov,
        )

    # Choose Loss function
    if config.loss_func.lower() == "cce":
        loss_obj = torch.nn.CrossEntropyLoss(reduction="none")
        loss_func = lambda input, target, onehot: loss_obj(input, target)
    elif config.loss_func.lower() == "mse":
        loss_obj = torch.nn.MSELoss(reduction="none")
        loss_func = (
            lambda input, target, onehot: loss_obj(input, onehot).mean(axis=1).float()
        )
    else:
        raise ValueError()

        # measuring speed of one pass
        # flops.FLOP_step_track(config.dataset, network, device, out_shape, loss_func, config.algorithm, config.num_perts)    if "ffd" in config.algorithm.lower():
    """ 
    if "ffd" in config.algorithm.lower():
        network.load_state_dict(
            torch.load("/home/markra/outputs/32_FFD.pth", weights_only=True)
        )
    """
    # main training loop
    with tqdm(range(config.nb_epochs)) as t:
        for e in t:
            metrics = utils.next_epoch(
                network,
                metrics,
                device,
                fwd_optimizer,
                test_loader,
                train_loader,
                loss_func,
                e,
                config.bias,
                loud_test=config.loud_test,
                loud_train=config.loud_train,
                comp_angles=config.comp_angles,
                validation=config.validation,
                num_classes=out_shape,
            )

            if np.isnan(metrics["test"]["loss"][-1]) or np.isnan(
                metrics["train"]["loss"][-1]
            ):
                print("NaN detected, aborting training")
                break

            if e == 115 and config.save_model:
                torch.save(network.state_dict(), "/home/markra/outputs/32_FFD.pth")
                print("model saved, quitting")
                break

            ### Early stopping below here
            """   

            if config.validation and (
                e > 10 and metrics["test"]["acc"][-1] < 12
            ):  # early stopping, but only when not testing.
                print(
                    "Network is not learning fast enough, or has too high of a loss, aborting training"
                )
                break

            if config.validation and (
                (e > 20 and metrics["test"]["acc"][-1] < 15)
                or metrics["test"]["loss"][-1] > 2.8
            ):  # early stopping, but only when not testing.
                print(
                    "Network is not learning fast enough, or has too high of a loss, aborting training"
                )
                break
            if config.validation and (
                (e > 40 and metrics["test"]["acc"][-1] < 20)
                or metrics["test"]["loss"][-1] > 2.8
            ):  # early stopping, but only when not testing.
                print(
                    "Network is not learning fast enough, or has too high of a loss, aborting training"
                )
                break
            if config.validation and (
                e > 50 and metrics["test"]["acc"][-1] < 24
            ):  # early stopping, but only when not testing.
                print(
                    "Network is not learning fast enough, or has too high of a loss, aborting training"
                )
                break
            if config.validation and (
                (e == 100) and metrics["test"]["acc"][-1] < 29
            ):  # early stopping, but only when not testing.
                print(
                    "Network is not learning fast enough, or has too high of a loss, aborting training"
                )
                break
            """
    if config.comp_angles:
        wandb.log(
            {
                "angle/mean angle": np.mean(metrics["angle"]),
                "angle/std angle": np.std(metrics["angle"]),
            }
        )
    wandb.log({"Runtime": round(t.format_dict["elapsed"], 1)})
    wandb.finish()


if __name__ == "__main__":
    run()
