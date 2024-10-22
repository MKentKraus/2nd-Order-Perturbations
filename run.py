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


@hydra.main(version_base="1.3", config_path="", config_name="config")
def run(config) -> None:
    cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    wandb.init(
            config=cfg,
            entity=config.entity,
            project=config.project,
            name=config.name,
        )

    # Initializing random seeding
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    print(f'Setting CUDA visible devices to [{config.device}]')
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{config.device}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader, test_loader, in_shape, out_shape = utils.construct_dataloaders(
        config.dataset, config.batch_size, device
    )
    in_shape = np.prod(in_shape) #for linear networks

    #Define network
    network = None
    if config.algorithm.lower() == "wp":
        dist_sampler = utils.make_dist_sampler(
            config.sigma,
            config.distribution,
            device)
        
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            WPLinear(in_shape, out_shape, config.pert_type,
                     dist_sampler=dist_sampler, sample_wise=False),
        ).to(device)

        model_bp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_shape, out_shape),
        ).to(device)

        network = PerturbNet(model, model_bp)

    elif config.algorithm.lower() == "bp":

        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_shape, out_shape),
        ).to(device)

        network = BPNet(model)


    # Initialize metric storage
    metrics = utils.init_metric(config.comp_angles)


    # Define optimizers
    fwd_optimizer = None
    if config.optimizer_type.lower() == "adam":
        fwd_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate)
    elif config.optimizer_type.lower() == "sgd":
        fwd_optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.learning_rate)

    #Choose Loss function
    if config.loss_func.lower() == "cce":
        loss_obj = torch.nn.CrossEntropyLoss(reduction="none")
        loss_func = lambda input, target, onehot: loss_obj(input, target)
    elif config.loss_func.lower() == "mse":
        loss_obj = torch.nn.MSELoss(reduction="none")
        loss_func = lambda input, target, onehot: loss_obj(input, onehot).mean(axis=1).float()

    print("crash here?")

    for e in tqdm(range(config.nb_epochs)):
        metrics = utils.next_epoch(
                network,
                metrics,
                device,
                fwd_optimizer,
                test_loader,
                train_loader,
                loss_func,
                e,
                loud_test=config.loud_test,
                loud_train=config.loud_train,
                comp_angles=config.comp_angles,
                wandb=wandb,
                num_perts=config.num_perts,
                num_classes=out_shape,
            )


        if np.isnan(metrics["test"]["loss"][-1]) or np.isnan(
            metrics["train"]["loss"][-1]
        ):
            print("NaN detected, aborting training")
            break

    #torch.save(network.state_dict(),"2nd-Order-Perturbations/results/models/BP-MNIST-1e-6.pt")
    #np.save("2nd-Order-Perturbations/results/metrics/Metrics-BP-MNIST-1e-6.npy", metrics)

    wandb.finish()

if __name__ == "__main__":
    run()
