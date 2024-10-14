import utils
from wp import WPLinear
import torch
import numpy as np
import random
from tqdm import tqdm
from net import PerturbNet, BPNet

def run() -> None:
    seed = 42
    # Initializing random seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    dataset = "MNIST"
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer_type = "sgd"
    learning_rate = 1e-6 #1e-2 for CCE, 1e-6 for MSE
    algorithm = "BP"
    sigma = 1e-3
    distribution = "normal"
    nb_epochs = 100
    loss_func = "mse"

    # Load dataset
    train_loader, test_loader, in_shape, out_shape = utils.construct_dataloaders(
        dataset, batch_size, device
    )
    in_shape = np.prod(in_shape)

    if algorithm.lower() == "wp":
        dist_sampler = utils.make_dist_sampler(
            sigma,
            distribution,
            device)
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            WPLinear(in_shape, out_shape, clean_pass=True, dist_sampler=dist_sampler, sample_wise=False),
        ).to(device)
    elif algorithm.lower() == "bp":
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_shape, out_shape),
        ).to(device)

    # Initialize metric storage
    metrics = utils.init_metric()


    # Define optimizers
    fwd_optimizer = None
    if optimizer_type.lower() == "adam":
        fwd_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
    elif optimizer_type.lower() == "sgd":
        fwd_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    if loss_func.lower() == "cce":
        loss_obj = torch.nn.CrossEntropyLoss(reduction="none")
        loss_func = lambda input, target, onehot: loss_obj(input, target)
    elif loss_func.lower() == "mse":
        loss_obj = torch.nn.MSELoss(reduction="none")
        loss_func = lambda input, target, onehot: loss_obj(input, onehot).mean(axis=1).float()
    
    
    if algorithm.lower() == "bp":
        network = BPNet(model)
    elif algorithm.lower() == "wp":
        network = PerturbNet(model)

    for e in tqdm(range(nb_epochs)):
        metrics = utils.update_metrics(
            network,
            metrics,
            device,
            "train",
            train_loader,
            loss_func,
            e,
            loud=True,
            num_classes=out_shape,
        )

        metrics = utils.update_metrics(
            network,
            metrics,
            device,
            "test",
            test_loader,
            loss_func,
            e,
            loud=False,
            num_classes=out_shape,
        )


        if e < nb_epochs:
            utils.train(
                network,
                device,
                train_loader,
                fwd_optimizer,
                e,
                loss_func,
                loud=False,
                num_classes=out_shape,
            )
        if np.isnan(metrics["test"]["loss"][-1]) or np.isnan(
            metrics["train"]["loss"][-1]
        ):
            print("NaN detected, aborting training")
            break

if __name__ == "__main__":
    run()