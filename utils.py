"""Dataset manipulation and network training functions"""

import os
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as v2


def make_dist_sampler(
    sigma: float,
    distribution: str = "normal",
    device: torch.device | str = "cpu",
    **kwargs,
) -> torch.distributions.Distribution:
    """Create a distribution sampler for the noise"""

    if distribution.lower() == "normal":
        dist_sampler = lambda x: sigma * (
            torch.empty(x, device=device).normal_(mean=0, std=1)
        )
    elif distribution.lower() == "bernoulli":
        distribution = torch.distributions.Bernoulli(
            torch.tensor([0.5]).to(torch.float32).to(device)
        )
        dist_sampler = lambda x: sigma * (distribution.sample(x).squeeze_(-1) - 0.5)
    else:
        raise ValueError(f"Distribution {distribution} not recognized")

    return dist_sampler


class ClassificationLoadedDataset(torch.utils.data.Dataset):
    """Classification Dataset in Memory"""

    def __init__(self, x, y, transform=None, return_onehot=False):
        self.x = x
        self.y = y
        self.return_onehot = return_onehot
        self.transform = transform
        self.classes = torch.unique(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample = self.transform(x_sample)

        if self.return_onehot:
            y_sample = torch.nn.functional.one_hot(y_sample, len(self.classes))

        return x_sample, y_sample


def format_tin_val(datadir):
    val_dir = datadir + "/tiny-imagenet-200/val"
    print("Formatting: %s" % val_dir)
    val_annotations = "%s/val_annotations.txt" % val_dir
    val_dict = {}
    with open(val_annotations, "r") as f:
        for line in f:
            line = line.strip().split()
            assert len(line) == 6
            wnind = line[1]
            img_name = line[0]
            boxes = "\t".join(line[2:])
            if wnind not in val_dict:
                val_dict[wnind] = []
            entries = val_dict[wnind]
            entries.append((img_name, boxes))
    assert len(val_dict) == 200
    for wnind, entries in val_dict.items():
        val_wnind_dir = "%s/%s" % (val_dir, wnind)
        val_images_dir = "%s/images" % val_dir
        val_wnind_images_dir = "%s/images" % val_wnind_dir
        os.mkdir(val_wnind_dir)
        os.mkdir(val_wnind_images_dir)
        wnind_boxes = "%s/%s_boxes.txt" % (val_wnind_dir, wnind)
        f = open(wnind_boxes, "w")
        for img_name, box in entries:
            source = "%s/%s" % (val_images_dir, img_name)
            dst = "%s/%s" % (val_wnind_images_dir, img_name)
            os.system("cp %s %s" % (source, dst))
            f.write("%s\t%s\n" % (img_name, box))
        f.close()
    os.system("rm -rf %s" % val_images_dir)
    print("Cleaning up: %s" % val_images_dir)
    print("Formatting val done")


def load_dataset(dataset_importer, device, fltype, validation, mean, std):
    if isinstance(dataset_importer, str) and dataset_importer.lower() == "tin":

        if os.path.exists("./datasets/tiny-imagenet-200/y_train.npy"):
            print("Loading TinyImageNet")
            x_train = np.load("./datasets/tiny-imagenet-200/x_train.npy")
            y_train = np.load("./datasets/tiny-imagenet-200/y_train.npy").astype(int)
            x_test = np.load("./datasets/tiny-imagenet-200/x_test.npy")
            y_test = np.load("./datasets/tiny-imagenet-200/y_test.npy").astype(int)
        else:
            print("Down-Loading TinyImageNet")
            zip_md5 = "90528d7ca1a48142e341f4ef8d21d0de"
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            torchvision.datasets.utils.download_and_extract_archive(
                url,
                "./datasets/",
                extract_root="./datasets/",
                remove_finished=True,
                md5=zip_md5,
            )

            format_tin_val("./datasets/")

            train_datasetpath = "./datasets/tiny-imagenet-200/train/"
            test_datasetpath = "./datasets/tiny-imagenet-200/val/"

            train_dataset = torchvision.datasets.ImageFolder(train_datasetpath)
            test_dataset = torchvision.datasets.ImageFolder(test_datasetpath)

            x_test = np.empty((len(test_dataset.targets), 3, 64, 64), dtype=np.float32)
            y_test = np.empty((len(test_dataset.targets)))
            for indx, (img, label) in enumerate(test_dataset.imgs):
                x_test[indx] = torchvision.transforms.ToTensor()(
                    test_dataset.loader(img).convert("RGB")
                )
                y_test[indx] = label
            x_test = np.asarray(x_test)
            y_test = np.asarray(y_test)
            print("TinyImageNet test set loaded")

            np.save("./datasets/tiny-imagenet-200/x_test.npy", x_test)
            np.save("./datasets/tiny-imagenet-200/y_test.npy", y_test)

            x_train = np.empty(
                (len(train_dataset.targets), 3, 64, 64), dtype=np.float32
            )
            y_train = np.empty((len(train_dataset.targets)))
            for indx, (img, label) in enumerate(train_dataset.imgs):
                x_train[indx] = torchvision.transforms.ToTensor()(
                    train_dataset.loader(img).convert("RGB")
                )
                y_train[indx] = label
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            print("TinyImageNet training set loaded")

            np.save("./datasets/tiny-imagenet-200/x_train.npy", x_train)
            np.save("./datasets/tiny-imagenet-200/y_train.npy", y_train)

    else:
        train_dataset = dataset_importer("./datasets/", train=True, download=True)
        test_dataset = dataset_importer("./datasets/", train=False, download=True)

        # Loading dataset
        x_train = train_dataset.data
        y_train = train_dataset.targets
        x_test = test_dataset.data
        y_test = test_dataset.targets

    # Reshaping to flat digits
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Extracting a validation, rather than test, set
    # Last 10K samples taken as test
    if validation:
        # First shuffle the data
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

        nb_train_samples = len(x_train) - 10_000

        x_test = x_train[nb_train_samples:]
        y_test = y_train[nb_train_samples:]
        x_train = x_train[:nb_train_samples]
        y_train = y_train[:nb_train_samples]

    # Squeezing out any excess dimension in the labels (true for CIFAR10/100)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test)
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train)
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test)

    # Data to device (datasets small enough to fit directly)
    x_train = x_train.to(device).type(fltype)
    y_train = y_train.type(torch.FloatTensor).to(device)

    x_test = x_test.to(device).type(fltype)
    y_test = y_test.type(torch.FloatTensor).to(device)

    maxval = torch.max(x_train)
    x_train = x_train / maxval
    x_test = x_test / maxval

    if dataset_importer == "TIN":
        x_train = x_train.reshape(-1, 3, 64, 64)
        x_test = x_test.reshape(-1, 3, 64, 64)

        means = torch.mean(x_train, axis=(0, 2, 3))[None, :, None, None]
        stds = (torch.std(x_train, axis=(0, 2, 3)) + 1e-8)[None, :, None, None]
        x_train = (x_train - means) / stds
        x_test = (x_test - means) / stds

    return x_train, y_train, x_test, y_test


def construct_dataloaders(
    dataset_name,
    batch_size=64,
    mean=None,
    std=None,
    validation=False,
    device="cpu",
):

    # Check if dataset is supported
    assert dataset_name.lower() in [
        "mnist",
        "cifar10",
        "cifar100",
        "tin",
    ], f"Dataset {dataset_name} not supported"

    # Load dataset
    if dataset_name.lower() == "mnist":
        tv_dataset = torchvision.datasets.MNIST
    elif dataset_name.lower() == "cifar10":
        tv_dataset = torchvision.datasets.CIFAR10
    elif dataset_name.lower() == "cifar100":
        tv_dataset = torchvision.datasets.CIFAR100
    elif dataset_name.lower() == "tin":
        tv_dataset = "tin"

    train_kwargs = {"batch_size": batch_size, "num_workers": 0, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "num_workers": 0, "shuffle": False}

    train_transforms, test_transforms = None, None
    if tv_dataset in (
        torchvision.datasets.CIFAR10,
        torchvision.datasets.CIFAR100,
        torchvision.datasets.MNIST,
    ):
        if tv_dataset in (torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100):
            train_transforms = v2.Compose(
                [
                    v2.RandomCrop(32, padding=4),
                    v2.RandomHorizontalFlip(),
                    v2.ToTensor(),
                    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            test_transforms = v2.Compose(
                [
                    v2.ToTensor(),
                    v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

        if tv_dataset == torchvision.datasets.MNIST:
            train_transforms = v2.Compose(
                [
                    v2.ToTensor(),
                    v2.Normalize((0.1307,), (0.3081,)),
                ]
            )

            test_transforms = v2.Compose(
                [v2.ToTensor(), v2.Normalize((0.1307,), (0.3081,))]
            )

        train_dataset = tv_dataset(
            root="./datasets", train=True, download=True, transform=train_transforms
        )
        if validation:
            train_dataset, test_dataset = torch.utils.data.random_split(
                train_dataset, [len(train_dataset) - 10_000, 10_000]
            )
        else:
            test_dataset = tv_dataset(
                root="./datasets", train=False, download=True, transform=test_transforms
            )

        # ? Why do these not use the kwargs made above?
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

    else:
        if tv_dataset == "tin":

            # ? Why the random crop? The size of the images is then also no longer 64x64 as was specified...
            train_transforms = v2.Compose(
                [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(56)]
            )
            test_transforms = v2.Compose([v2.CenterCrop(56)])

            x_train, y_train, x_test, y_test = load_dataset(
                tv_dataset,
                device,
                torch.float32,
                validation=validation,
                mean=mean,
                std=std,
            )

            # ? What is the purpose of the onehot return? CCE loss? If so, we need change the logic here.
            train_dataset = ClassificationLoadedDataset(
                x_train, y_train, train_transforms, return_onehot=False
            )
            test_dataset = ClassificationLoadedDataset(
                x_test, y_test, test_transforms, return_onehot=False
            )

            train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
            test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        # elif ... # TODO: Add more datasets here and add to the assert above

    in_shape = tuple(train_dataset[0][0].shape)
    out_shape = len(train_dataset.classes)

    return train_loader, test_loader, in_shape, out_shape


def next_epoch(
    network,
    metrics,
    device,
    optimizer,
    test_loader,
    train_loader,
    loss_func,
    epoch,
    loud_test=True,
    loud_train=False,
    wandb=None,
    num_classes=10,
):
    """Trains and tests the network for one epoch. 
    Returns loss and accuracy.

    Parameters
    ----------
    loud_test: bool 
        If True, prints average loss and accuracy during the evaluation

    loud_train: bool 
        If True, prints average loss and accuracy during training. How often is determined by log_interval.   

    log_interval : int
        During training, controls how many batches there are between printing the models new accuracy
    
    
    """

    test_loss, test_acc = test(
            network, device, test_loader, epoch, loss_func, loud_test, num_classes=num_classes
        )

    metrics["test"]["loss"].append(test_loss)
    metrics["test"]["acc"].append(test_acc)

    train_loss, train_acc = test(
            network, device, train_loader, epoch, loss_func, loud_train, num_classes=num_classes
        )

    metrics["train"]["loss"].append(train_loss)
    metrics["train"]["acc"].append(train_acc)


    _, _ = train(
                    network, device, train_loader, optimizer ,epoch, loss_func, loud=loud_train, num_classes=num_classes
                )

    if wandb is not None:
        wandb.log(  {"test/loss": test_loss, "test/acc": test_acc, "train/loss": train_loss, "train/acc": train_acc}, step=epoch)
    return metrics


def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    loss_func,
    log_interval=100,
    loud=False,
    num_classes=10,
):
    """
    Trains model for one epoch

    Parameters
    ----------
    loud : bool 
        If True, prints average loss and accuracy

    log_interval : int
        Determines the number of batches between the logging of accuracy
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        onehots = torch.nn.functional.one_hot(target, num_classes).to(device).to(data.dtype)
        data, target = data.to(device), target.to(device)
        loss = model.train_step(data, target, onehots, loss_func)
        optimizer.step()

        if (batch_idx % log_interval == 0) and loud:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    loss /= len(train_loader)
    return loss, (100.0 * batch_idx / len(train_loader.dataset))


def test(
    model, device, test_loader, epoch, loss_func, loud=True, num_classes=10
):
    """
    Computes loss of model on test set
    Parameters
    ----------
    loud : bool 
        If True, prints average loss and accuracy at the end of epoch
    """

    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        onehots = torch.nn.functional.one_hot(target, num_classes).to(device)
        data, target = data.to(device), target.to(device)
        t1, t2 = model.BP_grads(data, target, onehots, loss_func)
        with torch.no_grad():
            loss, output = model.test_step(data, target, onehots, loss_func)
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    if loud:
        print(
            "\n Test Epoch {}: Loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                epoch,
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    test_loss /= len(test_loader)
    return test_loss, (100.0 * correct / len(test_loader.dataset)) 


def init_metric(validation=False):
    """
    Define Dictionary to hold loss and accuracy for trained model.
    
    Parameters
    ----------
    validation : bool
        If True, dictionary contains "val" instead of "test" keyword.

    Returns
    -------
    Dictionary of Dictionaries. 
    """
    if validation:
        return {"train": {"loss": [], "acc": []}, "val": {"loss": [], "acc": []}}
    return {"train": {"loss": [], "acc": []}, "test": {"loss": [], "acc": []}}


def plot_metrics(metrics):
    """Plot loss and accuracy over epochs for training and testing dataset."""
    plt.subplot(2, 1, 1)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(metrics["train"]["loss"])
    plt.plot(metrics["test"]["loss"])
    plt.subplot(2, 1, 2)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(metrics["train"]["acc"])
    plt.plot(metrics["test"]["acc"])
    plt.show()
