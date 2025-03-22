"""Dataset manipulation and network training functions"""

import os
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as v2
import wandb


def make_dist_sampler(
    distribution: str = "normal",
    device: torch.device | str = "cpu",
    **kwargs,
) -> torch.distributions.Distribution:
    """Create a distribution sampler for the noise"""

    if distribution.lower() == "normal":
        dist_sampler = lambda x: (torch.empty(x, device=device).normal_(mean=0, std=1))

    elif distribution.lower() == "bernoulli":
        distribution = torch.distributions.Bernoulli(
            torch.tensor([0.5]).to(torch.float32).to(device)
        )
        dist_sampler = lambda x: (distribution.sample(x).squeeze_(-1) - 0.5)
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
    y_train = y_train.type(torch.LongTensor).to(device)

    x_test = x_test.to(device).type(fltype)
    y_test = y_test.type(torch.LongTensor).to(device)

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

        in_shape = tuple(train_dataset[0][0].shape)
        out_shape = len(train_dataset.classes)

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
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
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

    return train_loader, test_loader, in_shape, out_shape


@staticmethod
def gram_schmidt(input):
    """Orthogonalize a set of vectors stored as the columns of matrix A."""
    shape = input.shape
    num_perts = shape[0]
    input = input.reshape(
        num_perts, -1
    ).t()  # input is flattened over the weights, and then turned so the perturbations form the columns

    for pert in range(num_perts):
        for previus_pert in range(pert):
            # print(torch.dot(input[:, previus_pert], input[:, pert]))
            input[:, pert] -= (
                torch.dot(input[:, previus_pert], input[:, pert])
                / torch.dot(input[:, previus_pert], input[:, previus_pert])
                * input[:, previus_pert]
            )
            # print(torch.dot(input[:, previus_pert], input[:, pert]))

    input = input.t().reshape(shape)

    return input


def next_epoch(
    network,
    metrics,
    device,
    optimizer,
    test_loader,
    train_loader,
    loss_func,
    epoch,
    bias=True,
    loud_test=True,
    loud_train=False,
    comp_angles=False,
    validation=False,
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

    test_metrics = test(
        network,
        device,
        test_loader,
        epoch,
        loss_func,
        comp_angles,
        loud_test,
        num_classes=num_classes,
    )

    metrics["test"]["loss"].append(test_metrics[0])
    metrics["test"]["acc"].append(test_metrics[1])
    if comp_angles:
        metrics["angle"].append(test_metrics[2])

    train_metrics = test(
        network,
        device,
        train_loader,
        epoch,
        loss_func,
        comp_angles=False,
        loud=False,
        num_classes=num_classes,
    )
    metrics["train"]["loss"].append(train_metrics[0])
    metrics["train"]["acc"].append(train_metrics[1])

    train_results = train(
        network,
        device,
        train_loader,
        optimizer,
        epoch,
        loss_func,
        comp_angles=comp_angles,
        loud=loud_train,
        num_classes=num_classes,
    )

    w = network.state_dict().get("network.1.weight")
    w = torch.linalg.vector_norm(w) / torch.numel(w)

    if bias:
        b = network.state_dict().get("network.1.bias")

        b = torch.linalg.vector_norm(b) / torch.numel(b)

    wandb.log(
        {
            "weights/learned weights scale": w,
            "weights/learned biases scale": b if bias else 0,
        },
        step=epoch,
    )
    if comp_angles:
        wandb.log(
            {
                "angle/angle": test_metrics[2],
                "angle/OSE": train_results[-1],
            },
            step=epoch,
        )

    if validation:
        wandb.log(
            {
                "validation/loss": test_metrics[0],
                "validation/acc": test_metrics[1],
                "train/loss": train_metrics[0],
                "train/acc": train_metrics[1],
            },
            step=epoch,
        )
    else:
        wandb.log(
            {
                "test/loss": test_metrics[0],
                "test/acc": test_metrics[1],
                "train/loss": train_metrics[0],
                "train/acc": train_metrics[1],
            },
            step=epoch,
        )
    return metrics


def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    loss_func,
    comp_angles=False,
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

    bp_loss = np.zeros(2)
    wp_loss = np.zeros(2)
    i = 0

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        onehots = (
            torch.nn.functional.one_hot(target, num_classes).to(device).to(data.dtype)
        )
        data, target = data.to(device), target.to(device)

        if (
            batch_idx == len(train_loader) - 2 or batch_idx == len(train_loader) - 3
        ) and comp_angles:
            _, bp_loss[i], loss = model.compare_BP(
                data, target, onehots, loss_func, load_weights=(i == 0)
            )
            wp_loss[i] = loss
            i += 1
        else:
            loss = model.train_step(data, target, onehots, loss_func)

        optimizer.step()

        if (batch_idx % log_interval == 0) and loud:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss,
                )
            )

    loss /= len(train_loader.dataset)

    train_results = [loss, (100.0 * batch_idx / len(train_loader.dataset))]

    if comp_angles:
        ose = (wp_loss[0] - wp_loss[1]) / (
            bp_loss[0] - bp_loss[1] + 1e-16
        )  # loss improvement in WP over the loss improvement in BP
        train_results.append(ose)

    return train_results


def test(
    model,
    device,
    test_loader,
    epoch,
    loss_func,
    comp_angles,
    loud=True,
    num_classes=10,
):
    """
    Computes loss of model on test set, and computes the angle between the gradients derived by perturbation and backpropagation
    Parameters
    ----------
    loud : bool
        If True, prints average loss and accuracy at the end of epoch
    """

    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        angle = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            onehots = (
                torch.nn.functional.one_hot(target, num_classes)
                .to(device)
                .to(data.dtype)
            )

            data, target = data.to(device), target.to(device)
            loss, output = model.test_step(data, target, onehots, loss_func)
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
    if comp_angles:
        assert (
            type(test_loader.sampler) is torch.utils.data.sampler.SequentialSampler
        ), "Shuffle must be False for the Dataloader, to ensure that angles are compared between the same elements each epoch"

        angle, _, _ = model.compare_BP(
            data, target, onehots, loss_func
        )  # compare angles on the final batch of each epoch

    if loud:
        print(
            "\n Test Epoch {}: Angle: {}, Loss: {:.15f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                epoch,
                angle if comp_angles else "not measured",
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    return test_loss, (100.0 * correct / len(test_loader.dataset)), angle


def init_metric(comp_angles=False):
    """
    Define Dictionary to hold loss and accuracy for trained model.

    Parameters
    ----------
    comp_angles : bool
        If True, dictionary contains "angle" and "one step effectiveness" fields.

    Returns
    -------
    Dictionary of Dictionaries.
    """
    if comp_angles:
        return {
            "train": {"loss": [], "acc": []},
            "test": {"loss": [], "acc": []},
            "angle": [],
            "one step effectiveness": [],
        }
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
