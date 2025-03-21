"""Neural Network definition"""

import torch
import numpy as np


class PerturbNet(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
        num_perts: int = 1,
        pert_type: str = "ffd",
        BP_network: torch.nn.Module = None,
    ):
        super(PerturbNet, self).__init__()
        self.network = network
        self.pert_type = pert_type
        self.num_perts = num_perts

        if BP_network is not None:
            self.BP_network = BP_network
            pytorch_total_params = sum(
                p.numel() for p in BP_network.parameters() if p.requires_grad
            )  # should get the total number of trainable parameters
            self.comp_params = np.random.permutation(pytorch_total_params)[
                :20
            ]  # randomly selects a thousand random parameters to compare the angle on.

    @torch.inference_mode()
    def forward(self, x):

        if self.training:
            dim = torch.ones(
                x.dim(), dtype=torch.int8
            ).tolist()  # repeat requires a list/tuple of ints with all dimensions of the tensor
            if "ffd" in self.pert_type.lower():
                dim[0] = self.num_perts + 1
            elif "cfd" in self.pert_type.lower():
                dim[0] = self.num_perts * 2
            x = x.repeat(dim)
        return self.network(x)

    def apply_grad_scaling_to_noise_layers(self, network, scaling):
        for child in network.children():  # iterates over layers of the network.
            # are you a container?
            if len(list(child.children())) > 0:
                self.apply_grad_scaling_to_noise_layers(child, scaling)
            else:
                if hasattr(child, "update_grads"):
                    child.update_grads(scaling)

    def get_network_noise_normalizers(self, network):
        num_parameters = 0
        normalizer = 0
        for child in network.children():
            # are you a container?
            if len(list(child.children())) > 0:
                ps, ns = self.get_network_noise_normalizers(child)
                num_parameters += ps
                normalizer += ns
            else:
                if hasattr(child, "get_noise_squarednorm"):
                    num_parameters += child.get_number_perturbed_params()
                    normalizer += child.get_noise_squarednorm()

        return num_parameters, normalizer

    def get_normalization(self, network):
        num_params, normalizer = self.get_network_noise_normalizers(network)
        normalization = torch.div(torch.tensor(num_params), normalizer)

        return normalization

    def compare_BP(self, data, target, onehots, loss_func, load_weights=True):
        """Compare a weight perturbation update to a backpropagation update"""
        assert (
            self.BP_network is not None
        ), "To compare against BP, an equivalent model using default torch layers must be provided"

        if load_weights:
            WP_dict = {
                k: v
                for k, v in self.network.state_dict().items()
                if k in self.BP_network.state_dict()
            }  # Filters the WP dict so that only those weights that are also in the BP model remain

            self.BP_network.load_state_dict(WP_dict)

        bp_loss = self.BP_update(data, target, onehots, loss_func)
        BP_grads = self.get_grads(self.BP_network)[
            self.comp_params
        ]  # gets all the gradients, then selects the ones for comparison

        wp_loss = self.train_step(data, target, onehots, loss_func)

        WP_grads = self.get_grads(self.network)[self.comp_params]
        # compute the angle between the two vectors by using the dot produ ct
        WP_grads = torch.div(WP_grads, torch.linalg.vector_norm(WP_grads))
        BP_grads = torch.div(BP_grads, torch.linalg.vector_norm(BP_grads))
        return torch.acos(torch.dot(WP_grads, BP_grads)).item(), bp_loss, wp_loss

    @torch.inference_mode()
    def train_step(self, data, target, onehots, loss_func):
        # This function performs the forward and backward pass
        self.train()

        loss, loss_differential = self.forward_pass(data, target, onehots, loss_func)
        self.backward_pass(loss_differential)

        return loss

    @torch.inference_mode()
    def forward_pass(self, data, target, onehots, loss_func):
        self.train()

        output = self(data)
        batch_size = data.shape[0]

        if "ffd" in self.pert_type.lower():
            loss = loss_func(output[:batch_size], target, onehots)  # clean loss
            loss_2 = loss_func(
                output[batch_size:],
                target.repeat(self.num_perts),
                onehots.repeat(self.num_perts, 1),
            )

            loss_differential = loss_2 - loss.repeat(self.num_perts)

        elif "cfd" in self.pert_type.lower():
            half = self.num_perts * batch_size
            loss_1 = loss_func(
                output[:half],
                target.repeat(self.num_perts),
                onehots.repeat(self.num_perts, 1),
            )
            loss_2 = loss_func(
                output[half:],
                target.repeat(self.num_perts),
                onehots.repeat(self.num_perts, 1),
            )
            loss = torch.mean(torch.stack((loss_1, loss_2), dim=0), dim=0)
            loss_differential = loss_1 - loss_2

        loss_differential = loss_differential.view(
            self.num_perts, -1
        )  # dim num_perts, batch size
        return loss.sum().item() / self.num_perts, loss_differential

    @torch.inference_mode()
    def backward_pass(self, loss_differential):
        normalization = self.get_normalization(self.network).unsqueeze(
            1
        )  # dim [num_perts, 1]

        grad_scaling = torch.mul(
            loss_differential, normalization
        ).t()  # Loss, across perturbations [num_perts, batch_size]
        self.apply_grad_scaling_to_noise_layers(self.network, grad_scaling)

    @torch.inference_mode()
    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        output = self(data)
        loss = torch.sum(loss_func(output, target, onehots))  # sum up batch loss
        loss = loss.item()
        return loss, output

    def get_grads(self, network):
        grad_params = []
        for child in network.children():  # iterates over layers of the network.
            # are you a container?
            if len(list(child.children())) > 0:
                grad_params.append(self.get_grads(child).view(-1))
            else:
                if hasattr(child, "weight"):
                    grad_params.append(child.weight.grad.view(-1))
                if hasattr(child, "bias"):
                    if child.bias is not None:
                        grad_params.append(child.bias.grad.view(-1))
        grad_params = torch.cat(grad_params)
        return grad_params

    def BP_update(self, data, target, onehots, loss_func):
        """Calculates the gradient using backpropagation for the current inputs and targets"""
        assert self.BP_network is not None

        output = self.BP_network(data)
        loss = loss_func(output, target, onehots).sum()
        loss.backward()
        return loss.item()


class BPNet(torch.nn.Module):
    def __init__(self, network):
        super(BPNet, self).__init__()
        self.network = network

    def make_network(self):
        raise NotImplementedError("This is a base class")

    def forward(self, x):
        return self.network(x)

    def train_step(self, data, target, onehots, loss_func):
        self.train()
        output = self(data)

        loss = loss_func(output, target, onehots).sum()

        loss.backward()
        return loss.item()

    def forward_pass(self, data, target, onehots, loss_func):
        self.train()
        output = self(data)
        loss = loss_func(output, target, onehots).sum()
        return loss, loss

    def backward_pass(self, loss):
        loss.backward()

    @torch.inference_mode()
    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        output = self(data)
        loss = torch.sum(loss_func(output, target, onehots)).item()  # sum up batch loss
        return loss, output
