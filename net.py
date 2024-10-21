"""Neural Network definition"""

import torch

class PerturbNet(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
    ):
        super(PerturbNet, self).__init__()
        self.network = network
    
    @torch.inference_mode
    def forward(self, x):
        if self.training:
            x = torch.concatenate([x, x.clone()])
        return self.network(x)


    def apply_grad_scaling_to_noise_layers(self, network, scaling): 
        for child in network.children(): #iterates over layers of the network.
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
        normalization = num_params / normalizer
        return normalization
    
    def compare_BPangles(self, data, target, onehots, loss_func):
        self.BP_update(data, target, onehots, loss_func)
        BP_grads = self.get_grads()

        self.train_step(data, target, onehots, loss_func)
        WP_grads = self.get_grads()

        return torch.arccos(torch.clip(torch.dot(  WP_grads / torch.linalg.vector_norm(WP_grads), BP_grads / torch.linalg.vector_norm(BP_grads)), -1.0, 1.0))

    @torch.inference_mode()
    def train_step(self, data, target, onehots, loss_func):
        self.train()

        output = self(data)

        w1_loss = loss_func(output[: len(data)], target, onehots)  # sum up batch loss under first set of weights (can but do not have to be the clean weights)
        w2_loss = loss_func(output[len(data) :], target, onehots)  # sum up batch loss

        # Multiply grad by loss differential and normalize with unit norms
        loss_differential = w1_loss - w2_loss
        normalization = self.get_normalization(self.network)
        grad_scaling = loss_differential * normalization #normalizes the loss

        self.apply_grad_scaling_to_noise_layers(self.network, grad_scaling) #updates the gradient of the params

        return w1_loss.mean().item()

    @torch.inference_mode()
    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        output = self(data)
        loss = torch.mean(
            loss_func(output, target, onehots)
        )  # sum up batch loss
        loss = loss.item()
        return loss, output

    def get_grads(self):
        grad_params = []
        for param in self.network.parameters():
            grad_params.append(param.grad.view(-1))
        grad_params = torch.cat(grad_params)
        return grad_params 

    def BP_update(self, data, target, onehots, loss_func):
        self.eval() #https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial.html
        output = self.network(data).to(torch.float32)
        target = target.to(torch.float32)
        onehots = onehots.to(torch.float32)
        loss = loss_func(output, target, onehots)
        loss = loss.mean().to(torch.float32)
        loss.backward()

class BPNet(torch.nn.Module):
    def __init__(
        self,
        network
    ):
        super(BPNet, self).__init__()
        self.network = network

    def make_network(self):
        raise NotImplementedError("This is a base class")

    def forward(self, x):
        return self.network(x)

    def train_step(self, data, target, onehots, loss_func):
        self.train()
        output = self(data)
        loss = loss_func(output, target, onehots)
        total_loss = loss.mean()
        total_loss.backward()

        return total_loss.item()

    @torch.inference_mode()
    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        output = self(data)
        loss = torch.mean(
            loss_func(output, target, onehots)
        ).item()  # sum up batch loss
        return loss, output
    
