"""Neural Network definition"""

import torch

class PerturbForwNet(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
    ):
        super(PerturbForwNet, self).__init__()
        self.network = network

    @torch.inference_mode()
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

    @torch.inference_mode()
    def train_step(self, data, target, onehots, loss_func):
        
        #Get current models loss and accuracy
        self.eval() 
        output = self(data) 
        clean_loss = loss_func(output, target, onehots) 


        #get two points to determine gradient
        self.train()

        output = self(data)

        w1_loss = loss_func(output[: len(data)], target, onehots)  # sum up batch loss under first set of weights (can but do not have to be the clean weights)
        w2_loss = loss_func(output[len(data) :], target, onehots)  # sum up batch loss

        # Multiply grad by loss differential and normalize with unit norms
        loss_differential = w1_loss - w2_loss 
        normalization = self.get_normalization(self.network)
        grad_scaling = loss_differential * normalization #normalizes the loss

        self.apply_grad_scaling_to_noise_layers(self.network, grad_scaling) #updates the gradient of the params

        return clean_loss.mean().item()





    @torch.inference_mode()
    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        output = self(data)[: len(data)]
        loss = torch.mean(
            loss_func(output, target, onehots)
        ).item()  # sum up batch loss
        return loss, output


class PerturbCentNet(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
    ):
        super(PerturbCentNet, self).__init__()
        self.network = network

    @torch.inference_mode()
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

    @torch.inference_mode()
    def train_step(self, data, target, onehots, loss_func):
        # Duplicate data for network clean/noisy pass
        self.train()
        output = self(data)

        clean_loss = loss_func(
            output[: len(data)], target, onehots
        )  # sum up batch loss
        noisy_loss = loss_func(
            output[len(data) :], target, onehots
        )  # sum up batch loss

        # Multiply grad by loss differential and normalize with unit norms
        loss_differential = clean_loss - noisy_loss #change to CDF
        normalization = self.get_normalization(self.network)
        grad_scaling = loss_differential * normalization
        self.apply_grad_scaling_to_noise_layers(self.network, grad_scaling)

        return clean_loss.mean().item()

    @torch.inference_mode()
    def test_step(self, data, target, onehots, loss_func):
        self.eval()
        output = self(data)[: len(data)]
        loss = torch.mean(
            loss_func(output, target, onehots)
        ).item()  # sum up batch loss
        return loss, output

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
    