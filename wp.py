"""Weight perturbation core functionality"""

import torch
import torch.nn.functional as F
from utils import add_noise

class WPLinearFunc(torch.autograd.Function):
    """Linear layer with noise injection at weights"""

    @staticmethod
    def forward(ctx, input, weights, biases, clean_pass, dist_sampler, sample_wise, direction='Ran'):
        # Matrix multiplying both the clean and noisy forward signals
        output = torch.mm(input, weights.t())
        half_batch_width = len(input) // 2

        noise_dim = half_batch_width if sample_wise else 1 #Whether the noise is unique or shared for each element of the batch, determined by sample_wise

        # Determining the shape of the noise
        # Noise shape based upon weight shape and batch size
        noise_shape = list(weights.shape)
        if sample_wise:
            noise_shape.insert(0,noise_dim)


        ###Perturb original data
        if clean_pass:
            w_noise_1 = torch.zeros(noise_shape, device=input.device)
        else:
            w_noise_1 = dist_sampler(noise_shape)
            output[:half_batch_width] += add_noise(input[:half_batch_width],w_noise_1, sample_wise)

        ###Perturb copy of data
        w_noise_2 = dist_sampler(noise_shape)
        weight_diff = w_noise_1 - w_noise_2        
        output[half_batch_width:] += add_noise(input[half_batch_width:], w_noise_2, sample_wise)


        #perturbing the biases
        b_noise_1, b_noise_2 = None, None
        bias_diff = None
        if biases is not None: 


            noise_shape = [noise_dim] + list(biases.shape)


            #Samples noise to add to biases. Second half is always perturbed
            if clean_pass:
                b_noise_1 = torch.zeros(noise_shape, device=input.device)
                b_noise_2 = dist_sampler(noise_shape)
            else:
                b_noise_1 = dist_sampler(noise_shape) 
                b_noise_2 = dist_sampler(noise_shape)

            bias_diff = b_noise_1 - b_noise_2

            output[:half_batch_width] += biases + b_noise_1
            output[half_batch_width:] += biases + b_noise_2
        # compute the output
        return output, weight_diff, bias_diff

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None


class WPLinear(torch.nn.Linear):
    """Weight Perturbation layer with saved noise"""

    def __init__(
        self,
        *args,
        clean_pass: bool = False,
        dist_sampler: torch.distributions.Distribution = None,
        sample_wise: bool = False,
        direction: str = "For",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.clean_pass = clean_pass
        self.dist_sampler = dist_sampler
        self.sample_wise = sample_wise
        self.direction = direction
        self.square_norm = None
        self.weight_diff = None
        self.bias_diff = None

    def __str__(self):
        return "WPLinear"

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # A clean and noisy input are both processed by a layer to produce
        if self.training:
            (output, weight_diff, bias_diff) = WPLinearFunc().apply(
                input,
                self.weight,
                self.bias,
                self.clean_pass,
                self.dist_sampler,
                self.sample_wise,
            )

            half_batch_width = len(input) // 2

            self.weight_diff = weight_diff
            self.bias_diff = bias_diff
            noise_dim = half_batch_width if self.sample_wise else 1

            self.square_norm = torch.sum(
                (self.weight_diff.reshape(noise_dim, -1)) ** 2, axis=1
            )
            if self.bias is not None:
                self.square_norm += torch.sum(
                    self.bias_diff.reshape(noise_dim, -1) ** 2, axis=1
                )

    
        else: #Do not perturb if weights are not being trained.
            output = F.linear(input, self.weight, self.bias)
        return output



    @torch.inference_mode()
    def update_grads(self, scaling_factor):
        # Rescale grad data - to be used at end of gradient pass
        scaled_weight_diff = scaling_factor[:, None, None] * self.weight_diff
        self.weight.grad = torch.sum(scaled_weight_diff, axis=0)

        if self.bias is not None:
            scaled_bias_diff = scaling_factor[:, None] * self.bias_diff
            self.bias.grad = torch.sum(scaled_bias_diff, axis=0)

    def get_noise_squarednorm(self):
        assert self.square_norm is not None, "square_norm has not been computed"
        return self.square_norm

    def get_number_perturbed_params(self):
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)