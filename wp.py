"""Weight perturbation core functionality"""

import torch
import torch.nn.functional as F


class WPLinearFunc(torch.autograd.Function):
    """Linear layer with noise injection at weight"""

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        sigma,
        pert_type,
        dist_sampler,
        num_perts,
        batch_size,
        first_layer,
        device,
    ):

        assert num_perts > 0, "Number of perturbations should never be zero"
        assert batch_size > 0, "Batch size must be non-zero"

        output = torch.zeros(
            size=(input.shape[0], weight.shape[0]), device=device
        )  # of shape [batch size, output size]

        seed = torch.randint(-int(1e10), int(1e10), size=(1,))
        torch.manual_seed(seed)

        # Perturb weights

        w_noise_shape = [num_perts] + list(
            weight.shape
        )  # Whether the noise is shared for each element of the batch

        w_noise = WPLinearFunc.sample_noise(dist_sampler, w_noise_shape, sigma)

        if "ffd" in pert_type.lower():
            output[:batch_size] += torch.mm(input[:batch_size], weight.t())
            output[batch_size:] += WPLinearFunc.add_noise(
                input[batch_size:], torch.add(weight, w_noise)
            )

        elif "cfd" in pert_type.lower():
            halfway = batch_size * num_perts
            if first_layer:
                perturbation = WPLinearFunc.add_noise(
                    input[:halfway], torch.add(weight, w_noise)
                )
                output[:halfway], output[halfway:] = (
                    perturbation,
                    -perturbation,
                )

            else:
                output[:halfway] = WPLinearFunc.add_noise(
                    input[:halfway], torch.add(weight, w_noise)
                )
                output[halfway:] = WPLinearFunc.add_noise(
                    input[halfway:], torch.subtract(weight, w_noise)
                )

        else:
            raise ValueError("Other weight perturbation types not yet implemented.")

        square_norm = torch.sum(w_noise**2, dim=(1, 2))

        # Perturb biases
        if bias is not None:
            b_noise_shape = [num_perts] + list(bias.shape)
            b_noise = WPLinearFunc.sample_noise(dist_sampler, b_noise_shape, sigma)

            if "ffd" in pert_type.lower():
                output[batch_size:] += torch.tile(
                    torch.add(bias, b_noise), (batch_size, 1)
                )

            elif "cfd" in pert_type.lower():

                output[:halfway] += torch.tile(
                    torch.add(bias, b_noise), (batch_size, 1)
                )
                output[halfway:] -= torch.tile(
                    torch.add(bias, b_noise), (batch_size, 1)
                )
            else:
                raise ValueError("Other bias perturbation types not yet implemented.")
            square_norm += torch.sum(b_noise**2, dim=1)

        else:
            b_noise = None

        return output, seed, square_norm

    @staticmethod
    def add_noise(inputs: torch.Tensor, noisy_weight: torch.Tensor):
        """Adds noise to the weight"""

        num_perts = noisy_weight.shape[0]
        elements_per_batch = int(inputs.shape[0] / noisy_weight.shape[0])
        reshaped_inputs = inputs.view(num_perts, elements_per_batch, -1)
        outputs = torch.einsum("boi,bni->bno", noisy_weight, reshaped_inputs).reshape(
            num_perts * elements_per_batch, -1
        )

        return outputs

    @staticmethod
    def sample_noise(sampler, shape, sigma):
        noise = sampler(shape) * sigma
        return noise

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None


class WPLinear(torch.nn.Linear):
    """Weight Perturbation layer with saved noise"""

    def __init__(
        self,
        *args,
        pert_type: str = "ffd",
        dist_sampler: torch.distributions.Distribution = None,
        sigma,
        num_perts: int = 1,
        device: str = "cuda:0",
        first_layer: bool = False,
        zero_masking: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.pert_type = pert_type
        self.dist_sampler = dist_sampler
        self.square_norm = None
        self.seed = None
        self.num_perts = num_perts
        self.first_gradient = True
        self.first_layer = first_layer
        self.zero_masking = zero_masking
        self.device = device
        self.sigma = sigma

    def __str__(self):
        return "WPLinear"

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # A clean and noisy input are both processed by a layer to produce
        if self.training:

            if "ffd" in self.pert_type.lower():
                self.batch_size = int(input.shape[0] / (self.num_perts + 1))
            elif "cfd" in self.pert_type.lower():
                self.batch_size = int(input.shape[0] / (self.num_perts * 2))
            else:
                raise ValueError("Other perturbation types not yet implemented.")
            (output, seed, square_norm) = WPLinearFunc().apply(
                input,
                self.weight,
                self.bias,
                self.sigma,
                self.pert_type,
                self.dist_sampler,
                self.num_perts,
                self.batch_size,
                self.first_layer,
                self.device,
            )
            if self.zero_masking:
                self.mask = input[: self.batch_size] != 0
            self.seed = seed
            self.square_norm = square_norm

        else:  # Do not perturb if weight are not being trained.
            output = F.linear(input, self.weight, self.bias)
        return output

    @torch.inference_mode()
    def update_grads(self, scaling_factor):
        # Rescale grad data - to be used at end of gradient pass

        # Scaling factor \in [num_pert]
        # weight \in [num_pert, out, in]

        torch.manual_seed(self.seed)

        # Set gradients for weights

        w_noise_shape = [self.num_perts] + list(self.weight.shape)

        scaled_weight_diff = torch.mul(
            scaling_factor[:, None, None],
            WPLinearFunc.sample_noise(self.dist_sampler, w_noise_shape, self.sigma),
        )
        # scaled weight diff has shape num pert, output shape, input shape
        if self.zero_masking:
            self.weight.grad = torch.mul(
                self.mask, torch.mean(scaled_weight_diff, axis=0)
            )

        else:
            self.weight.grad = torch.mean(scaled_weight_diff, axis=0)

        # Set gradients for biases
        if self.bias is not None:
            b_noise_shape = [self.num_perts] + list(self.bias.shape)

            scaled_bias_diff = torch.mul(
                scaling_factor[:, None],
                WPLinearFunc.sample_noise(self.dist_sampler, b_noise_shape, self.sigma),
            )

            self.bias.grad = torch.mean(scaled_bias_diff, axis=0)

    def get_noise_squarednorm(self):
        assert self.square_norm is not None, "square_norm has not been computed"
        return self.square_norm

    def get_number_perturbed_params(self):
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
