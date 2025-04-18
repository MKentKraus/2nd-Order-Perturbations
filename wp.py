"""Weight perturbation core functionality"""

import torch
import torch.nn.functional as F
import scipy


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
        device,
        orthogonal_perts,
    ):

        assert num_perts > 0, "Number of perturbations should never be zero"
        assert batch_size > 0, "Batch size must be non-zero"

        if (
            bias is not None
        ):  # Initialize matrix of shape [batch size, output size]. Prefilled with biases, if they exist.
            output = torch.tile(bias, (input.shape[0], 1))
        else:
            output = torch.zeros(size=(input.shape[0], weight.shape[0]), device=device)

        seed = torch.randint(-int(1e10), int(1e10), size=(1,))
        torch.manual_seed(seed)

        # Perturb weights

        w_noise_shape = [num_perts] + list(
            weight.shape
        )  # Whether the noise is shared for each element of the batch

        w_noise = WPLinearFunc.sample_noise(
            dist_sampler, w_noise_shape, sigma, orthogonal_perts
        )

        if "ffd" in pert_type.lower():
            output[:batch_size] += torch.mm(input[:batch_size], weight.t())  # f(w*x)
            output[batch_size:] += WPLinearFunc.add_noise(
                torch.add(weight, w_noise), input[batch_size:]
            )  # f((w+h) * x)

        elif "cfd" in pert_type.lower():

            halfway = batch_size * num_perts
            output[:halfway] += WPLinearFunc.add_noise(
                torch.add(weight, w_noise), input[:halfway]
            )  # f((w+h) * x)

            output[halfway:] += WPLinearFunc.add_noise(
                torch.subtract(weight, w_noise), input[halfway:]
            )  # f((w-h) * x)

        else:
            raise ValueError("Other weight perturbation types not yet implemented.")

        square_norm = torch.sum(w_noise**2, dim=(1, 2))

        # Perturb biases
        if bias is not None:
            b_noise_shape = [num_perts] + list(bias.shape)
            b_noise = WPLinearFunc.sample_noise(
                dist_sampler, b_noise_shape, sigma, orthogonal_perts
            )

            if "ffd" in pert_type.lower():
                output[batch_size:] += torch.tile(b_noise, (batch_size, 1))

            elif "cfd" in pert_type.lower():

                output[:halfway] += torch.tile(b_noise, (batch_size, 1))
                output[halfway:] -= torch.tile(b_noise, (batch_size, 1))
            else:
                raise ValueError("Other bias perturbation types not yet implemented.")
            square_norm += torch.sum(b_noise**2, dim=1)

        else:
            b_noise = None

        return output, seed, square_norm

    @staticmethod
    def add_noise(noisy_weight: torch.Tensor, inputs: torch.Tensor):
        """Adds noise to the weight"""

        num_perts = noisy_weight.shape[0]
        elements_per_batch = int(inputs.shape[0] / noisy_weight.shape[0])
        reshaped_inputs = inputs.reshape(num_perts, elements_per_batch, -1)
        outputs = torch.einsum("boi,bni->bno", noisy_weight, reshaped_inputs).reshape(
            num_perts * elements_per_batch, -1
        )
        return outputs

    @staticmethod
    def sample_noise(sampler, shape, sigma, orthogonal_perts: bool = False):
        if orthogonal_perts:
            u, _, vh = torch.linalg.svd(sampler(shape), full_matrices=False)
            noise = u @ vh * sigma

            """
            noise = torch.empty((shape))  # batch,
            for i in range(shape[0]):
                noise[i, :, :] = torch.tensor(
                    scipy.stats.ortho_group.rvs(shape[2]).T[:500, :]
                )
            """
        else:
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
        zero_masking: bool = True,
        orthogonal_perts: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.pert_type = pert_type
        self.dist_sampler = dist_sampler
        self.square_norm = None
        self.seed = None
        self.num_perts = num_perts
        self.first_gradient = True
        self.zero_masking = zero_masking
        self.device = device
        self.sigma = sigma
        self.orthogonal_perts = orthogonal_perts

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
                self.device,
                self.orthogonal_perts,
            )
            if self.zero_masking:
                self.mask = torch.where(input[: self.batch_size] != 0, 1.0, 0.0)
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

        if self.zero_masking:
            scaling_factor = torch.mul(
                scaling_factor[:, :, None], self.mask[:, None, :]
            )

            scaled_weight_diff = torch.mul(
                scaling_factor[:, :, None, :],
                WPLinearFunc.sample_noise(
                    self.dist_sampler, w_noise_shape, self.sigma, self.orthogonal_perts
                )[None, :, :, :],
            )  # [batch, pert, in] * [1, pert, out, in]

            self.weight.grad = torch.sum(torch.mean(scaled_weight_diff, axis=0), dim=0)

            scaling_factor = torch.sum(torch.mean(scaling_factor, dim=-1), dim=0)

        else:
            scaling_factor = torch.sum(scaling_factor, dim=0)
            scaled_weight_diff = torch.mul(
                scaling_factor[:, None, None],
                WPLinearFunc.sample_noise(
                    self.dist_sampler, w_noise_shape, self.sigma, self.orthogonal_perts
                ),
            )
            self.weight.grad = torch.mean(scaled_weight_diff, axis=0)
        # scaled weight diff has shape num pert, output shape, input shape

        # Set gradients for biases
        if self.bias is not None:
            b_noise_shape = [self.num_perts] + list(self.bias.shape)

            scaled_bias_diff = torch.mul(
                scaling_factor[:, None],
                WPLinearFunc.sample_noise(
                    self.dist_sampler, b_noise_shape, self.sigma, self.orthogonal_perts
                ),
            )

            self.bias.grad = torch.mean(scaled_bias_diff, axis=0)

    def get_noise_squarednorm(self):
        assert self.square_norm is not None, "square_norm has not been computed"
        return self.square_norm

    def get_number_perturbed_params(self):
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
