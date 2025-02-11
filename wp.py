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
        weight_sigma,
        weight_mu,
        bias,
        bias_sigma,
        bias_mu,
        pert_type,
        dist_sampler,
        sample_wise,
        num_perts,
        mu_scaling,
        batch_size,
    ):

        if sample_wise:
            assert num_perts == 1, "Don't use multiple perturbations for a fixed noise"
        assert num_perts > 0, "Number of perturbations should never be zero"

        output = torch.mm(input, weight.t())
        if bias is not None:
            output += bias

        # Whether the noise is unique or shared for each element of the batch, determined by sample_wise
        noise_shape = (
            [batch_size] + list(weight.shape)
            if sample_wise
            else [num_perts] + list(weight.shape)
        )

        seed = torch.randint(-int(1e10), int(1e10), size=(1,))
        torch.manual_seed(seed)
        w_noise = (
            dist_sampler(noise_shape) * weight_sigma.repeat(noise_shape[0], 1, 1)
        ) + weight_mu.repeat(
            noise_shape[0], 1, 1
        ) * mu_scaling  # sample from a normal gaussian, then reshape into desired variance and mean.

        # expand noise to be separate for each batch part

        w_noise = torch.unsqueeze(w_noise, 1).repeat(1, batch_size, 1, 1)

        w_noise = torch.where(
            torch.unsqueeze(input[:batch_size], 1).expand(-1, weight.shape[0], -1) == 0,
            0,
            w_noise,
        )  # Do not perturb zero inputs.

        b_noise = None

        if "ffd" in pert_type.lower():

            assert batch_size > 0
            output[batch_size:] += WPLinearFunc.add_noise(
                input[batch_size:], w_noise, sample_wise
            )
            if bias is not None:
                b_noise_shape = [noise_shape[0]] + list(bias.shape)
                b_noise = (
                    dist_sampler(b_noise_shape) * bias_sigma.repeat(noise_shape[0], 1)
                ) + bias_mu.repeat(noise_shape[0], 1) * mu_scaling

                if sample_wise:
                    output[batch_size:] += b_noise
                else:
                    output[batch_size:] += torch.tile(b_noise, (batch_size, 1))

        elif "cfd" in pert_type.lower():

            half = batch_size * num_perts

            output[:half] += WPLinearFunc.add_noise(input[:half], w_noise, sample_wise)
            output[half:] += WPLinearFunc.add_noise(input[half:], -w_noise, sample_wise)

            if bias is not None:
                b_noise_shape = [noise_shape[0]] + list(bias.shape)

                b_noise = (
                    dist_sampler(b_noise_shape) * bias_sigma.repeat(noise_shape[0], 1)
                ) + bias_mu.repeat(noise_shape[0], 1) * mu_scaling

                if sample_wise:
                    output[:half] += b_noise
                    output[half:] -= b_noise
                else:
                    output[:half] += torch.tile(b_noise, (batch_size, 1))
                    output[half:] -= torch.tile(b_noise, (batch_size, 1))

        else:
            raise ValueError("Other perturbation types not yet implemented.")

        square_norm = torch.sum(w_noise**2, dim=(1, 2, 3))
        if bias is not None:
            square_norm += torch.sum(b_noise**2, dim=1)

        return output, seed, square_norm, w_noise != 0

    @staticmethod
    def add_noise(inputs: torch.Tensor, noisy_weight: torch.Tensor, sample_wise: bool):
        """Adds noise to the weight. If sample_wise is true, noise is assumed to be unique for each element"""

        if sample_wise:
            assert (
                noisy_weight.shape[0] == inputs.shape[0]
            ), "Shape of perturbations should be same as inputs!"
            outputs = torch.einsum("noi,ni->no", noisy_weight, inputs)
        else:
            num_perts = noisy_weight.shape[0]
            elements_per_batch = int(inputs.shape[0] / noisy_weight.shape[0])
            reshaped_inputs = inputs.view(num_perts, elements_per_batch, -1)

            outputs = torch.einsum("pboi,pbi->pbo", noisy_weight, reshaped_inputs)
            outputs = outputs.reshape(num_perts * elements_per_batch, -1)

        return outputs

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
        mu_scaling_factor,
        meta_lr,
        sample_wise: bool = False,
        num_perts: int = 1,
        device,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.pert_type = pert_type
        self.dist_sampler = dist_sampler
        self.sample_wise = sample_wise
        self.square_norm = None
        self.seed = None
        self.num_perts = num_perts
        self.mu_scaling_factor = mu_scaling_factor
        self.meta_lr = meta_lr  # the meta learning rate. How much of the past gradient estimate is carried over as momentum
        self.first_gradient = True

        self.weight_sigma = torch.full(
            size=(self.weight.shape),
            fill_value=sigma,
            dtype=torch.float32,
            device=device,
        )

        self.register_buffer(
            "weight_mu",
            torch.zeros(
                size=(self.weight.shape),
                dtype=torch.float32,
                device=device,
            ),
        )

        if self.bias is not None:
            self.bias_sigma = torch.full(
                size=(self.bias.shape),
                fill_value=sigma,
                dtype=torch.float32,
                device=device,
            )

            self.register_buffer(
                "bias_mu",
                torch.zeros(
                    size=(self.bias.shape),
                    dtype=torch.float32,
                    device=device,
                ),
            )
        else:
            self.bias_sigma = None
            self.bias_mu = None

        if "grad" in self.pert_type.lower():

            self.grad_w_est = torch.zeros(
                size=(self.weight.shape),
                dtype=torch.float32,
                device=device,
            )

            if self.bias is not None:

                self.grad_b_est = torch.zeros(
                    size=(self.bias.shape),
                    dtype=torch.float32,
                    device=device,
                )

    def __str__(self):
        return "WPLinear"

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # A clean and noisy input are both processed by a layer to produce
        if self.training:

            if "ffd" in self.pert_type.lower():
                batch_size = int(input.shape[0] / (self.num_perts + 1))
            elif "cfd" in self.pert_type.lower():
                batch_size = int(input.shape[0] / (self.num_perts * 2))
            (output, seed, square_norm, mask) = WPLinearFunc().apply(
                input,
                self.weight,
                self.weight_sigma,
                self.weight_mu,
                self.bias,
                self.bias_sigma,
                self.bias_mu,
                self.pert_type,
                self.dist_sampler,
                self.sample_wise,
                self.num_perts,
                self.mu_scaling_factor,
                batch_size,
            )
            self.mask = mask
            self.seed = seed
            self.square_norm = square_norm

        else:  # Do not perturb if weight are not being trained.
            output = F.linear(input, self.weight, self.bias)
        return output

    @torch.inference_mode()
    def update_grads(self, scaling_factor):
        # Rescale grad data - to be used at end of gradient pass

        # Scaling factor \in [batch, num_pert]
        # weight \in [batch OR pert, out, in]

        torch.manual_seed(self.seed)

        batch_size = self.mask.shape[1]
        noise_shape = (
            [batch_size] + list(self.weight.shape)
            if self.sample_wise
            else [self.num_perts] + list(self.weight.shape)
        )
        b_noise_shape = [noise_shape[0]] + list(self.bias.shape)

        w_noise = (
            self.dist_sampler(noise_shape)
            * self.weight_sigma.repeat(noise_shape[0], 1, 1)
        ) + self.weight_mu.repeat(noise_shape[0], 1, 1) * self.mu_scaling_factor

        w_noise = torch.unsqueeze(w_noise, 1).repeat(1, batch_size, 1, 1)

        w_noise = torch.where(self.mask, w_noise, 0)
        scaled_weight_diff = (
            scaling_factor[:, :, None, None] * w_noise[:, :, :, :]
        )  # scaling factor times the reconstructed noise

        self.weight.grad = torch.sum(torch.mean(scaled_weight_diff, axis=1), dim=0)

        if self.bias is not None:
            if self.sample_wise:
                scaled_bias_diff = (
                    scaling_factor[:, :, None]
                    * (
                        (
                            self.dist_sampler(b_noise_shape)
                            * self.bias_sigma.repeat(noise_shape[0], 1)
                        )
                        + self.bias_mu.repeat(noise_shape[0], 1)
                        * self.mu_scaling_factor
                    )[:, None, :]
                )
            else:
                scaled_bias_diff = (
                    scaling_factor[:, :, None]
                    * (
                        (
                            self.dist_sampler(b_noise_shape)
                            * self.bias_sigma.repeat(noise_shape[0], 1)
                        )
                        + self.bias_mu.repeat(noise_shape[0], 1)
                        * self.mu_scaling_factor
                    )[:, None, :]
                )

            self.bias.grad = torch.sum(torch.mean(scaled_bias_diff, axis=1), dim=0)

        elif "meta" in self.pert_type.lower():

            if self.first_gradient:
                self.weight_mu = self.weight.grad

                if self.bias is not None:
                    self.bias_mu = self.bias.grad
                self.first_gradient = False

            else:
                self.weight_mu = (
                    self.weight_mu * self.meta_lr
                    + (1 - self.meta_lr) * self.weight.grad
                )

                if self.bias is not None:
                    self.bias_mu = (
                        self.bias_mu * self.meta_lr
                        + (1 - self.meta_lr) * self.bias.grad
                    )

        if "grad" in self.pert_type.lower():

            if self.first_gradient:
                self.grad_w_est = self.weight.grad
                if self.bias is not None:
                    self.grad_b_est = self.bias.grad
                self.first_gradient = False

            else:
                self.grad_w_est = (
                    self.mu_scaling_factor
                ) * self.grad_w_est + self.weight.grad

                self.weight.grad = self.grad_w_est

                if self.bias is not None:
                    self.grad_b_est = (
                        self.mu_scaling_factor
                    ) * self.grad_b_est + self.bias.grad

                    self.bias.grad = self.grad_b_est

    def get_noise_squarednorm(self):
        assert self.square_norm is not None, "square_norm has not been computed"
        return self.square_norm

    def get_number_perturbed_params(self):
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
