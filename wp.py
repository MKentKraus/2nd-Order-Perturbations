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
        mu_scaling_factor,
        batch_size,
    ):

        if sample_wise:
            assert num_perts == 1, "Don't use multiple perturbations for a fixed noise"
        assert num_perts > 0, "Number of perturbations should never be zero"
        assert batch_size > 0, "Batch size must be non-zero"

        output = torch.mm(input, weight.t())
        if bias is not None:
            output += bias

        seed = torch.randint(-int(1e10), int(1e10), size=(1,))
        torch.manual_seed(seed)

        # Perturb weights

        w_noise_shape = (
            [batch_size] + list(weight.shape)
            if sample_wise
            else [num_perts] + list(weight.shape)
        )  # Whether the noise is unique or shared for each element of the batch, determined by sample_wise

        w_noise = WPLinearFunc.sample_noise(
            dist_sampler, w_noise_shape, weight_sigma, weight_mu, mu_scaling_factor
        )

        if "ffd" in pert_type.lower():
            output[batch_size:] += WPLinearFunc.add_noise(
                input[batch_size:], w_noise, sample_wise
            )

        elif "cfd" in pert_type.lower():
            halfway = batch_size * num_perts
            output[:halfway] += WPLinearFunc.add_noise(
                input[:halfway], w_noise, sample_wise
            )
            output[halfway:] += WPLinearFunc.add_noise(
                input[halfway:], -w_noise, sample_wise
            )

        else:
            raise ValueError("Other weight perturbation types not yet implemented.")

        square_norm = torch.sum(w_noise**2, dim=(1, 2))

        # Perturb biases
        if bias is not None:
            b_noise_shape = (
                [batch_size] + list(bias.shape)
                if sample_wise
                else [num_perts] + list(bias.shape)
            )
            b_noise = WPLinearFunc.sample_noise(
                dist_sampler, b_noise_shape, bias_sigma, bias_mu, mu_scaling_factor
            )

            if "ffd" in pert_type.lower():
                if sample_wise:
                    output[batch_size:] += b_noise
                else:
                    output[batch_size:] += torch.tile(b_noise, (batch_size, 1))

            elif "cfd" in pert_type.lower():
                if sample_wise:
                    output[:halfway] += b_noise
                    output[halfway:] -= b_noise
                else:
                    output[:halfway] += torch.tile(b_noise, (batch_size, 1))
                    output[halfway:] -= torch.tile(b_noise, (batch_size, 1))
            else:
                raise ValueError("Other bias perturbation types not yet implemented.")
            square_norm += torch.sum(b_noise**2, dim=1)

        else:
            b_noise = None

        return output, seed, square_norm

    @staticmethod
    def add_noise(inputs: torch.Tensor, noisy_weight: torch.Tensor, sample_wise: bool):
        """Adds noise to the weight. If sample_wise is true, noise is assumed to be unique for each element"""

        if sample_wise:
            assert (
                noisy_weight.shape[0] == inputs.shape[0]
            ), "Shape of perturbations should be same as inputs!"
            outputs = torch.einsum("noi,ni->no", noisy_weight[0], inputs)
        else:
            num_perts = noisy_weight.shape[0]
            elements_per_batch = int(inputs.shape[0] / noisy_weight.shape[0])
            reshaped_inputs = inputs.view(num_perts, elements_per_batch, -1)
            outputs = torch.einsum(
                "boi,bni->bno", noisy_weight, reshaped_inputs
            ).reshape(num_perts * elements_per_batch, -1)

        return outputs

    @staticmethod
    def sample_noise(sampler, shape, sigma, mu, mu_scaling_factor):
        dims = torch.ones(len(shape), dtype=torch.int8).tolist()
        dims[0] = shape[0]
        noise = (
            sampler(shape) * sigma.repeat(dims) + mu.repeat(dims) * mu_scaling_factor
        )
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
        mu_scaling_factor,
        meta_lr,
        sample_wise: bool = False,
        num_perts: int = 1,
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
            device="cuda:0",
        )

        self.register_buffer(
            "weight_mu",
            torch.zeros(
                size=(self.weight.shape),
                dtype=torch.float32,
                device="cuda:0",
            ),
        )

        if self.bias is not None:
            self.bias_sigma = torch.full(
                size=(self.bias.shape),
                fill_value=sigma,
                dtype=torch.float32,
                device="cuda:0",
            )

            self.register_buffer(
                "bias_mu",
                torch.zeros(
                    size=(self.bias.shape),
                    dtype=torch.float32,
                    device="cuda:0",
                ),
            )
        else:
            self.bias_sigma = None
            self.bias_mu = None

        if "grad" in self.pert_type.lower():

            self.grad_w_est = torch.zeros(
                size=(self.weight.shape),
                dtype=torch.float32,
                device="cuda:0",
            )

            if self.bias is not None:

                self.grad_b_est = torch.zeros(
                    size=(self.bias.shape),
                    dtype=torch.float32,
                    device="cuda:0",
                )

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
                self.batch_size,
            )

            self.seed = seed

            self.square_norm = square_norm

        else:  # Do not perturb if weight are not being trained.
            output = F.linear(input, self.weight, self.bias)
        return output

    @torch.inference_mode()
    def update_grads(self, scaling_factor):
        # Rescale grad data - to be used at end of gradient pass
        scaling_factor = scaling_factor.transpose(0, 1)

        # Scaling factor \in [batch, num_pert]
        # weight \in [batch OR pert, out, in]

        torch.manual_seed(self.seed)

        # Set gradients for weights

        w_noise_shape = (
            [self.batch_size] + list(self.weight.shape)
            if self.sample_wise
            else [self.num_perts] + list(self.weight.shape)
        )
        if self.sample_wise:
            scaled_weight_diff = (
                scaling_factor[:, :, None, None]
                * WPLinearFunc.sample_noise(
                    self.dist_sampler,
                    w_noise_shape,
                    self.weight_sigma,
                    self.weight_mu,
                    self.mu_scaling_factor,
                )[:, None, :, :]
            )
        else:
            scaled_weight_diff = (
                scaling_factor[:, :, None, None]
                * WPLinearFunc.sample_noise(
                    self.dist_sampler,
                    w_noise_shape,
                    self.weight_sigma,
                    self.weight_mu,
                    self.mu_scaling_factor,
                )[None, :, :, :]
            )
        # scaled weight diff has shape batch, num pert, output shape, input shape
        self.weight.grad = torch.sum(torch.mean(scaled_weight_diff, axis=1), dim=0)

        # Set gradients for biases
        if self.bias is not None:
            b_noise_shape = (
                [self.batch_size] + list(self.bias.shape)
                if self.sample_wise
                else [self.num_perts] + list(self.bias.shape)
            )

            if self.sample_wise:
                scaled_bias_diff = (
                    scaling_factor[:, :, None]
                    * WPLinearFunc.sample_noise(
                        self.dist_sampler,
                        b_noise_shape,
                        self.bias_sigma,
                        self.bias_mu,
                        self.mu_scaling_factor,
                    )[:, None, :]
                )
            else:
                scaled_bias_diff = (
                    scaling_factor[:, :, None]
                    * WPLinearFunc.sample_noise(
                        self.dist_sampler,
                        b_noise_shape,
                        self.bias_sigma,
                        self.bias_mu,
                        self.mu_scaling_factor,
                    )[None, :, :]
                )
            self.bias.grad = torch.sum(torch.mean(scaled_bias_diff, axis=1), dim=0)

        if "meta" in self.pert_type.lower():

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

        elif "grad" in self.pert_type.lower():

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
