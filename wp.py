"""Weight perturbation core functionality"""

import torch
import torch.nn.functional as F
import scipy
import utils


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
            dist_sampler,
            w_noise_shape,
            weight_sigma,
            weight_mu,
            orthogonal_perts,
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

        square_norm = torch.sum(torch.pow(w_noise, 2), dim=(1, 2))

        # Perturb biases
        if bias is not None:
            b_noise_shape = [num_perts] + list(bias.shape)
            b_noise = WPLinearFunc.sample_noise(
                dist_sampler,
                b_noise_shape,
                bias_sigma,
                bias_mu,
                orthogonal_perts,
            )

            if "ffd" in pert_type.lower():
                output[batch_size:] += torch.tile(b_noise, (batch_size, 1))

            elif "cfd" in pert_type.lower():

                output[:halfway] += torch.tile(b_noise, (batch_size, 1))
                output[halfway:] -= torch.tile(b_noise, (batch_size, 1))
            else:
                raise ValueError("Other bias perturbation types not yet implemented.")
            square_norm += torch.sum(torch.pow(b_noise, 2), dim=1)

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
    def sample_noise(sampler, shape, sigma, mu, orthogonal_perts: bool = False):
        dims = torch.ones(len(shape), dtype=torch.int8).tolist()
        dims[0] = shape[0]
        if orthogonal_perts:
            noise = torch.nn.init.orthogonal_(
                torch.ones(size=shape, device="cuda:0"), gain=sigma
            )  # gain=sigma
            noise = noise + mu.repeat(dims)
        else:
            noise = sampler(shape) * sigma.repeat(dims) + mu.repeat(dims)
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
        self.orthogonal_perts = orthogonal_perts
        self.meta_lr = meta_lr  # the meta learning rate. How much of the past gradient estimate is carried over as momentum

        self.mu_scaling_factor = torch.tensor(
            mu_scaling_factor, dtype=torch.float32, device=device
        )
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
                    self.dist_sampler,
                    w_noise_shape,
                    self.weight_sigma,
                    self.weight_mu,
                    self.orthogonal_perts,
                )[None, :, :, :],
            )  # [batch, pert, in] * [1, pert, out, in]

            self.weight.grad = torch.sum(torch.mean(scaled_weight_diff, axis=0), dim=0)

            scaling_factor = torch.sum(torch.mean(scaling_factor, dim=-1), dim=0)

        if "greedy" in self.pert_type.lower():
            torch.manual_seed(self.seed)

            scaling_factor = torch.sum(scaling_factor, dim=0)
            max_grad_index = torch.argmax(scaling_factor)
            scaling_factor = scaling_factor[max_grad_index]
            w_noise = WPLinearFunc.sample_noise(
                self.dist_sampler,
                w_noise_shape,
                self.weight_sigma,
                self.weight_mu,
                self.orthogonal_perts,
            )[max_grad_index]

            self.weight.grad = torch.mul(
                scaling_factor,
                w_noise,
            )  # [1] * [out, in]

            if self.bias is not None:
                b_noise_shape = [self.num_perts] + list(self.bias.shape)
                b_noise = WPLinearFunc.sample_noise(
                    self.dist_sampler,
                    b_noise_shape,
                    self.bias_sigma,
                    self.bias_mu,
                    self.orthogonal_perts,
                )[max_grad_index]

                self.bias.grad = torch.mul(
                    scaling_factor,
                    b_noise,
                )
        else:
            scaling_factor = torch.sum(scaling_factor, dim=0)
            scaled_weight_diff = torch.mul(
                scaling_factor[:, None, None],
                WPLinearFunc.sample_noise(
                    self.dist_sampler,
                    w_noise_shape,
                    self.weight_sigma,
                    self.weight_mu,
                    self.orthogonal_perts,
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
                        self.dist_sampler,
                        b_noise_shape,
                        self.bias_sigma,
                        self.bias_mu,
                        self.orthogonal_perts,
                    ),
                )

                self.bias.grad = torch.mean(scaled_bias_diff, axis=0)

        if "meta" in self.pert_type.lower():

            if self.first_gradient:
                self.weight_mu = self.weight.grad * self.mu_scaling_factor

                if self.bias is not None:
                    self.bias_mu = self.bias.grad * self.mu_scaling_factor
                self.first_gradient = False

            else:

                self.weight_mu = (
                    torch.mul(self.weight_mu, self.meta_lr) + self.weight.grad
                ) * self.mu_scaling_factor

                if self.bias is not None:
                    self.bias_mu = (
                        torch.mul(self.bias_mu, self.meta_lr) + self.bias.grad
                    ) * self.mu_scaling_factor

        if "meta-pert" in self.pert_type.lower():
            torch.manual_seed(self.seed)

            b_noise_shape = [self.num_perts] + list(self.bias.shape)

            if self.first_gradient:
                self.weight_mu = torch.mean(
                    WPLinearFunc.sample_noise(
                        self.dist_sampler,
                        b_noise_shape,
                        self.bias_sigma,
                        self.bias_mu,
                        self.orthogonal_perts,
                    ),
                    dim=0,
                ) * torch.sign(self.weight.grad)

                if self.bias is not None:

                    self.bias_mu = torch.mean(
                        WPLinearFunc.sample_noise(
                            self.dist_sampler,
                            b_noise_shape,
                            self.bias_sigma,
                            self.bias_mu,
                            self.orthogonal_perts,
                        ),
                        dim=0,
                    ) * torch.sign(self.bias.grad)
                self.first_gradient = False

            else:
                reg_term = 0.1

                self.weight_mu = torch.mul(self.weight_mu, self.meta_lr) + torch.mean(
                    WPLinearFunc.sample_noise(
                        self.dist_sampler,
                        b_noise_shape,
                        self.bias_sigma,
                        self.bias_mu,
                        self.orthogonal_perts,
                    ),
                    dim=0,
                )

                if self.bias is not None:
                    self.bias_mu = (
                        torch.mul(self.bias_mu, self.meta_lr)
                        + torch.mean(
                            WPLinearFunc.sample_noise(
                                self.dist_sampler,
                                b_noise_shape,
                                self.bias_sigma,
                                self.bias_mu,
                                self.orthogonal_perts,
                            ),
                            dim=0,
                        )
                        * torch.sign(self.bias.grad)
                        - self.bias_mu * reg_term
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
