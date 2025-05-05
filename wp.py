"""Weight perturbation core functionality"""

import torch
import torch.nn.functional as F
import numpy as np
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

        # Initialize matrix of shape [batch size, output size]. Prefilled with biases, if they exist.
        if bias is not None:
            output = torch.tile(bias, (input.shape[0], 1))
        else:
            output = torch.zeros(size=(input.shape[0], weight.shape[0]), device=device)

        seed = torch.randint(-int(1e10), int(1e10), size=(1,))
        torch.manual_seed(seed)

        # Sample weight noise
        w_noise_shape = [num_perts] + list(weight.shape)
        w_noise = WPLinearFunc.sample_noise(
            dist_sampler, w_noise_shape, weight_sigma, weight_mu, orthogonal_perts
        )

        normalizer = 1.0
        if "dmomentum" in pert_type.lower():
            normalizer = torch.sqrt((weight_sigma**2) / (w_noise**2).mean(-1).mean(-1))
            w_noise *= normalizer.unsqueeze(-1).unsqueeze(-1)

        square_norm = (w_noise**2).sum(-1).sum(-1)

        # Weight perturbation
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

            if "dmomentum" in pert_type.lower():
                normalizer = torch.sqrt((bias_sigma**2) / (b_noise**2).mean(-1))
                b_noise *= normalizer.unsqueeze(-1)

            square_norm += (b_noise**2).sum(-1)

            if "ffd" in pert_type.lower():
                output[batch_size:] += torch.tile(b_noise, (batch_size, 1))

            elif "cfd" in pert_type.lower():

                output[:halfway] += torch.tile(b_noise, (batch_size, 1))
                output[halfway:] -= torch.tile(b_noise, (batch_size, 1))
            else:
                raise ValueError("Other bias perturbation types not yet implemented.")

        else:
            b_noise = None

        return output, seed, normalizer, square_norm

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
            noise = torch.nn.init.orthogonal_(torch.empty(size=shape, device="cuda:0"))

            if len(shape) == 2:
                noise = noise * (np.sqrt(shape[1]))
            if len(shape) == 3:
                noise = noise * (np.sqrt(shape[1] * shape[2]))

            noise = noise * sigma
        else:
            noise = sampler(shape) * sigma

        noise = torch.add(noise, mu.unsqueeze(0))

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
        self.weight_sigma = sigma

        self.register_buffer(
            "weight_mu",
            torch.zeros(
                size=(self.weight.shape),
                dtype=torch.float32,
                device=device,
            ),
        )

        if self.bias is not None:
            self.bias_sigma = sigma

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
            (output, seed, normalizer, square_norm) = WPLinearFunc().apply(
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
            self.normalizer = normalizer

        else:  # Do not perturb if weight are not being trained.
            output = F.linear(input, self.weight, self.bias)
        return output

    @torch.inference_mode()
    def update_grads(self, loss_differential):
        # Rescale grad data - to be used at end of gradient pass
        # loss_differential \in [num_perts, batch_size]
        # weight \in [num_pert, out, in]
        torch.manual_seed(self.seed)

        # if "dmomentum" in self.pert_type.lower():
        #     loss_differential /= self.normalizer.unsqueeze(-1)

        scaling_factor = loss_differential * (
            self.get_number_perturbed_params() / self.square_norm
        ).unsqueeze(
            1
        )  # normalize loss by the size of the noise sampled in this layer

        # resampling the noise
        w_noise_shape = [self.num_perts] + list(self.weight.shape)
        w_noise = WPLinearFunc.sample_noise(
            self.dist_sampler,
            w_noise_shape,
            self.weight_sigma,
            self.weight_mu,
            self.orthogonal_perts,
        )

        if self.bias is not None:
            b_noise_shape = [self.num_perts] + list(self.bias.shape)
            b_noise = WPLinearFunc.sample_noise(
                self.dist_sampler,
                b_noise_shape,
                self.bias_sigma,
                self.bias_mu,
                self.orthogonal_perts,
            )

        ### Different ways to estimate the gradient
        if (
            "greedy" in self.pert_type.lower()
        ):  # greedy selection of the best perturbation
            scaling_factor = torch.sum(scaling_factor, dim=1)
            max_grad_index = torch.argmax(torch.abs(scaling_factor))
            scaling_factor = scaling_factor[max_grad_index]
            w_noise = w_noise[max_grad_index]

            self.weight.grad = torch.mul(
                scaling_factor,
                w_noise,
            )  # [1] * [out, in]

            if self.bias is not None:
                b_noise = b_noise[max_grad_index]
                self.bias.grad = torch.mul(
                    scaling_factor,
                    b_noise,
                )

        elif "weighted" in self.pert_type.lower():  # weighted average of perturbations
            scaling_factor = torch.sum(scaling_factor, dim=1)
            ratio = torch.abs(scaling_factor / torch.sum(torch.abs(scaling_factor)))

            scaled_weight_diff = torch.mul(
                scaling_factor[:, None, None],
                w_noise,
            )

            self.weight.grad = torch.div(
                torch.sum(
                    torch.mul(scaled_weight_diff, ratio[:, None, None]), dim=0
                ),  # weighted sum
                torch.sum(ratio),  # weights
            )

            if self.bias is not None:
                scaled_bias_diff = torch.mul(
                    scaling_factor[:, None],
                    b_noise,
                )

                self.bias.grad = torch.div(  # weighted average formula (weights*x)/sum of weights -> should just be a division by 1.
                    torch.sum(
                        torch.mul(scaled_bias_diff, ratio[:, None]), dim=0
                    ),  # weighted sum
                    torch.sum(ratio),  # weights
                )
        else:  # average of multiple perturbations
            scaling_factor = torch.sum(scaling_factor, dim=1)
            scaled_weight_diff = torch.mul(
                scaling_factor[:, None, None],
                w_noise,
            )
            self.weight.grad = torch.mean(scaled_weight_diff, axis=0)
            # scaled weight diff has shape num pert, output shape, input shape

            # Set gradients for biases
            if self.bias is not None:

                scaled_bias_diff = torch.mul(
                    scaling_factor[:, None],
                    b_noise,
                )

                self.bias.grad = torch.mean(scaled_bias_diff, axis=0)
        ### Keeping track of past gradients for use.
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

        if "dmomentum" in self.pert_type.lower():

            if self.first_gradient:
                weighting = scaling_factor / torch.sum(scaling_factor)
                self.weight_mu = (
                    torch.mean(
                        weighting[:, None, None] * w_noise,
                        dim=0,
                    )
                    * self.meta_lr
                )  # Multiplied by num params / square norm (for this layer)

                # Note that this works because you can move the normalizer from the net py code (where it is multiplied by the scaling factor) and instead multiply it by the noise here and get the same answer
                if self.bias is not None:
                    self.bias_mu = (
                        torch.mean(
                            weighting[:, None] * b_noise,
                            dim=0,
                        )
                        * self.meta_lr
                    )
                self.first_gradient = False

            else:
                weighting = scaling_factor / torch.sum(torch.abs(scaling_factor))

                self.weight_mu = (
                    self.weight_mu
                    + torch.mean(
                        weighting[:, None, None]
                        * w_noise,  # Multiplied by num params / square norm (for this layer)
                        dim=0,
                    )
                ) * self.meta_lr

                if self.bias is not None:
                    self.bias_mu = (
                        self.bias_mu
                        + torch.mean(
                            weighting[:, None]
                            * b_noise,  # Multiplied by num params / square norm (for this layer)
                            dim=0,
                        )
                    ) * self.meta_lr

    def get_noise_squarednorm(self):
        assert self.square_norm is not None, "square_norm has not been computed"
        return self.square_norm

    def get_number_perturbed_params(self):
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
