"""Weight perturbation core functionality"""

import torch
import torch.nn.functional as F


class WPLinearFunc(torch.autograd.Function):
    """Linear layer with noise injection at weights"""

    @staticmethod
    def forward(
        ctx,
        input,
        weights,
        biases,
        sigmas,
        pert_type,
        dist_sampler,
        sample_wise,
        pert_num,
        batch_size,
    ):

        if sample_wise:
            assert pert_num == 1, "Don't use multiple perturbations for a fixed noise"
        assert pert_num > 0, "Number of perturbations should never be zero"

        output = torch.mm(input, weights.t())
        if biases is not None:
            output += biases

        if (
            sample_wise
        ):  # Whether the noise is unique or shared for each element of the batch, determined by sample_wise
            noise_shape = batch_size
        else:
            noise_shape = pert_num

        sigmas = sigmas.repeat(
            noise_shape, 1, 1
        )  # noise shape should be copied, input and output size kept the same
        print(sigmas.shape)
        noise_shape = [noise_shape] + list(weights.shape)
        print(noise_shape)
        w_noise = dist_sampler(noise_shape) * sigmas
        if pert_type.lower() == "forw":
            assert batch_size > 0
            output[batch_size:] += WPLinearFunc.add_noise(
                input[batch_size:], w_noise, sample_wise
            )
            if biases is not None:
                b_noise = dist_sampler([noise_shape[0]] + list(biases.shape))
                if sample_wise:
                    output[batch_size:] += b_noise
                else:
                    output[batch_size:] += torch.tile(b_noise, (batch_size, 1))

        elif pert_type.lower() == "cent":
            half = batch_size * pert_num
            output[:half] += WPLinearFunc.add_noise(input[:half], w_noise, sample_wise)
            output[half:] += WPLinearFunc.add_noise(input[half:], -w_noise, sample_wise)

            if biases is not None:
                b_noise = dist_sampler([noise_shape[0]] + list(biases.shape))

                if sample_wise:
                    output[:half] += b_noise
                    output[half:] -= b_noise
                else:
                    output[:half] += torch.tile(b_noise, (batch_size, 1))
                    output[half:] -= torch.tile(b_noise, (batch_size, 1))

        else:
            raise ValueError("Other perturbation types not yet implemented.")

        return output, w_noise, b_noise

    @staticmethod
    def add_noise(inputs: torch.Tensor, noisy_weights: torch.Tensor, sample_wise: bool):
        """Adds noise to the weights. If sample_wise is true, noise is assumed to be unique for each element"""

        if sample_wise:
            assert (
                noisy_weights.shape[0] == inputs.shape[0]
            ), "Shape of perturbations should be same as inputs!"
            outputs = torch.einsum("noi,ni->no", noisy_weights[0], inputs)
        else:
            pert_num = noisy_weights.shape[0]
            elements_per_batch = int(inputs.shape[0] / noisy_weights.shape[0])
            reshaped_inputs = inputs.view(pert_num, elements_per_batch, -1)
            outputs = torch.einsum(
                "boi,bni->bno", noisy_weights, reshaped_inputs
            ).reshape(pert_num * elements_per_batch, -1)

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None


class WPLinear(torch.nn.Linear):
    """Weight Perturbation layer with saved noise"""

    def __init__(
        self,
        *args,
        pert_type: str = "forw",
        dist_sampler: torch.distributions.Distribution = None,
        sigmas: torch.tensor,
        sample_wise: bool = False,
        num_perts: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.pert_type = pert_type
        self.dist_sampler = dist_sampler
        self.sample_wise = sample_wise
        self.square_norm = None
        self.weight_diff = None
        self.bias_diff = None
        self.num_perts = num_perts
        self.sigmas = torch.nn.parameter.Parameter(sigmas, requires_grad=True)

    def __str__(self):
        return "WPLinear"

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # A clean and noisy input are both processed by a layer to produce
        if self.training:

            if self.pert_type.lower() == "forw":
                batch_size = int(input.shape[0] / (self.num_perts + 1))
            elif self.pert_type.lower() == "cent":
                batch_size = int(input.shape[0] / (self.num_perts * 2))

            (output, weight_diff, bias_diff) = WPLinearFunc().apply(
                input,
                self.weight,
                self.bias,
                self.sigmas,
                self.pert_type,
                self.dist_sampler,
                self.sample_wise,
                self.num_perts,
                batch_size,
            )

            self.weight_diff = (
                weight_diff  # dims are number of perturbations, outputs, inputs
            )
            self.bias_diff = bias_diff  # dims are number of perturbations, outputs

            if self.sample_wise:
                noise_dim = batch_size
            else:
                noise_dim = self.num_perts

            self.square_norm = torch.sum(
                (self.weight_diff.reshape(noise_dim, -1)) ** 2, axis=1
            )

            if self.bias is not None:
                self.square_norm += torch.sum(
                    self.bias_diff.reshape(noise_dim, -1) ** 2, axis=1
                )

        else:  # Do not perturb if weights are not being trained.
            output = F.linear(input, self.weight, self.bias)
        return output

    @torch.inference_mode()
    def update_grads(self, scaling_factor):
        # Rescale grad data - to be used at end of gradient pass
        scaling_factor = scaling_factor.transpose(0, 1)

        # Scaling factor \in [batch, num_pert]
        # Weights \in [batch OR pert, out, in]

        self.sigmas.grad = torch.full(size=(self.sigmas.shape), fill_value=8000.08)

        if self.sample_wise:
            scaled_weight_diff = (
                scaling_factor[:, :, None, None] * self.weight_diff[:, None, :, :]
            )
        else:
            scaled_weight_diff = (
                scaling_factor[:, :, None, None] * self.weight_diff[None, :, :, :]
            )

        self.weight.grad = torch.sum(torch.mean(scaled_weight_diff, axis=1), dim=0)

        if self.bias is not None:
            if self.sample_wise:
                scaled_bias_diff = (
                    scaling_factor[:, :, None] * self.bias_diff[:, None, :]
                )
            else:
                scaled_bias_diff = (
                    scaling_factor[:, :, None] * self.bias_diff[None, :, :]
                )

            self.bias.grad = torch.sum(torch.mean(scaled_bias_diff, axis=1), dim=0)

    def get_noise_squarednorm(self):
        assert self.square_norm is not None, "square_norm has not been computed"
        return self.square_norm

    def get_number_perturbed_params(self):
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
