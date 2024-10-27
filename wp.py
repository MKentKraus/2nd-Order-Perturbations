"""Weight perturbation core functionality"""

import torch
import torch.nn.functional as F



class WPLinearFunc(torch.autograd.Function):
    """Linear layer with noise injection at weights"""

    @staticmethod
    def forward(ctx, input, weights, biases, pert_type, dist_sampler, sample_wise, batch_size):
        # Matrix multiplying both the clean and noisy forward signals

        outputs = None
        bias_diffs = 0
        weight_diffs = 0
        if input.shape[0] > batch_size*2: #recursively call the function until the input is only as large as two passes
            outputs, weight_diffs, bias_diffs = WPLinearFunc.forward(ctx, input[batch_size*2:], weights, biases, pert_type, dist_sampler, sample_wise, batch_size)

        #recursion - save outputs, since none is apriori better than the other, and the loss should be computed on all of them
        input = input[:batch_size*2]

        output = torch.mm(input, weights.t())
        half_batch_width = batch_size // 2

        noise_dim = half_batch_width if sample_wise else 1 #Whether the noise is unique or shared for each element of the batch, determined by sample_wise

        # Determining the shape of the noise
        # Noise shape based upon weight shape and batch size
        noise_shape = list(weights.shape)
        if sample_wise:
            noise_shape.insert(0, noise_dim)


        ###Perturb original data
        if pert_type.lower() == "forw":
            w_noise_1 = torch.zeros(noise_shape, device=input.device)
        else:
            w_noise_1 = dist_sampler(noise_shape)
            output[:half_batch_width] += WPLinearFunc.add_noise(input[:half_batch_width], w_noise_1, sample_wise)

        ###Perturb copy of data
        if pert_type.lower() == "cent":
            w_noise_2 = torch.mul(w_noise_1, -1) #gets vector of same magnitude as w_noise_1 in the opposite direction
        else:
            w_noise_2 = dist_sampler(noise_shape)


        weight_diff = torch.sub(w_noise_1, w_noise_2)
        output[half_batch_width:] += WPLinearFunc.add_noise(input[half_batch_width:], w_noise_2, sample_wise)

        #perturbing the biases
        b_noise_1, b_noise_2 = None, None
        bias_diff = None
        if biases is not None:
            noise_shape = [noise_dim] + list(biases.shape)
            #Samples noise to add to biases. Second half is always perturbed
            if pert_type.lower() == "forw":
                b_noise_1 = torch.zeros(noise_shape, device=input.device)
            else:
                b_noise_1 = dist_sampler(noise_shape)

            if pert_type.lower() == "cent":                   
                b_noise_2 = torch.mul(b_noise_1, -1)
            else:
                b_noise_2 = dist_sampler(noise_shape)

            bias_diff = torch.sub(b_noise_1, b_noise_2)

            output[:half_batch_width] += biases + b_noise_1
            output[half_batch_width:] += biases + b_noise_2




        if outputs is not None: #clean this mess up
            if weight_diffs.ndim == 3:
                weight_diff = torch.cat( (weight_diffs, weight_diff.unsqueeze(0)))
                if biases is not None:
                    bias_diff = torch.cat( (bias_diffs, bias_diff.unsqueeze(0)))
            else:
                weight_diff = torch.stack( (weight_diff, weight_diffs))
                if biases is not None:
                    bias_diff = torch.stack( (bias_diff, bias_diffs))

            output = torch.cat( (output, outputs))
        return output, weight_diff, bias_diff


    @staticmethod
    def add_noise(weights: torch.Tensor, noise: torch.Tensor, sample_wise: bool):
            """Adds noise to the weights. If sample_wise is true, noise is assumed to be unique for each element"""
            if sample_wise:
                out = torch.einsum("ni,nki->nk", weights, noise)
            else:
                out = torch.einsum("ni,ki->nk", weights, noise)
            return out
    
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
        self.num_perts = num_perts*2

    def __str__(self):
        return "WPLinear"

    @torch.inference_mode()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # A clean and noisy input are both processed by a layer to produce
        if self.training:
                batch_size = input.shape[0] // self.num_perts
                (output, weight_diff, bias_diff) = WPLinearFunc().apply(
                    input,
                    self.weight,
                    self.bias,
                    self.pert_type,
                    self.dist_sampler,
                    self.sample_wise,
                    batch_size,
                )


                self.weight_diff = weight_diff
                self.bias_diff = bias_diff
                noise_dim = batch_size//2 if self.sample_wise else 1
                
                self.square_norm = torch.sum( #check properly if this works with more perturbations and sample wise noise additions
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

        if(self.weight_diff.ndim > 2):
            weight_diff = self.weight_diff[0]
            self.weight_diff = self.weight_diff[1:]

        else:
            weight_diff = self.weight_diff

        scaled_weight_diff = scaling_factor[:, None, None] * weight_diff
        self.weight.grad = torch.sum(scaled_weight_diff, axis=0)
        
        if self.bias is not None:
            if(self.weight_diff.ndim > 2):
                bias_diff = self.bias_diff[0]
                self.bias_diff = self.bias_diff[1:]

            else:
                bias_diff = self.bias_diff
            scaled_bias_diff = scaling_factor[:, None] * bias_diff
            self.bias.grad = torch.sum(scaled_bias_diff, axis=0)

    def get_noise_squarednorm(self):
        assert self.square_norm is not None, "square_norm has not been computed"
        return self.square_norm

    def get_number_perturbed_params(self):
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)


