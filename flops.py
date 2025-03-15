import torch
import wp
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any
from numbers import Number
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
import utils
import wandb

aten = torch.ops.aten


def get_shape(i):
    return i.shape


def prod(x):
    res = 1
    for i in x:
        res *= i
    return res


def addmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    # Count flops for nn.Linear
    # inputs is a list of length 3 - bias, inputs [batch size, input feature dimension], weights  [input feature dimension, output feature dimension].
    input_shapes = [get_shape(v) for v in inputs]

    assert len(input_shapes[1]) == 2, input_shapes[0]
    assert len(input_shapes[2]) == 2, input_shapes[1]
    assert (
        input_shapes[2][1] == input_shapes[0][0]
    ), "output dimension and the number of biases should be the same"
    batch_size, input_dim = input_shapes[1]
    output_dim = input_shapes[2][1]
    flops = (
        batch_size * output_dim + 2 * batch_size * input_dim * output_dim
    )  # bias + matrix multiplication

    return flops


def bmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # not made by Martin
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop


def mul_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for torch.mul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    assert len(inputs) == 2, inputs
    in1 = torch.tensor(inputs[0])
    in2 = torch.tensor(inputs[1])
    input_shapes = torch.cat((torch.tensor(in1.shape), torch.tensor(in2.shape)))
    input_shapes = input_shapes.unique()
    return torch.prod(input_shapes.unique())


def matmul_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = 2 * prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop


def sum_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the sum operation.
    """
    # check if it gives me the index which it sums over -> compare outputs to the inputs
    if inputs[0].squeeze().shape != outputs[0].squeeze().shape:
        flops = int(torch.prod(torch.tensor(inputs[0].shape)))
    else:
        flops = 0
    return flops


def einsum_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the einsum operation.
    """
    # input is tuple of (einsum specification, [ tensor 1, tensor 2])

    input_shapes = [get_shape(v) for v in inputs[1][:]]
    assert len(input_shapes) == 2, input_shapes
    assert (
        inputs[0] == "boi,bni->bno"
    ), "einsum flops are only implented for boi,bni->bno"

    # torch.Size([1, 10, 3072]) perturbation
    # torch.Size([1, 1, 3072]) input
    # torch.Size([1, 1, 10])

    flop = (
        2 * prod(input_shapes[0]) * input_shapes[-1][-2]
    )  # 2 * (num_perts * output_size * number of weights) * batch_size

    return flop


def add_flop(inputs: List[Any], outputs: List[Any]) -> Number:

    # Count flops for the sum operation.
    # Had to remove because of flop counting counting itself

    shape1 = prod(inputs[0].shape) if not isinstance(inputs[1], int) else 1
    shape2 = prod(inputs[1].shape) if not isinstance(inputs[1], int) else 1

    flops = max(shape1, shape2)

    return flops if flops > 1 else 0


def get_inputs(inputs: List[Any], outputs: List[Any]) -> Number:
    print(inputs)
    print(outputs)
    return 0


def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])


flop_mapping = {
    aten.mm: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.sum: sum_flop,
    aten.mul: mul_flop,
    aten.mean: sum_flop,
    aten.einsum: einsum_flop,
    aten.add_: add_flop,
    # aten.expand: get_inputs,
    # aten._log_softmax_backward_data: get_inputs,
    # aten.nll_loss_backward: get_inputs,
}


# Note that division was excluded, because it caused the flop counter to count itself, and because the only operation where it appears is equal to the number of perturbations.


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


class FlopCounterMode(TorchDispatchMode):
    # credit for this implementation goes to Horace He https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505
    # While the FLOPS functions were changed/expanded, the core code was taken from: https://pastebin.com/V3wATa7w

    def __init__(self, network=None, loud=False):
        self.flop_counts = defaultdict(lambda: defaultdict(int))
        self.parents = ["Global"]
        self.loud = loud
        if network is not None:
            for name, module in dict(network.named_children()).items():
                module.register_forward_pre_hook(self.enter_module(name))
                module.register_forward_hook(self.exit_module(name))

    def enter_module(self, name):
        def f(module, inputs):
            self.parents.append(name)
            inputs = normalize_tuple(inputs)
            out = self.create_backwards_pop(name)(*inputs)
            return out

        return f

    def exit_module(self, name):
        def f(module, inputs, outputs):
            assert self.parents[-1] == name
            self.parents.pop()
            outputs = normalize_tuple(outputs)
            return self.create_backwards_push(name)(*outputs)

        return f

    def create_backwards_push(self, name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(
                    lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args
                )
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.parents.append(name)
                return grad_outs

        return PushState.apply

    def create_backwards_pop(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(
                    lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args
                )
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                assert self.parents[-1] == name
                self.parents.pop()
                return grad_outs

        return PopState.apply

    def __enter__(self):
        self.flop_counts.clear()
        super().__enter__()

    def __exit__(self, *args):
        print(f"Total: {sum(self.flop_counts['Global'].values()) } FLOPS")
        for mod in self.flop_counts.keys():
            print(f"Module: ", mod)
            for k, v in self.flop_counts[mod].items():
                print(f"{k}: {v} FLOPS")
            print()

        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if self.loud:
            print(func_packet)
        if func_packet in flop_mapping:
            flop_count = flop_mapping[func_packet](args, normalize_tuple(out))
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count

        return out


def BP_linear_flops(layer, num_perts, algorithm):
    # Function to count flops of nn.Linear. Does not associate any costs with calculating the loss or the optimizer step.
    weight_shape = torch.tensor(layer.weight.shape)
    bias_shape = torch.tensor(layer.bias.shape)

    forward_pass_flops = (
        torch.prod(weight_shape) * 2 + bias_shape
    )  # assumes a batch size of 1, for a matrix multiplication of [1, input] vs [input, output]. Bias is also added, which costs [output] flops.
    backward_pass_flops = torch.prod(weight_shape) * 2 + bias_shape
    return (
        forward_pass_flops,
        backward_pass_flops,
        forward_pass_flops + backward_pass_flops,
    )


def WP_linear_flops(layer, num_perts, algorithm):
    # Function to count flops of nn.Linear. Does not associate any costs with calculating the loss, the optimizer step or how the perturbations are sampled. It does however include the cost of adding the perturbations

    weight_shape = torch.tensor(layer.weight.shape)
    bias_shape = torch.tensor(layer.bias.shape)
    if algorithm.lower() == "ffd":
        num_forw_passes = (
            num_perts + 1
        )  # ffd consists of one clean pass, plus num_pert passes.
        weight_perturbation = num_perts * torch.prod(
            weight_shape
        )  # the cost of adding perturbations to the weights
        bias_perturbation = num_perts * bias_shape
    elif algorithm.lower() == "cfd":
        num_forw_passes = num_perts * 2  # cfd needs two forward passes per perturbation
        weight_perturbation = num_perts * torch.prod(weight_shape) * 2
        bias_perturbation = num_perts * bias_shape * 2
    else:
        raise ValueError("only ffd and cfd flops are currently supported")

    forward_pass_flops = (
        (torch.prod(weight_shape) * 2 * num_forw_passes)
        + weight_perturbation
        + bias_perturbation
        + bias_shape
    )  # Matrix multiplication with shapes [1, input] vs [num_forw_passes, input, output]. Assumes a batch size of 1.

    num_perts = torch.tensor(num_perts).reshape(1)

    # Assumes that masking does not happen! Also ignores the cost of summing the loss over the batch.
    backward_pass_flops = (
        torch.prod(torch.cat((weight_shape, num_perts))) + num_perts * bias_shape
    )  # cost of elementwise multiplication, leading to a broadcasted matrix of [num_perts, input, output]

    if num_perts > 1:
        backward_pass_flops += (
            torch.prod(torch.cat((weight_shape, num_perts))) + num_perts * bias_shape
        )
        # cost of meaning operation for weights and biases. Assumes that pytorch does not actually perform meaning when there is only one element to mean over.
    print(forward_pass_flops)
    print(forward_pass_flops[0])

    print("this is WP linear layer")
    return (
        forward_pass_flops[0],
        backward_pass_flops[0],
        forward_pass_flops[0] + backward_pass_flops[0],
    )


layer_mapping = {torch.nn.Linear: BP_linear_flops, wp.WPLinear: WP_linear_flops}


def FLOP_abstract_track(algorithm, num_perts, network):

    forward_pass_flops = 0
    backward_pass_flops = 0
    total_flops = 0
    for layer in network.modules():
        if type(layer) in layer_mapping:
            flops = layer_mapping[type(layer)](layer, num_perts, algorithm)
            print(flops)
            forward_pass_flops, backward_pass_flops, total_flops = (
                forward_pass_flops + flops[0],
                backward_pass_flops + flops[1],
                total_flops + flops[2],
            )
    return [forward_pass_flops, backward_pass_flops, total_flops]


def FLOP_real_track(
    network,
    device,
    dataset,
    out_shape,
    loss_func,
):

    train_loader, _, _, out_shape = utils.construct_dataloaders(
        dataset, 1, device, validation=True
    )

    # print(dict(network.named_modules()))
    for batch_idx, (data, target) in enumerate(train_loader):
        onehots = (
            torch.nn.functional.one_hot(target, out_shape).to(device).to(data.dtype)
        )
        data, target = data.to(device), target.to(device)
        # _, loss_differential = network.forward_pass(data, target, onehots, loss_func)

        loss1, loss_differential = network.test_step(data, target, onehots, loss_func)
        # print("loss on clean FFD pass - test")
        # print(loss1)
        loss, loss_differential = network.forward_pass(data, target, onehots, loss_func)
        # print("loss on clean FFD pass - training")

        # print(loss)
        # print(torch.sqrt(torch.sum(torch.pow(torch.subtract(loss1, loss), 2))))

        if type(loss_differential) != torch.Tensor:
            loss_differential = torch.tensor(loss_differential)

        if batch_idx == 5:

            flop_counter = FlopCounterMode(network, loud=False)
            with flop_counter:
                network.train_step(data, target, onehots, loss_func)
                # network.forward_pass(data, target, onehots, loss_func)
                # network.backward_pass(loss_differential)
            exit(0)

            # wandb.log({"FLOPS": flops}, step=0)
