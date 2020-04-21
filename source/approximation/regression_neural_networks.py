from __future__ import annotations

import json
from typing import Dict, Any, Optional, Sequence

import numpy
import torch
from torch.nn import Module

from source.approximation.abstract import Approximation, INPUT_VALUE, OUTPUT_VALUE


class TorchModel(Approximation[Sequence[float], Sequence[float]]):
    def fit(self, in_value: INPUT_VALUE, target_value: OUTPUT_VALUE, drag: int):
        pass

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> TorchModel:
        dim_in = d["dim_in"]
        dim_out = d["dim_out"]
        weights = d["weights"]
        device_str = d["device_str"]
        iterations = d["iterations"]

        torch_model = TorchModel(dim_in, dim_out, weights=weights, device_str=device_str, iterations=iterations)

        state_dict_model = d["model"]
        torch_model.model.load_state_dict(state_dict_model)
        torch_model.model.eval()

        state_dict_optimizer = d["optimizer"]
        torch_model.optimizer.load_state_dict(state_dict_optimizer)

        return torch_model

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v.state_dict() if k in ("model", "optimizer") else v
            for k, v in self.__dict__.items()
            if k not in ("device", "loss_fn", "optimizer")
        }

    def __init__(self,
                 dim_in: int, dim_out: int,
                 weights: Optional[Sequence[float]] = None, device_str: str = "gpu", iterations: int = 10,
                 activation: Module = torch.nn.ReLU,
                 learning_rate: float = .1,):
        super().__init__(dim_in, dim_out)
        assert 0 < iterations

        self.device_str = device_str
        self.device = torch.device(device_str)  # "cuda:0" https://github.com/huggingface/transformers/issues/227
        self.iterations = iterations

        # https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_in, 32), activation(),
            torch.nn.Linear(32, 16), activation(),
            torch.nn.Linear(16, dim_out), torch.nn.Softmax(dim=1),
        )

        self.weights = weights
        if weights is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            weights_tensor = torch.from_numpy(numpy.array(weights)).float().to(self.device)
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

        self.learning_rate = learning_rate

        # todo: replace by adam!
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _validate(self, inputs: numpy.ndarray, targets: numpy.ndarray) -> float:
        error = 0.
        for input_example, target_example in zip(inputs, targets):
            output_example = self.output(input_example)
            error += ((target_example - output_example) ** 2).sum()
        return error

    def fit(self,
                    keys: Sequence[str], inputs: numpy.ndarray, targets: numpy.ndarray,
                    path_output: Optional[str],
                    function_class_balancing: Callable[[numpy.ndarray], float],
                    error_key: Callable[[numpy.ndarray, numpy.ndarray], float]) -> Sequence[float]:

        no_examples = len(keys)
        ratio_validation = .1
        indices_random = numpy.random.permutation(no_examples)

        size_validation = round(no_examples * ratio_validation)
        size_train = no_examples - size_validation

        indices_train, indices_validation = indices_random[:size_train], indices_random[size_train:]

        inputs_train, targets_train = inputs[indices_train, :], targets[indices_train, :]
        inputs_validation, targets_validation = inputs[indices_validation, :], targets[indices_validation, :]

        error_validation_min = -1.
        model_best = None

        losses = []
        for _i in range(self.iterations):
            in_array = torch.from_numpy(inputs_train).float().to(self.device)
            target_array = torch.from_numpy(targets_train).float().to(self.device)

            out_array = self.model(in_array)

            # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
            loss = self.loss_fn(
                out_array,
                torch.from_numpy(
                    numpy.array(
                        [_target.argmax() for _target in target_array]
                    )
                )
            )    # https://pytorch.org/docs/stable/nn.html#crossentropyloss

            self.model.zero_grad()

            loss.backward()

            self.optimizer.step()

            each_loss = loss.tolist()   # doesnt return list?
            losses.append(each_loss)

            each_error = self._validate(inputs_validation, targets_validation)
            if error_validation_min < 0. or each_error < error_validation_min:
                message = f"Validation error reduced from {error_validation_min:.2f} to {each_error:.2f} in iteration {_i + 1:d} of {self.iterations:d}. Storing new model..."
                Logger.log(message)
                model_best = copy.deepcopy(self.model)
                error_validation_min = each_error

        if model_best is not None:
            self.model = model_best

        return losses

    def output(self, input_value: Sequence[float]) -> Sequence[float]:
        input_example = numpy.array(input_value)
        in_reshaped = input_example.reshape((1, *input_example.shape))
        in_array = torch.from_numpy(in_reshaped).float().to(self.device)
        out_array = self.model(in_array)
        numpy_out = out_array.detach().numpy()
        return list(numpy_out[0])
