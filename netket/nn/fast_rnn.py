# Copyright 2022 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from flax import linen as nn
from flax.linen.dtypes import promote_dtype

from jax.nn.initializers import zeros

from netket.nn.rnn import GRULayer1D, LSTMLayer1D, RNNLayer, RNNLayer1D
from netket.utils import deprecate_dtype
from netket.utils.types import Array


class FastRNNLayer1D(RNNLayer):
    """
    Base class for 1D recurrent neural network layers with fast sampling.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """

    @nn.compact
    def update_site(self, inputs: Array, index: int) -> Array:
        """
        Applies the RNN cell to a batch of input sites at a given index,
        and stores the updated memories in the cache.

        Args:
          inputs: an input site with dimensions (batch, features).
          index: the index of the output site. The index of the input site should be `index - self.exclusive`.

        Returns:
          The output site with dimensions (batch, features).
        """
        batch_size = inputs.shape[0]
        recur_func = self._get_recur_func(inputs)

        inputs = promote_dtype(inputs, dtype=self.param_dtype)[0]

        _cell = self.variable(
            "cache", "cell", zeros, None, (batch_size, self.features), inputs.dtype
        )
        _hidden = self.variable(
            "cache", "hidden", zeros, None, (batch_size, self.features), inputs.dtype
        )

        cell, hidden = recur_func(inputs, _cell.value, _hidden.value)

        initializing = self.is_mutable_collection("params")
        if not initializing:
            _cell.value = cell
            _hidden.value = hidden

        return hidden

    def __call__(self, inputs: Array) -> Array:
        return RNNLayer1D.__call__(self, inputs)


@deprecate_dtype
class FastLSTMLayer1D(FastRNNLayer1D):
    """
    1D long short-term memory layer with fast sampling.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """

    def _get_recur_func(self, inputs):
        return LSTMLayer1D._get_recur_func(self, inputs)


@deprecate_dtype
class FastGRULayer1D(FastRNNLayer1D):
    """
    1D gated recurrent unit layer with fast sampling.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """

    def _get_recur_func(self, inputs):
        return GRULayer1D._get_recur_func(self, inputs)