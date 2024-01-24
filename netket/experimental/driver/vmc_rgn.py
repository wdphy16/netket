# Copyright 2021 The NetKet Authors - All rights reserved.
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

from typing import Callable

from netket.driver import VMC
from netket.jax import tree_cast

from netket.experimental.optimizer import RGN
from netket.experimental.optimizer.rgn import (
    centered_jacobian_and_mean,
    loss_grad_and_rhessian,
)


class VMC_RGN(VMC):
    def __init__(
        self,
        eps_schedule: Callable,
        diag_shift_schedule: Callable,
        mode: str,
        chunk_size: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.eps_schedule = eps_schedule
        self.diag_shift_schedule = diag_shift_schedule
        self.mode = mode
        self.chunk_size = chunk_size

    def _forward_and_backward(self):
        self.state.reset()

        con_samples, mels = self._ham.get_conn_padded(self.state.samples)

        def forward_fn(W, σ):
            return self.state._apply_fun({"params": W, **self.state.model_state}, σ)

        jac, jac_mean = centered_jacobian_and_mean(
            forward_fn,
            self.state.parameters,
            self.state.samples,
            self.mode,
            self.chunk_size,
        )
        self._loss_stats, self._loss_grad, rhessian = loss_grad_and_rhessian(
            forward_fn,
            self.state.parameters,
            self.state.samples,
            con_samples,
            mels,
            self.mode,
            self.chunk_size,
        )

        eps = self.eps_schedule(self.step_count)
        diag_shift = self.diag_shift_schedule(self.step_count)
        preconditioner = RGN(
            jac=jac,
            jac_mean=jac_mean,
            rhes=rhessian,
            grad=self._loss_grad,
            loss=self._loss_stats.mean,
            eps=eps,
            diag_shift=diag_shift,
            mode=self.mode,
            params=self.state.parameters,
        )

        self._dp = preconditioner(self._loss_grad)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = tree_cast(self._dp, self.state.parameters)

        return self._dp
