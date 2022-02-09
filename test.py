import unittest
import torch
import numpy as np
import torch.nn.functional as F
from ssnt_loss import (
    ssnt_loss,
    ssnt_loss_mem,
    log_exclusive_cumprod,
    log1mexp,
    lengths_to_padding_mask
)

import hypothesis.strategies as st
from hypothesis import assume, given, settings
from torch.testing._internal.common_utils import TestCase

TEST_CUDA = torch.cuda.is_available()


class SSNTLossTest(TestCase):
    def _test_ssnt_loss_ref(
        self,
        log_probs,
        targets,
        log_p_choose,
        source_lengths,
        target_lengths,
        neg_inf: float = -1e4,
        reduction="none",
        return_lattice=False
    ):
        """ raw array test """
        bsz, tgt_len, src_len = log_p_choose.size()

        logcumprod_1mp = log_exclusive_cumprod(
            log1mexp(log_p_choose), dim=-1)

        source_padding_mask = lengths_to_padding_mask(source_lengths)
        if source_padding_mask.any():
            log_p_choose = log_p_choose.masked_fill(source_padding_mask.unsqueeze(1), neg_inf)

        p_a1 = log_p_choose.new_zeros([bsz, src_len])
        p_j_jm1 = log_p_choose.new_zeros([bsz, tgt_len, src_len, src_len])
        log_alpha = log_p_choose.new_zeros([bsz, tgt_len, src_len])
        output = log_p_choose.new_zeros([bsz])
        for t in range(bsz):
            for i in range(src_len):
                p_a1[t, i] = logcumprod_1mp[t, 0, i] + log_p_choose[t, 0, i]

            for j in range(tgt_len):
                for i in range(src_len):
                    for k in range(i + 1):
                        if k < i:
                            x = log_p_choose[t, j, i] + logcumprod_1mp[t, j, i] - logcumprod_1mp[t, j, k]
                        else:
                            x = log_p_choose[t, j, i]
                        p_j_jm1[t, j, i, k] = x

            for i in range(src_len):
                y = targets[t, 0]
                log_alpha[t, 0, i] = p_a1[t, i] + log_probs[t, 0, i, y]

            for j in range(1, tgt_len):
                for i in range(src_len):
                    y = targets[t, j]
                    agg = []
                    for k in range(i + 1):
                        agg.append(
                            log_alpha[t, j - 1, k] + p_j_jm1[t, j, i, k]
                        )
                    x = torch.stack(agg, dim=0).logsumexp(0)
                    log_alpha[t, j, i] = x + log_probs[t, j, i, y]

            output[t] = log_alpha[t, target_lengths[t] - 1, source_lengths[t] - 1]

        if return_lattice:
            return log_alpha

        if reduction == "sum":
            output = output.sum()
        elif reduction == "mean":
            output = output.mean()

        return -output

    def _test_custom_ssnt_loss_impl(
        self, *args, **kwargs
    ):
        return ssnt_loss(*args, **kwargs)

    def _test_custom_ssnt_loss_mem_impl(
        self, *args, **kwargs
    ):
        return ssnt_loss_mem(*args, **kwargs)

    @settings(deadline=None)
    @given(
        B=st.integers(1, 10),
        T=st.integers(1, 20),
        S=st.integers(1, 200),
        V=st.integers(1, 20),
        device=st.sampled_from(["cpu", "cuda"]),
    )
    def test_ssnt_loss(self, B, T, S, V, device):

        assume(device == "cpu" or TEST_CUDA)

        # inputs
        lattice = torch.rand(B, T, S, V, device=device).log_softmax(-1)
        log_emit = F.logsigmoid(torch.rand(B, T, S, device=device))
        targets = torch.randint(0, V, (B, T), device=device)
        source_lengths = torch.full((B,), S, device=device)
        target_lengths = torch.full((B,), T, device=device)

        inputs = {
            "log_probs": lattice,
            "targets": targets,
            "log_p_choose": log_emit,
            "source_lengths": source_lengths,
            "target_lengths": target_lengths,
            "return_lattice": True
        }

        # test normal ver
        y = self._test_ssnt_loss_ref(**inputs).cpu().detach().numpy()
        x = self._test_custom_ssnt_loss_impl(**inputs).cpu().detach().numpy()
        np.testing.assert_allclose(
            x,
            y,
            atol=1e-3,
            rtol=1e-3,
        )

        # compare loss
        inputs = {
            "log_probs": lattice,
            "targets": targets,
            "log_p_choose": log_emit,
            "source_lengths": source_lengths,
            "target_lengths": target_lengths,
            "reduction": "none"
        }

        # test normal ver
        y = self._test_ssnt_loss_ref(**inputs).cpu().detach().numpy()
        x = self._test_custom_ssnt_loss_impl(**inputs).cpu().detach().numpy()
        np.testing.assert_allclose(
            x,
            y,
            atol=1e-3,
            rtol=1e-3,
        )
        # test mem ver
        self.convert_mem(inputs, target_lengths)
        z = self._test_custom_ssnt_loss_mem_impl(**inputs).cpu().detach().numpy()
        np.testing.assert_allclose(
            z,
            y,
            atol=1e-3,
            rtol=1e-3,
        )

    def convert_mem(self, inputs, target_lengths):
        mask = ~lengths_to_padding_mask(target_lengths)
        for k in "log_probs", "targets", "log_p_choose":
            inputs[k] = inputs[k][mask]
        return inputs, mask


if __name__ == "__main__":
    unittest.main()
