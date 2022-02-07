# ssnt-loss

> :information_source: This is a WIP project. the implementation is still being tested.

A pure PyTorch implementation of the loss described in *"Online Segment to Segment Neural Transduction"*  https://arxiv.org/abs/1609.08194.

## Usage
There are two versions, a normal version and a memory efficient version. They should give the same output, please inform me if they don't.
```python
def ssnt_loss_mem(
    log_probs: Tensor,
    targets: Tensor,
    log_p_choose: Tensor,
    source_lengths: Tensor,
    target_lengths: Tensor,
    neg_inf: float = -1e4,
    reduction="mean",
):
    """The memory efficient implementation concatenates along the targets
    dimension to reduce wasted computation on padding positions.

    Assuming the summation of all targets in the batch is T_flat, then
    the original B x T x ... tensor is reduced to T_flat x ...

    The input tensors can be obtained by using target mask:
    Example:
        >>> target_mask = targets.ne(pad)   # (B, T)
        >>> targets = targets[target_mask]  # (T_flat,)
        >>> log_probs = log_probs[target_mask]  # (T_flat, S, V)

    Args:
        log_probs (Tensor): Word prediction log-probs, should be output of log_softmax.
            tensor with shape (T_flat, S, V)
            where T_flat is the summation of all target lengths,
            S is the maximum number of input frames and V is
            the vocabulary of labels.
        targets (Tensor): Tensor with shape (T_flat,) representing the
            reference target labels for all samples in the minibatch.
        log_p_choose (Tensor): emission log-probs, should be output of F.logsigmoid.
            tensor with shape (T_flat, S)
            where T_flat is the summation of all target lengths,
            S is the maximum number of input frames.
        source_lengths (Tensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        target_lengths (Tensor): Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.
        neg_inf (float, optional): The constant representing -inf used for masking.
            Default: -1e4
        reduction (string, optional): Specifies reduction. suppoerts mean / sum.
            Default: None.
    """
```
### Minimal example
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ssnt_loss import ssnt_loss_mem, lengths_to_padding_mask
B, S, H, T, V = 2, 100, 256, 10, 2000

# model
transcriber = nn.LSTM(input_size=H, hidden_size=H, num_layers=1).cuda()
predictor = nn.LSTM(input_size=H, hidden_size=H, num_layers=1).cuda()
joiner_trans = nn.Linear(H, V, bias=False).cuda()
joiner_alpha = nn.Sequential(
    nn.Linear(H, 1, bias=True),
    nn.Tanh()
).cuda()

# inputs
src_embed = torch.rand(B, S, H).cuda().requires_grad_()
tgt_embed = torch.rand(B, T, H).cuda().requires_grad_()
targets = torch.randint(0, V, (B, T)).cuda()
adjust = lambda x, goal: x * goal // x.max()
source_lengths = adjust(torch.randint(1, S+1, (B,)).cuda(), S)
target_lengths = adjust(torch.randint(1, T+1, (B,)).cuda(), T)

# forward
src_feats, (h1, c1) = transcriber(src_embed.transpose(1, 0))
tgt_feats, (h2, c2) = predictor(tgt_embed.transpose(1, 0))

# memory efficient joint
mask = ~lengths_to_padding_mask(target_lengths)
lattice = F.relu(
    src_feats.transpose(0, 1).unsqueeze(1) + tgt_feats.transpose(0, 1).unsqueeze(2)
)[mask]
log_alpha = F.logsigmoid(joiner_alpha(lattice)).squeeze(-1)
lattice = joiner_trans(lattice).log_softmax(-1)

# normal ssnt loss
loss = ssnt_loss_mem(
    lattice,
    targets[mask],
    log_alpha,
    source_lengths=source_lengths,
    target_lengths=target_lengths,
    reduction="sum"
) / (B*T)
loss.backward()
print(loss.item())
```

## Note
This implementation is based on the simplifying derivation proposed for monotonic attention, where they use parallelized `cumsum` and `cumprod` to compute the alignment. Based on the similarity of SSNT and monotonic attention, we can infer that the forward variable alpha(i,j) can be computed similarly.

Feel free to contact me if there are bugs in the code.

## Reference
- [Online Segment to Segment Neural Transduction](https://arxiv.org/abs/1609.08194)
- [Online and Linear-Time Attention by Enforcing Monotonic Alignments](https://arxiv.org/abs/1704.00784)
- [Fairseq's implementation of monotonic attention](https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/examples/simultaneous_translation/utils/monotonic_attention.py#L12)
- [Derivation](https://hackmd.io/@EZc8fZcyQAO21PgxpD1CUQ/rJTsWDWBF)