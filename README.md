# ssnt-loss

A pure PyTorch implementation of the loss described in *"Online Segment to Segment Neural Transduction"*  https://arxiv.org/abs/1609.08194.

## Usage
There are two versions, a normal version and a memory efficient version. They should give the same output, please inform me if they don't.
```python
def ssnt_loss_mem(
    log_probs: Tensor,
    targets: Tensor,
    source_lengths: Tensor,
    target_lengths: Tensor,
    emit_logits: Optional[Tensor] = None,
    emit_probs: Optional[Tensor] = None,
    neg_inf: float = -1e4,
    reduction="none",
    fastemit_lambda=0
):
    """The memory efficient implementation concatenates along the targets
    dimension to reduce wasted computation on padding positions.

    N is the minibatch size
    T is the maximum number of output labels
    S is the maximum number of input frames
    V is the vocabulary of labels.
    T_flat is the summation of lengths of all output labels

    Assuming the original tensor is of (N, T, ...), then it should be reduced to
    (T_flat, ...). This can be obtained by using a target mask.
    For example:
        >>> target_mask = targets.ne(pad)   # (B, T)
        >>> targets = targets[target_mask]  # (T_flat,)
        >>> log_probs = log_probs[target_mask]  # (T_flat, S, V)

    Args:
        log_probs (Tensor): (T_flat, S, V) Word prediction log-probs, should be output of log_softmax.
        targets (Tensor): (T_flat,) target labels for all samples in the minibatch.
        source_lengths (Tensor): (N,) Length of the source frames for each sample in the minibatch.
        target_lengths (Tensor): (N,) Length of the target labels for each sample in the minibatch.
        emit_logits, emit_probs (Tensor, optional): (T_flat, S) Emission logits (before sigmoid) or
            probs (after sigmoid). If both are provided, logits is used.
        neg_inf (float, optional): The constant representing -inf used for masking.
            Default: -1e4
        reduction (string, optional): Specifies reduction. suppoerts mean / sum.
            Default: None.
        fastemit_lambda (float, optional): Scale the emission gradient of emission paths to
            encourage low latency. https://arxiv.org/pdf/2010.11148.pdf
            Default: 0
    """
```
### Minimal example
```bash
python example.py
```

## Note
> :information_source: This is a WIP project. the implementation is still being tested.
- This implementation is based on the parallelized `cumsum` and `cumprod` operations proposed in monotonic attention. Since the alignments in SSNT and monotonic attention is almost identical, we can infer that the forward variable alpha(i,j) of the SSNT can be computed similarly.
- Run test by `python test.py` (requires `pip install expecttest`).
- Feel free to contact me if there are bugs in the code.

## Reference
- [Online Segment to Segment Neural Transduction](https://arxiv.org/abs/1609.08194)
- [Online and Linear-Time Attention by Enforcing Monotonic Alignments](https://arxiv.org/abs/1704.00784)
- [Fairseq's implementation of monotonic attention](https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/examples/simultaneous_translation/utils/monotonic_attention.py#L12)
- [Derivation](https://hackmd.io/@EZc8fZcyQAO21PgxpD1CUQ/rJTsWDWBF)