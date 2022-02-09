import torch
from typing import Optional
from torch import Tensor


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def log_exclusive_cumprod(tensor, dim: int):
    """
    Implementing exclusive cumprod in log space (assume tensor is in log space)
    exclusive cumprod(x) = [1, x1, x1x2, x1x2x3, ..., prod_{i=1}^{n-1} x_i]
    """
    return exclusive_cumsum(tensor, dim)


def exclusive_cumsum(tensor, dim: int):
    tensor = tensor.roll(1, dims=dim)  # right shift 1
    tensor.select(dim, 0).fill_(0)
    tensor = tensor.cumsum(dim)
    return tensor


def prob_check(tensor, eps=1e-10, neg_inf=-1e8, logp=False):
    assert not torch.isnan(tensor).any(), (
        "Nan in a probability tensor."
    )
    # Add the eps here to prevent errors introduced by precision
    if logp:
        assert tensor.le(0).all() and tensor.ge(neg_inf).all(), (
            "Incorrect values in a log-probability tensor"
            ", -inf <= tensor <= 0"
        )
    else:
        assert tensor.le(1.0 + eps).all() and tensor.ge(0.0 - eps).all(), (
            "Incorrect values in a probability tensor"
            ", 0.0 <= tensor <= 1.0"
        )


def ssnt_loss(
    log_probs: Tensor,
    targets: Tensor,
    source_lengths: Tensor,
    target_lengths: Tensor,
    emit_logits: Optional[Tensor] = None,
    emit_probs: Optional[Tensor] = None,
    neg_inf: float = -1e4,
    reduction="none",
    return_lattice=False,
    fastemit_lambda=0
):
    """The SNNT loss is very similar to monotonic attention,
    except taking word prediction probability p(y_j | h_i, s_j)
    into account.

    N is the minibatch size
    T is the maximum number of output labels
    S is the maximum number of input frames
    V is the vocabulary of labels.

    Args:
        log_probs (Tensor): (N, T, S, V) Word prediction log-probs, should be output of log_softmax.
        targets (Tensor): (N, T) target labels for all samples in the minibatch.
        source_lengths (Tensor): (N,) Length of the source frames for each sample in the minibatch.
        target_lengths (Tensor): (N,) Length of the target labels for each sample in the minibatch.
        emit_logits, emit_probs (Tensor, optional): (N, T, S) Emission logits (before sigmoid) or
            probs (after sigmoid). If both are provided, logits is used.
        neg_inf (float, optional): The constant representing -inf used for masking.
            Default: -1e4
        reduction (string, optional): Specifies reduction. suppoerts mean / sum.
            Default: None.
        return_lattice (bool, optional): Returns the log alpha scores. e.g. for debug.
            Default: False.
        fastemit_lambda (float, optional): Scale the emission gradient of emission paths to
            encourage low latency. https://arxiv.org/pdf/2010.11148.pdf
            Default: 0
    """
    prob_check(log_probs, neg_inf=neg_inf, logp=True)

    if emit_logits is None:
        assert emit_probs is not None, "emit_probs and emit_logits cannot both be None."
        prob_check(emit_probs)
        log_p_choose = torch.log(emit_probs)
        log_1mp = torch.log1p(-emit_probs)
    else:
        log_p_choose = torch.nn.functional.logsigmoid(emit_logits)
        log_1mp = torch.nn.functional.logsigmoid(-emit_logits)

    # p_choose: bsz, tgt_len, src_len
    bsz, tgt_len, src_len = log_p_choose.size()
    # dtype = log_p_choose.dtype
    log_p_choose = log_p_choose.float()

    source_padding_mask = lengths_to_padding_mask(source_lengths)
    if source_padding_mask.any():
        log_p_choose = log_p_choose.masked_fill(source_padding_mask.unsqueeze(1), neg_inf)

    def clamp_logp(x, min=neg_inf, max=0):
        return x.clamp(min=min, max=max)

    # cumprod_1mp : bsz, tgt_len, src_len
    log_cumprod_1mp = log_exclusive_cumprod(log_1mp, dim=-1)

    log_alpha = log_p_choose.new_zeros([bsz, 1 + tgt_len, src_len])
    log_alpha[:, 0, 1:] = neg_inf

    fastemit_const = torch.tensor([fastemit_lambda]).log1p().type_as(log_alpha)
    for i in range(tgt_len):
        # log_probs:    bsz, tgt_len, src_len, vocab
        # p_choose:     bsz, tgt_len, src_len
        # cumprod_1mp:  bsz, tgt_len, src_len
        # alpha[i]:     bsz, src_len

        # get p(y_i | h_*, s_i) -> bsz, src_len
        # log_probs[:,i]:   bsz, src_len, vocab
        # targets[:,i]:     bsz,
        logp_trans = log_probs[:, i].gather(
            dim=-1,
            index=targets[:, i].view(bsz, 1, 1).expand(-1, src_len, -1)
        ).squeeze(-1)
        log_alpha_i = clamp_logp(
            logp_trans
            + log_p_choose[:, i]
            + log_cumprod_1mp[:, i]
            + torch.logcumsumexp(
                fastemit_const + log_alpha[:, i] - log_cumprod_1mp[:, i], dim=1
            )
        )
        log_alpha[:, i + 1] = log_alpha_i

    if return_lattice:
        return log_alpha[:, 1:]

    # alpha: bsz, 1 + tgt_len, src_len
    # seq-loss: alpha(J, I)
    # pick source endpoints
    log_alpha = log_alpha.gather(
        dim=2,
        index=(source_lengths - 1).view(bsz, 1, 1).expand(-1, 1 + tgt_len, -1)
    )
    # pick target endpoints
    log_alpha = log_alpha.gather(
        dim=1,
        index=target_lengths.view(bsz, 1, 1)
    ).view(bsz)

    prob_check(log_alpha, neg_inf=neg_inf, logp=True)

    if reduction == "sum":
        log_alpha = log_alpha.sum()
    elif reduction == "mean":
        log_alpha = log_alpha.mean()

    return -log_alpha


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
    prob_check(log_probs, neg_inf=neg_inf, logp=True)
    if emit_logits is None:
        assert emit_probs is not None, "emit_probs and emit_logits cannot both be None."
        prob_check(emit_probs)
        log_p_choose = torch.log(emit_probs)
        log_1mp = torch.log1p(-emit_probs)
    else:
        log_p_choose = torch.nn.functional.logsigmoid(emit_logits)
        log_1mp = torch.nn.functional.logsigmoid(-emit_logits)

    bsz = source_lengths.size(0)
    tgt_len_flat, src_len = log_p_choose.size()
    log_p_choose = log_p_choose.float()

    source_lengths_rep = torch.repeat_interleave(source_lengths, target_lengths, dim=0)
    source_padding_mask = lengths_to_padding_mask(source_lengths_rep)
    if source_padding_mask.any():
        assert source_padding_mask.size() == log_p_choose.size()
        log_p_choose = log_p_choose.masked_fill(source_padding_mask, neg_inf)

    def clamp_logp(x, min=neg_inf, max=0):
        return x.clamp(min=min, max=max)

    # cumprod_1mp : tgt_len_flat, src_len
    log_cumprod_1mp = log_exclusive_cumprod(log_1mp, dim=-1)

    log_alpha = log_p_choose.new_zeros([tgt_len_flat + bsz, src_len])
    offsets = exclusive_cumsum(target_lengths, dim=0)
    offsets_out = exclusive_cumsum(target_lengths + 1, dim=0)
    log_alpha[offsets_out, 1:] = neg_inf

    fastemit_const = torch.tensor([fastemit_lambda]).log1p().type_as(log_alpha)
    for i in range(target_lengths.max()):
        # log_probs:    tgt_len_flat, src_len, vocab
        # p_choose:     tgt_len_flat, src_len
        # cumprod_1mp:  tgt_len_flat, src_len

        # operate on fake bsz (aka indices.size(0) below)
        # get p(y_i | h_*, s_i) -> bsz, src_len
        # log_probs[indices]:   bsz, src_len, vocab
        # targets[indices]:     bsz,

        indices = (offsets + i)[i < target_lengths]
        indices_out = (offsets_out + i)[i < target_lengths]
        fake_bsz = indices.numel()

        logp_trans = (
            log_probs[indices]
            .gather(-1, index=targets[indices].view(fake_bsz, 1, 1).expand(-1, src_len, -1))
        ).squeeze(-1)
        log_alpha_i = clamp_logp(
            logp_trans
            + log_p_choose[indices]
            + log_cumprod_1mp[indices]
            + torch.logcumsumexp(
                fastemit_const + log_alpha[indices_out] - log_cumprod_1mp[indices], dim=1
            )
        )
        log_alpha[indices_out + 1] = log_alpha_i

    # alpha: tgt_len_flat + bsz, src_len
    # seq-loss: alpha(J, I)
    # pick target endpoints (bsz, src_len)
    log_alpha = log_alpha[offsets_out + target_lengths]
    # pick source endpoints
    log_alpha = log_alpha.gather(
        dim=-1,
        index=(source_lengths - 1).view(bsz, 1)
    ).view(bsz)

    prob_check(log_alpha, neg_inf=neg_inf, logp=True)

    if reduction == "sum":
        log_alpha = log_alpha.sum()
    elif reduction == "mean":
        log_alpha = log_alpha.mean()

    return -log_alpha
