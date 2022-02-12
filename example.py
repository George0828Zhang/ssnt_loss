import torch
import torch.nn as nn
import torch.nn.functional as F
from ssnt_loss import ssnt_loss_mem, lengths_to_padding_mask

if __name__ == "__main__":
    B, S, H, T, V = 2, 100, 256, 10, 2000

    # model
    transcriber = nn.LSTM(input_size=H, hidden_size=H, num_layers=1).cuda()
    predictor = nn.LSTM(input_size=H, hidden_size=H, num_layers=1).cuda()
    joiner_word = nn.Linear(H, V).cuda()
    joiner_emit = nn.Linear(H, 1).cuda()

    # inputs
    src_embed = torch.rand(B, S, H).cuda().requires_grad_()
    tgt_embed = torch.rand(B, T, H).cuda().requires_grad_()
    targets = torch.randint(0, V, (B, T)).cuda()

    def adjust(x, goal):
        return x * goal // x.max()
    source_lengths = adjust(torch.randint(1, S + 1, (B,)).cuda(), S)
    target_lengths = adjust(torch.randint(1, T + 1, (B,)).cuda(), T)

    # forward
    src_feats, (h1, c1) = transcriber(src_embed.transpose(1, 0))
    tgt_feats, (h2, c2) = predictor(tgt_embed.transpose(1, 0))

    # memory efficient joint
    mask = ~lengths_to_padding_mask(target_lengths)
    lattice = F.relu(
        src_feats.transpose(0, 1).unsqueeze(1) + tgt_feats.transpose(0, 1).unsqueeze(2)
    )[mask]
    logp = joiner_word(lattice).log_softmax(-1)
    emit_logits = joiner_emit(lattice).squeeze(-1)
    targets = targets[mask]

    # normal ssnt loss
    loss, _, _ = ssnt_loss_mem(
        logp,
        targets,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
        emit_logits=emit_logits,
        reduction="sum"
    ) / (B * T)
    loss.backward()
    print(loss.item())
