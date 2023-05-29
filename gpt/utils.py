import torch


def compute_mask_indices(
    input_feature: torch.Tensor,  # (B, T, C)
    mask_prob: float,
    mask_length: int,
    min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length, _ = input_feature.shape
    # input_feature = input_feature.view(batch_size, sequence_length)

    # compute how many tokens we want to mask
    num_masked_tokens = int(mask_prob * sequence_length)
    num_masked_tokens = max(num_masked_tokens, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_tokens > sequence_length:
        num_masked_tokens = sequence_length

    # SpecAugment mask to fill
    spec_aug_mask = torch.zeros(
        (batch_size, sequence_length), dtype=torch.bool, device=input_feature.device
    )

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=input_feature.device
    )

    # get random indices to mask
    spec_aug_mask_idxs = torch.multinomial(uniform_dist, num_masked_tokens)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = spec_aug_mask_idxs.unsqueeze(-1).expand(
        (batch_size, num_masked_tokens, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.contiguous().view(
        batch_size, num_masked_tokens * mask_length
    )

    offsets = (
        torch.arange(mask_length, device=input_feature.device)
        .unsqueeze(0)
        .expand((num_masked_tokens, mask_length))
    )
    offsets = offsets.contiguous().view(-1)

    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    spec_aug_mask = spec_aug_mask.scatter(1, spec_aug_mask_idxs, True)

    return spec_aug_mask


if __name__ == "__main__":
    input = torch.rand(2, 49, 768)
    print("the input matrix ", input)
    mask = compute_mask_indices(input, 0.02, 10, 10)

    input_masked = input.masked_fill(mask.unsqueeze(-1), 0)
    # input_masked = input.masked_fill(mask, 0)
    print(input_masked)