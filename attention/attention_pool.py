"""Pooling_Attention_Review.ipynb

Attention pooling with a padding mask: score tokens against a learned
query vector, mask padding to -inf, softmax over T, weighted sum to (B, D).
"""

import torch

torch.manual_seed(0)


# ---------------------------------------------------------------
# YOUR PART: implement this
# ---------------------------------------------------------------
def attention_pool(x, mask, w):
    """
    x:    (B, T, D) float   token vectors
    mask: (B, T)    bool    True = padding
    w:    (D,)      float   learned query vector
    returns:
        pooled:  (B, D)
        weights: (B, T)
    """

    # Scoring each token
    scores = x @ w

    # Using the mask to set padding scores to -inf
    scores = torch.where(mask,-torch.inf,scores)

    # Turning the scores into weights
    weights = torch.softmax(scores,-1)


    # multiplying each token vector by its weight and sum over T, shape (B,D)
    out = (x * weights[...,None]).sum(dim=1)

    return (out,weights)


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # Setup / data
    # ---------------------------------------------------------------
    B, T, D = 3, 6, 4
    x = torch.randn(B, T, D)
    w = torch.randn(D, requires_grad=True)

    F, Tr = False, True
    mask = torch.tensor([
        [F, F, F, F, Tr, Tr],      # row 0: 4 real tokens, 2 padding
        [F, F, F, F, F,  F ],      # row 1: no padding
        [F, Tr, Tr, Tr, Tr, Tr],   # row 2: only 1 real token
    ])


    # ---------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------
    pooled, weights = attention_pool(x, mask, w)

    # 1. Shapes
    assert pooled.shape == (B, D), pooled.shape
    assert weights.shape == (B, T), weights.shape

    # 2. Weights sum to 1 per row (tolerance: floating point)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B)), weights.sum(dim=-1)

    # 3. Weights are EXACTLY 0 on padding (no tolerance on purpose)
    assert (weights[mask] == 0).all(), weights[mask]

    # 4. Row 0 matches a manual softmax over only its real tokens
    real = ~mask[0]
    scores = x[0][real] @ w
    expected = torch.softmax(scores, dim=0) @ x[0][real]
    assert torch.allclose(pooled[0], expected), (pooled[0], expected)

    # 5. One real token -> its weight is 1, output is that token
    assert torch.allclose(weights[2, 0], torch.tensor(1.0))
    assert torch.allclose(pooled[2], x[2, 0])

    # 6. Padding contents must not matter: scramble padded positions
    x_scrambled = x.clone()
    x_scrambled[mask] = torch.randn(int(mask.sum()), D) * 100
    pooled_scrambled, _ = attention_pool(x_scrambled, mask, w)
    assert torch.allclose(pooled, pooled_scrambled)

    # 7. Gradients reach w and are finite
    w.grad = None
    p, _ = attention_pool(x, mask, w)
    p.sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all(), w.grad

    print("All tests passed.")


    # ---------------------------------------------------------------
    # Experiment: an all-padding row (answer the question from this)
    # ---------------------------------------------------------------
    mask_all = mask.clone()
    mask_all[1] = True             # make row 1 entirely padding

    w.grad = None
    p_all, wts_all = attention_pool(x, mask_all, w)
    print("weights, all-pad row:", wts_all[1])
    print("pooled,  all-pad row:", p_all[1])
    print("pooled,  other rows: ", p_all[[0, 2]])

    p_all.sum().backward()
    print("grad on w:", w.grad)
    # Questions: what do you see, why does softmax produce it,
    # what does it do to the WHOLE batch during training, and how would you guard against it?
