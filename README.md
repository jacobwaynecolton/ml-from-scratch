# ml-from-scratch

PyTorch building blocks for transformer inference, written from raw tensor ops and checked against PyTorch's own implementations or brute-force references.

Each file is a self-contained exercise: the implementation comes first, then an `if __name__ == "__main__":` block with assert-based tests. Most began as Colab notebooks and were exported to `.py`.

## Running

```bash
pip install torch
python decoding/beam_search.py   # any file; silent exit = all asserts passed
```

## Contents

### Tensor ops — [`tensor-ops/`](tensor-ops)

- [`gather_exercises.py`](tensor-ops/gather_exercises.py): `torch.gather` drills: sort/unsort with gather + scatter, target log-probs (checked against `F.cross_entropy`), masked sequence mean log-prob, last real hidden state per sequence

### Decoding — [`decoding/`](decoding)

- [`top_k_sampling.py`](decoding/top_k_sampling.py): top-k logit filtering, three ways (threshold mask, sort + index, topk + scatter)
- [`top_p_sampling.py`](decoding/top_p_sampling.py): top-p (nucleus) sampling: sort, cumsum mask, renormalize, sample, map back to vocab ids with `gather`
- [`beam_search.py`](decoding/beam_search.py): batched beam search over a toy bigram log-prob table: add beam scores, flatten, `topk`, recover parent beam (`// V`) and token (`% V`); greedy decoding as a baseline, checked equal to beam width 1 for every start token
- [`beam_search_drills.py`](decoding/beam_search_drills.py): beam search building blocks as separate drills: one batched next-token lookup, top-k over flattened beam scores, extending winning beams with `cat`, a clamped `topk`, and reordering per-beam state (a KV cache stand-in) by parent index; checked against a brute-force scan

### Attention — [`attention/`](attention)

- [`attention_pool.py`](attention/attention_pool.py): masked attention pooling over a padded batch: score tokens against a learned query, mask padding to `-inf`, softmax, weighted sum to `(B, D)`; tested for exact zero weight on padding, invariance to padded contents, the single-real-token case, and finite grads, plus an all-padding-row experiment
- [`multi_head_causal_self_attention.py`](attention/multi_head_causal_self_attention.py): multi-head causal self-attention as an `nn.Module`: Q/K/V projections, split into heads with `view` + `transpose`, scaled dot products, `triu` causal mask, softmax, merge heads, output projection; tested for shape, causality (future tokens can't change past outputs), parameter count, and an exact match against `nn.MultiheadAttention` loaded with the same weights
