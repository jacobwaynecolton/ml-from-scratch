# Bug hunts

> **These files contain deliberately planted bugs.** They are debugging exercises, kept here exactly as given (unsolved), so the code is wrong on purpose. Don't treat it as reference code, and don't take it as a sample of how I write code. My own implementations are in the other folders.

The task: find and explain every bug without looking anything up. The bugs are not marked, so the puzzle still works.

- [`mlp_training_loop.py`](mlp_training_loop.py): an MLP classifier on synthetic data: `Dataset`, `DataLoader`, training loop, evaluation. Bugs are spread across the data pipeline, training loop and evaluation.
- [`masked_multi_head_attention.py`](masked_multi_head_attention.py): masked multi-head self-attention with a causal mask and a key-padding mask. Bugs are in the scaling, masking, softmax and head-merging steps.
