# ml-from-scratch

From-scratch reimplementations for CSC2516 (Neural Networks and Deep Learning), MScAC @ UofT.

| # | Task | Pass criterion | Done |
|---|------|----------------|------|
| 1 | Autograd engine | Gradients match torch.autograd to 1e-6 on a 3-layer net | [ ] |
| 2 | Linear + logistic regression, hand-written SGD/momentum/Adam | My Adam matches torch.optim.Adam step-for-step to 1e-6 | [ ] |
| 3 | MLP on MNIST (hand-written SGD) | >97% test accuracy, no nn.Module, no optimizer | [ ] |
| 4 | LayerNorm + BatchNorm (fwd + bwd) | Outputs and grads match torch to 1e-5 | [ ] |
| 5 | CNN + residual block | Forward and backward match torch to 1e-5 | [ ] |
| 6 | RNN/LSTM char-level model | Generates plausible text; grads match torch to 1e-5 | [ ] |
| 7 | BPE tokenizer | encode/decode matches tiktoken GPT-2 byte-for-byte | [ ] |
| 8 | GPT-2 (load real 124M weights) | Logits match HuggingFace to 1e-4 | [ ] |
| 9 | KV cache | Identical output to uncached, >4x faster | [ ] |
| 10 | LoRA | <1% params trainable, beats frozen baseline | [ ] |

Tasks 1-6 and 8 follow the CSC2516 syllabus - each gets built the week its topic is lectured. Task 1 follows the micrograd approach. Tasks 7, 9, and 10 aren't covered in the course, doing them anyway.

## Extras

Practice outside the task list.

- [extras/sampling/top_k_sampling.py](extras/sampling/top_k_sampling.py): top-k logit filtering, three ways (threshold mask, sort + index, topk + scatter)
- [extras/sampling/top_p_sampling.py](extras/sampling/top_p_sampling.py): top-p (nucleus) sampling: sort, cumsum mask, renormalize, sample, map back to vocab ids with `gather`
- [extras/gather/gather_exercises.py](extras/gather/gather_exercises.py): `torch.gather` drills: sort/unsort with gather + scatter, target log-probs (checked against `F.cross_entropy`), masked sequence mean log-prob, last real hidden state per sequence
