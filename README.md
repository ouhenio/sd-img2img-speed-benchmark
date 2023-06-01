# Stable Diffusion img2img speed benchmarks

## What is this?

The goal of this repo is to measure and compare different img2img acceleration methods.

## Table of results

To replicate the results, simply run `main.py`.

Note that these measurements where done with an RTX 3090.

|                           Method | speed (seconds) |
|---------------------------------:|----------------:|
|                          Vanilla |            8.72 |
|                       `float.16` |            3.28 |
|                  `torch.compile` |            8.58 |
|     `torch.compile` + `float.16` |            3.15 |

## Todos

- [ ] Use different `torch.compile` `backend` methods.
- [ ] Use Pytorch's 2 [scaled_dot_product_attention](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention).
- [ ] Measure the difference between a manualy compiled pipeline (via `torch.jit.trace`) versus `torch.compile`.
- [ ] Experiment with different batch sizes.
