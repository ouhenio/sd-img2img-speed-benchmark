# Stable Diffusion img2img speed benchmarks

The goal of this repo is to store methods and metrics related to the inference time of Stable Diffusion's img2img.

|                          Method | speed (seconds) |
|--------------------------------:|----------------:|
|                         Vanilla |            8.72 |
|                      `float.16` |            3.28 |
|                 `torch.compile` |            8.58 |
|     `torch.compile`+ `float.16` |            3.15 |

The measurements in this tables where made with a 3090.