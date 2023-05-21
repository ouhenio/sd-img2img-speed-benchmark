import torch

from diffusers import StableDiffusionImg2ImgPipeline as pipeline
from torch.utils.benchmark import Timer
from PIL import Image

device = "cuda"
model = "runwayml/stable-diffusion-v1-5"
image_path = "image.jpg"
prompt = "a picture of the andes mountains in winter"

# Sample image
init_img = Image.open(image_path).convert("RGB")
init_img = init_img.resize((768, 512))

# Speed benchmarking wrapper
def benchmark_torch_function(f, *args, **kwargs):
    t0 = Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return round(t0.blocked_autorange(min_run_time=1).mean, 2)

# Basic img2img pipeline
pipe = pipeline.from_pretrained(model).to(device)
f = lambda : pipe(prompt=prompt, image=init_img)
time_basic_pipeline = benchmark_torch_function(f)

# Half-precision pipeline
pipe = pipeline.from_pretrained(model,torch_dtype=torch.float16).to(device)
f = lambda : pipe(prompt=prompt, image=init_img)
time_hp_pipeline = benchmark_torch_function(f)

# Compiled pipeline
pipe = pipeline.from_pretrained(model).to(device)
pipe.unet = torch.compile(pipe.unet)
pipe(prompt=prompt, image=init_img) # compile warmup
f = lambda : pipe(prompt=prompt, image=init_img)
time_compiled_pipeline = benchmark_torch_function(f)

# Compiled half-precision pipeline
pipe = pipeline.from_pretrained(model,torch_dtype=torch.float16).to(device)
pipe.unet = torch.compile(pipe.unet)
pipe(prompt=prompt, image=init_img) # compile warmup
f = lambda : pipe(prompt=prompt, image=init_img)
time_compiled_hp_pipeline = benchmark_torch_function(f)

print(f"Basic pipeline inference time: {time_basic_pipeline}s")
print(f"Half precision pipeline inference time: {time_hp_pipeline}s")
print(f"Compiled pipeline inference time: {time_compiled_pipeline}s")
print(f"Compiled hp pipeline inference time: {time_compiled_hp_pipeline}s")