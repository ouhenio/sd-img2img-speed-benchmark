import time

from diffusers import StableDiffusionImg2ImgPipeline as pipeline
from PIL import Image

# Time measurement decorator
def measure_inferece_speed(inference_method):
    def wrapper(pipe, prompt, image):
        start_time = time.time()
        inference_method(pipe, prompt, image, strength=0.75)
        end_time = time.time()
        inference_time = end_time - start_time

        print("Inference Time:", inference_time)

    return wrapper

# Inference wrapper
@measure_inferece_speed
def infer(pipe, prompt, image, strength):
    pipe(prompt=prompt, image=image, strength=strength)

# Sample image
init_img = Image.open("image.jpg").convert("RGB")
init_img = init_img.resize((768, 512))

# Basic img2img pipeline
pipe = pipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda")

# Sample prompt
prompt = "a picture of the andes mountains in winter"

infer(pipe=pipe, prompt=prompt, image=init_img)