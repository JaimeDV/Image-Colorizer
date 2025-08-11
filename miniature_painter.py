import sys
import os
import torch
from PIL import Image,ImageOps,ImageEnhance
import torchvision.transforms as T
import numpy as np
import cv2
import gradio as gr
import webbrowser
import threading

'''
/// @copyright © 2023+ Jaime díaz viéitez
/// @author Author: Jaime Díaz Viéitez
/// pamecin@gmail.com
/// @date 29/07/2025
/// @brief using Color2Embed colorizes an image using another as reference, creates a gradio interface for easy of use 
'''

#pip install torch torchvision Pillow numpy<2 opencv-python matplotlib gradio
#at settings add:
'''
    "python.analysis.extraPaths": [
    "./Color2Embed_pytorch"
    ]
'''

script_dir = os.path.dirname(__file__)

# === model image to image transform ===
def lab_tensor_to_rgb(L_tensor, ab_tensor, luminosity=100,blueBias=255,RedBias=255):
    L =  L_tensor.squeeze().cpu().numpy()
    ab = ab_tensor.squeeze().cpu().numpy()  # shape (2, H, W)
    print("a_raw → min, max, mean:", ab[0].min(), ab[0].max(), ab[0].mean())
    print("b_raw → min, max, mean:", ab[1].min(), ab[1].max(), ab[1].mean())
    # 1) Scale L to [0,100]
    L_scaled = (L * luminosity).astype(np.float32)

    # 2) Scale a to [–128,127]
    a_scaled = (ab[0] * RedBias - 128.0).astype(np.float32)
 

    # 3) Invert b so that >0.5 → yellow, <0.5 → blue
    b_scaled = ((0.5 - ab[1]) * blueBias).astype(np.float32)

    # 4) Stack into Lab and convert
    lab = np.stack([L_scaled, a_scaled, b_scaled], axis=-1)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

    # 5) Clip & convert to uint8
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb_uint8

# === Image loading ===
def load_image(image, grayscale=False,invert=False):
    image = image.convert("L" if grayscale else "RGB")
    original_size = image.size  # (width, height)

    if invert: image=ImageOps.invert(image)

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, original_size

def satu_img(image,saturation):
    enhancer = ImageEnhance.Color(image)
    # 0.0 = grayscale, 1.0 = original, >1.0 = more saturated
    return enhancer.enhance(saturation)

def colorize_image(BW_IMAGE,REFERENCE_IMAGE,luminosity,saturation,blueBias,RedBias):

    invert=False
    #to code use mode 

    # === dir loading ===
    script_dir = os.path.dirname(__file__)
    repo_path = os.path.join(script_dir, 'Color2Embed_pytorch')
    sys.path.append(repo_path)

    from color2embed import models, config, utils # type: ignore

    # === saturate ref image ===
    REFERENCE_IMAGE=satu_img(REFERENCE_IMAGE,saturation)


    # === File paths ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, "Color2Embed_pytorch/trained_model/Color2Embed_weights.pth")

    # === Setup device ===
    device = torch.device("cpu")

    # === Load model ===
    print("Loading model...")
    model = models.Color2Embed(config.COLOR_EMBEDDING_DIM)  # Uses resnet18(num_classes=512)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("Loading images...")
    bw_tensor, original_size = load_image(BW_IMAGE, True)
    ref_tensor, _ = load_image(REFERENCE_IMAGE,False,invert)
    bw_tensor = bw_tensor.to(device)
    ref_tensor = ref_tensor.to(device)

    # === Run model ===
    print("Colorizing...")
    with torch.no_grad():
        output = model(bw_tensor, ref_tensor)


    # === Convert LAB to RGB ===

    # === Save output ===
    output_rgb = lab_tensor_to_rgb(bw_tensor, output,luminosity,blueBias,RedBias)
    output_pil = Image.fromarray(output_rgb)

    # Resize to original dimensions
    output_resized = output_pil.resize(original_size, resample=Image.BICUBIC)
    
    return output_resized

# === opens gradio by default ===
def open_browser():
    webbrowser.open("http://localhost:7860")  


with gr.Blocks() as demo:
    gr.Markdown("# 🖌️ Image Colorizer")
    gr.Markdown("Colorizes a base image " \
    "by referencing another image for reference. If the results aren’t quite right, " \
    "try tweaking the mods or using a different reference image. images without backgrounds,real life photos, and oversaturared images produce better results." \
    "Images of patterns,cartoons, greenish and multicolor pictures tend to perform worse.")
    with gr.Row():
        imgbw = gr.Image(type="pil", label="Upload base")
        imgref = gr.Image(type="pil", label="Upload ref")

    sliderL = gr.Slider(label="luminosity",minimum=0,maximum=200, value=100)
    sliderS = gr.Slider(label="Saturation",minimum=0,maximum=2, value=1)
    sliderB = gr.Slider(label="BlueBias",minimum=0,maximum=400, value=255)
    sliderR = gr.Slider(label="RedBias",minimum=0,maximum=400, value=255)
    out = gr.Image(label="Result")
    btn = gr.Button("Generate")

    btn.click(colorize_image, inputs=[imgbw, imgref,sliderL,sliderS,sliderB,sliderR], outputs=out)

    threading.Timer(1.5, open_browser).start()
    demo.launch(share=True)
