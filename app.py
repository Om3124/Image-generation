import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk, Image
from authtoken import auth_token  # Ensure this exists

import torch
from torch.cuda.amp import autocast
from diffusers import StableDiffusionPipeline

# Create the main app window
app = ctk.CTk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Fix: Pass `app` as the master argument
prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512, text="")
lmain.place(x=10, y=110)

# Model setup
modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Since no GPU is available, use CPU

try:
    # Attempt to load the model from HuggingFace
    pipe = StableDiffusionPipeline.from_pretrained(
        modelid, revision="fp16", torch_dtype=torch.float16, token=auth_token
    )
    pipe.to(device)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check if the model is available, and ensure your token and network are correct.")
    exit(1)  # Exit the app if model loading fails

# Function to generate an image
def generate():
    text_prompt = prompt.get()
    try:
        image = pipe(text_prompt, guidance_scale=8.5).images[0]

        image.save("generatedimage.png")
        img = Image.open("generatedimage.png")
        img = ImageTk.PhotoImage(img)

        lmain.configure(image=img)
        lmain.image = img  # Prevent garbage collection
    except Exception as e:
        print(f"Error generating image: {e}")

# Fix: Pass `app` as the master argument
trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

# Run the app
app.mainloop()
