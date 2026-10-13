import os
import time
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

prompt = os.environ["PROMPT_EN"].strip()
duration = int(os.environ["DURATION"])
steps = int(os.environ.get("STEPS", "8"))

if duration > 11:
    raise SystemExit("Stable Audio Open Small supports a maximum duration of 11 seconds.")
if not os.environ.get("HF_TOKEN"):
    raise SystemExit(
        "HF_TOKEN is missing. This Hugging Face model is gated: accept its model terms, "
        "create a Hugging Face token, and add it to GitHub repository Secrets as HF_TOKEN."
    )

# HF_TOKEN is read by huggingface_hub automatically.
device = "cpu"
torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
start = time.time()
print("Loading Stable Audio Open Small (first run downloads about 1.7 GB of weights)...")
model, config = get_pretrained_model("stabilityai/stable-audio-open-small")
sample_rate = config["sample_rate"]
sample_size = min(int(sample_rate * duration), int(config["sample_size"]))
model = model.to(device).eval()
print(f"Model loaded in {time.time() - start:.1f}s; generating {duration}s on CPU.")

conditioning = [{"prompt": prompt, "seconds_total": duration}]
with torch.inference_mode():
    audio = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=1.0,
        conditioning=conditioning,
        sample_size=sample_size,
        sampler_type="pingpong",
        device=device,
    )

audio = rearrange(audio, "b d n -> d (b n)")
peak = audio.abs().max().clamp_min(1e-8)
audio = (audio.float() / peak).clamp(-1, 1).cpu()
os.makedirs("output", exist_ok=True)
torchaudio.save("output/music.wav", audio, sample_rate)
print(f"Saved output/music.wav ({time.time() - start:.1f}s total).")
