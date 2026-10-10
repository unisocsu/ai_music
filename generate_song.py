import os
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

root = Path(os.environ.get("ACESTEP_ROOT", ".")).resolve()
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

prompt = os.environ.get("SONG_PROMPT", "").strip()
style = os.environ.get("STYLE", "").strip()
lyrics = os.environ.get("LYRICS", "").strip()
if not prompt and not lyrics:
    raise SystemExit("Provide a song prompt or lyrics.")

caption = prompt
if style:
    caption = f"{caption}. Style: {style}" if caption else style
caption = caption[:512]

print("Initializing ACE-Step on CPU. First run downloads model weights and may take a long time.", flush=True)
dit = AceStepHandler()
status = dit.initialize_service(
    project_root=str(root),
    config_path="acestep-v15-turbo",
    device="cpu",
)
print(status, flush=True)
if status is False or (isinstance(status, tuple) and not status[-1]):
    raise RuntimeError(f"ACE-Step model initialization failed: {status}")

lm = LLMHandler()
checkpoint_dir = str(root / "checkpoints")
lm_status, lm_ok = lm.initialize(
    checkpoint_dir=checkpoint_dir,
    lm_model_path="acestep-5Hz-lm-0.6B",
    backend="pt",
    device="cpu",
)
print(lm_status, flush=True)
if not lm_ok:
    raise RuntimeError(f"ACE-Step language model initialization failed: {lm_status}")

params = GenerationParams(
    task_type="text2music",
    caption=caption,
    lyrics=lyrics,
    instrumental=not bool(lyrics),
    vocal_language="unknown",
    duration=15,
    inference_steps=8,
    thinking=False,
)
config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="mp3")
result = generate_music(dit, lm, params, config, save_dir=str(out_dir))
if not result.success or not result.audios:
    raise RuntimeError(f"Music generation failed: {result.error or result.status_message}")

valid_files = []
for audio in result.audios:
    raw_path = audio.get("path") if isinstance(audio, dict) else None
    if not raw_path:
        continue
    path = Path(raw_path)
    if path.is_file() and path.suffix.lower() == ".mp3" and path.stat().st_size >= 1024:
        valid_files.append(path.resolve())

if not valid_files:
    files = [str(p) for p in out_dir.rglob("*") if p.is_file()]
    raise RuntimeError(
        "ACE-Step reported success but no valid MP3 was saved. "
        f"Files found in output/: {files}"
    )

for path in valid_files:
    print(f"Generated audio: {path} ({path.stat().st_size:,} bytes)", flush=True)
