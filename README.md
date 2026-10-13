# AI Music — CPU music generation experiments

This repository includes a manually triggered GitHub Actions workflow for testing two Stability AI audio models on CPU:

- **Stable Audio 3 Small-Music** — intended for longer music clips, up to 120 seconds.
- **Stable Audio Open Small** — lightweight model intended for short clips, up to 11 seconds.

## Run a generation

1. Open the [Actions tab](https://github.com/unisocsu/ai_music/actions/workflows/generate-stable-audio-cpu.yml).
2. Select **Generate music (Stable Audio CPU)** and click **Run workflow**.
3. Choose the model, enter a prompt in Hebrew or English, and choose duration and sampling steps.
4. After the run finishes, download the `generated-music-...` artifact.

## Hebrew prompts

Hebrew prompts are automatically translated to English before generation, because these models are primarily documented and conditioned for English prompts. Translation uses an online translation service during the workflow. English prompts are passed through unchanged.

**Important:** this supports describing the sound in Hebrew; it does not add reliable Hebrew singing or Hebrew-lyrics generation. These are text-to-audio music/audio models, not dedicated Hebrew singing models.

## Hugging Face access

Stable Audio Open Small is a gated Hugging Face model. Before running it:

1. Sign in to Hugging Face and accept the model's access terms: https://huggingface.co/stabilityai/stable-audio-open-small
2. Create a Hugging Face access token.
3. In this GitHub repository, open **Settings → Secrets and variables → Actions → New repository secret**.
4. Name the secret `HF_TOKEN` and paste the token as its value.

If model access is denied, confirm that the account attached to the token has accepted the model terms. The workflow passes `HF_TOKEN` to Hugging Face libraries. The first run downloads model weights; subsequent runs may restore the Hugging Face cache.

## CPU and speed notes

- The workflow deliberately runs on a standard CPU runner; it does not request a GPU.
- First-run dependency/model downloads can be large. Cache hits may reduce setup time on later runs, but GitHub may evict caches.
- CPU speed depends on the hosted runner and can be much slower than vendor benchmarks on optimized hardware.
- Stable Audio Open Small is limited to 11 seconds. Choose Stable Audio 3 Small-Music for longer clips.
- Reducing sampling steps may reduce generation time, at some quality cost.

## License

Review the model-specific terms before using generated output, especially for commercial use. See the official model cards:
- https://huggingface.co/stabilityai/stable-audio-3-small-music
- https://huggingface.co/stabilityai/stable-audio-open-small
