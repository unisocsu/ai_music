import os
import re
from pathlib import Path

prompt = os.environ.get("ORIGINAL_PROMPT", "").strip()
if not prompt:
    raise SystemExit("The prompt is empty.")

# Avoid sending English prompts through an external translation service.
has_hebrew = bool(re.search(r"[\u0590-\u05FF]", prompt))
translated = prompt
if has_hebrew:
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="en").translate(prompt).strip()
        if not translated:
            raise ValueError("Translation returned an empty string")
        print("Hebrew prompt translated to English for the model.")
    except Exception as exc:
        # Preserve the original prompt as a fallback, but make the limitation clear.
        print(f"::warning::Automatic Hebrew-to-English translation failed ({exc}). Passing the original prompt to the model; results may be weaker.")
        translated = prompt
else:
    print("Prompt is already English; translation skipped.")

# Write to GITHUB_OUTPUT using a delimiter so punctuation and spaces are preserved.
output_path = os.environ.get("GITHUB_OUTPUT")
if output_path:
    delimiter = "PROMPT_EOF_9f4a2c"
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"prompt_en<<{delimiter}\n{translated}\n{delimiter}\n")
else:
    print(translated)
