import base64
import io
from typing import Any

import runpod
import soundfile as sf
import torch
from omnivoice import OmniVoice


if not torch.cuda.is_available():
    raise RuntimeError("OmniVoice Serverless handler requires a CUDA GPU.")

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16,
)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input")
    if not isinstance(payload, dict):
        raise ValueError("Job input must be a JSON object.")

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Job input must include non-empty 'text'.")

    generate_options: dict[str, str] = {}
    for key in ("language", "instruct"):
        value = payload.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(f"'{key}' must be a string.")
            generate_options[key] = value

    audio = model.generate(text=text, **generate_options)

    wav_buffer = io.BytesIO()
    sf.write(
        wav_buffer,
        audio[0],
        model.sampling_rate,
        format="WAV",
        subtype="PCM_16",
    )

    return {
        "audio_base64": base64.b64encode(wav_buffer.getvalue()).decode("ascii"),
        "sample_rate": model.sampling_rate,
        "content_type": "audio/wav",
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
