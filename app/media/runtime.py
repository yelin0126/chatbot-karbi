from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import (
    MEDIA_IMAGE_GENERATION_MODEL,
    MEDIA_IMAGE_GUIDANCE_SCALE,
    MEDIA_IMAGE_HEIGHT,
    MEDIA_IMAGE_MAX_STEPS,
    MEDIA_IMAGE_UNDERSTANDING_MODEL,
    MEDIA_IMAGE_WIDTH,
    MEDIA_VIDEO_FPS,
    MEDIA_VIDEO_GENERATION_MODEL,
    MEDIA_VIDEO_HEIGHT,
    MEDIA_VIDEO_NUM_FRAMES,
    MEDIA_VIDEO_WIDTH,
)

logger = logging.getLogger("tilon.media.runtime")


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_dependency(name: str) -> None:
    if not _dependency_available(name):
        raise RuntimeError(f"Required media dependency is missing: {name}")


class MediaRuntime:
    def __init__(self) -> None:
        self._image_understanding = None
        self._image_generation = None
        self._video_generation = None

    def _torch(self):
        _require_dependency("torch")
        import torch

        return torch

    def _device_and_dtype(self):
        torch = self._torch()
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        return "cpu", torch.float32

    def _load_image_understanding(self):
        if self._image_understanding is not None:
            return self._image_understanding

        _require_dependency("transformers")
        from transformers import AutoProcessor
        from transformers import Qwen2_5_VLForConditionalGeneration

        device, dtype = self._device_and_dtype()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MEDIA_IMAGE_UNDERSTANDING_MODEL,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if device == "cuda":
            model = model.to(device)
        processor = AutoProcessor.from_pretrained(
            MEDIA_IMAGE_UNDERSTANDING_MODEL,
            trust_remote_code=True,
        )
        self._image_understanding = {
            "model": model,
            "processor": processor,
            "device": device,
            "model_name": MEDIA_IMAGE_UNDERSTANDING_MODEL,
        }
        return self._image_understanding

    def _load_image_generation(self):
        if self._image_generation is not None:
            return self._image_generation

        _require_dependency("diffusers")
        from diffusers import FluxPipeline

        device, dtype = self._device_and_dtype()
        pipe = FluxPipeline.from_pretrained(
            MEDIA_IMAGE_GENERATION_MODEL,
            torch_dtype=dtype,
        )
        if device == "cuda":
            pipe = pipe.to(device)
        self._image_generation = {
            "pipeline": pipe,
            "device": device,
            "model_name": MEDIA_IMAGE_GENERATION_MODEL,
        }
        return self._image_generation

    def _load_video_generation(self):
        if self._video_generation is not None:
            return self._video_generation

        _require_dependency("diffusers")
        from diffusers import WanPipeline

        device, dtype = self._device_and_dtype()
        pipe = WanPipeline.from_pretrained(
            MEDIA_VIDEO_GENERATION_MODEL,
            torch_dtype=dtype,
        )
        if device == "cuda":
            pipe = pipe.to(device)
        self._video_generation = {
            "pipeline": pipe,
            "device": device,
            "model_name": MEDIA_VIDEO_GENERATION_MODEL,
        }
        return self._video_generation

    def analyze_image(self, *, image_path: Path, prompt: str) -> Dict[str, Any]:
        runtime = self._load_image_understanding()
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        processor = runtime["processor"]
        model = runtime["model"]
        device = runtime["device"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt or "이 이미지를 설명해줘."},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        if device == "cuda":
            inputs = {key: value.to(device) for key, value in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        prompt_length = inputs["input_ids"].shape[1]
        output_text = processor.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return {
            "answer": output_text,
            "model_name": runtime["model_name"],
        }

    def generate_image(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        runtime = self._load_image_generation()
        pipe = runtime["pipeline"]
        image = pipe(
            prompt=prompt,
            width=width or MEDIA_IMAGE_WIDTH,
            height=height or MEDIA_IMAGE_HEIGHT,
            num_inference_steps=num_inference_steps or MEDIA_IMAGE_MAX_STEPS,
            guidance_scale=MEDIA_IMAGE_GUIDANCE_SCALE if guidance_scale is None else guidance_scale,
        ).images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return {
            "model_name": runtime["model_name"],
            "width": image.width,
            "height": image.height,
        }

    def generate_video(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> Dict[str, Any]:
        runtime = self._load_video_generation()
        pipe = runtime["pipeline"]

        result = pipe(
            prompt=prompt,
            width=width or MEDIA_VIDEO_WIDTH,
            height=height or MEDIA_VIDEO_HEIGHT,
            num_frames=num_frames or MEDIA_VIDEO_NUM_FRAMES,
        )
        frames = getattr(result, "frames", None)
        if not frames:
            raise RuntimeError("Video generation returned no frames")

        frame_list = frames[0] if isinstance(frames, list) and frames and isinstance(frames[0], list) else frames
        try:
            import imageio.v2 as imageio
        except Exception as exc:
            raise RuntimeError("imageio is required to save generated videos") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_path, frame_list, fps=fps or MEDIA_VIDEO_FPS)
        return {
            "model_name": runtime["model_name"],
            "width": width or MEDIA_VIDEO_WIDTH,
            "height": height or MEDIA_VIDEO_HEIGHT,
            "num_frames": len(frame_list),
            "fps": fps or MEDIA_VIDEO_FPS,
        }
