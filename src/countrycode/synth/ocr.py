"""OCR error generator `ocr@1` (plan-03 Task 6): real misreadings of a real OCR engine.

Names are rendered (handwriting-style OFL fonts or DejaVu Sans), degraded (skew, blur, noise,
low resolution, JPEG) and read back by `microsoft/trocr-base-handwritten` (MIT) on the GPU in
batches. The readings are cached (`data/interim/synth/ocr_readings.jsonl`), so the generator is
reproducible; it keeps only readings that differ from the source and still resemble it.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rapidfuzz import fuzz

from countrycode.paths import CONFIGS
from countrycode.synth.framework import (
    MIN_NOISE_KEY,
    PRIORITY_LANGS,
    BaseAlias,
    BaseInfo,
    _alias_seed,
    is_latin,
    recoverable,
    seedable,
)
from countrycode.text import compact_key

CONFIG_PATH = CONFIGS / "synth" / "ocr.yaml"
CACHE_PATH = Path("data/interim/synth/ocr_readings.jsonl")


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def render(text: str, font_path: str, rng: np.random.Generator) -> Image.Image:
    size = int(rng.integers(30, 52))
    font = ImageFont.truetype(font_path, size)
    left, top, right, bottom = (int(v) for v in font.getbbox(text))
    pad = int(rng.integers(6, 18))
    w, h = right - left + 2 * pad, bottom - top + 2 * pad
    ink = int(rng.integers(0, 70))
    paper = int(rng.integers(215, 256))
    img = Image.new("L", (max(w, 8), max(h, 8)), paper)
    ImageDraw.Draw(img).text((pad - left, pad - top), text, font=font, fill=ink)
    img = img.rotate(float(rng.normal(0, 1.5)), expand=True, fillcolor=paper)
    if rng.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.2, 1.3))))
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0, float(rng.uniform(0, 18)), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        f = float(rng.uniform(0.35, 0.8))
        small = img.resize((max(8, int(img.width * f)), max(8, int(img.height * f))))
        img = small.resize(img.size)
    if rng.random() < 0.6:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(rng.integers(15, 60)))
        img = Image.open(io.BytesIO(buf.getvalue()))
    return img.convert("RGB")


def is_garbage(reading: str, source: str, min_ratio: float = 45.0) -> bool:
    k = compact_key(reading)
    if not k or sum(c.isalpha() for c in reading) < 0.6 * max(len(reading.strip()), 1):
        return True
    return fuzz.ratio(k, compact_key(source)) < min_ratio


def ocr_bases(bases: Sequence[BaseAlias]) -> list[BaseAlias]:
    return [
        b
        for b in bases
        if b.kind in ("name", "historical", "demonym")
        and is_latin(b.alias)
        and b.lang.split("_")[0] in PRIORITY_LANGS
        and 3 <= len(b.alias) <= 40
    ]


def build_cache(
    bases: Sequence[BaseAlias], seed: int = 13, batch: int = 96, path: Path = CACHE_PATH
) -> int:
    """Render and read every OCR base (renders_per_alias times); append-only cache."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    cfg = load_config()
    done: dict[str, list[str]] = read_cache(path)
    todo = [b for b in ocr_bases(bases) if b.alias not in done]
    proc = TrOCRProcessor.from_pretrained(cfg["engine"])
    model = VisionEncoderDecoderModel.from_pretrained(cfg["engine"]).eval()
    model = model.to("cuda")  # type: ignore[arg-type]  # transformers stubs: .to(device) is fine
    hw = [str(Path(cfg["fonts_dir"]) / f) for f in cfg["handwriting_fonts"]]
    printed = list(cfg["printed_fonts"])
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as out:
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            imgs, owners = [], []
            for b in chunk:
                rng = np.random.default_rng(_alias_seed(seed, "ocr", b.alias))
                for _ in range(int(cfg["renders_per_alias"])):
                    text = b.alias.upper() if rng.random() < cfg["upper_share"] else b.alias
                    fonts = hw if rng.random() < cfg["handwriting_share"] else printed
                    imgs.append(render(text, str(rng.choice(fonts)), rng))
                    owners.append(b.alias)
            with torch.inference_mode():
                pix = proc(images=imgs, return_tensors="pt").pixel_values.to("cuda")  # type: ignore[call-arg]
                ids = model.generate(pix, max_new_tokens=32, num_beams=1)  # type: ignore[attr-defined]
            texts = proc.batch_decode(ids, skip_special_tokens=True)
            readings: dict[str, list[str]] = {}
            for alias, t in zip(owners, texts, strict=True):
                readings.setdefault(alias, []).append(t.strip())
            for alias, rs in readings.items():
                out.write(json.dumps({"alias": alias, "readings": rs}, ensure_ascii=False) + "\n")
            n += len(readings)
    return n


def read_cache(path: Path = CACHE_PATH) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["alias"]] = d["readings"]
    return out


class OcrVariants:
    name = "ocr"
    version = "1"

    MAX_EDIT_RATIO = 0.15  # v2: one misread letter per ~7

    def __init__(
        self, cache: dict[str, list[str]] | None = None, info: dict[str, BaseInfo] | None = None
    ) -> None:
        cfg = load_config() if CONFIG_PATH.exists() else {"min_ratio": 45}
        self.min_ratio = float(cfg.get("min_ratio", 45))
        self.cache = cache if cache is not None else read_cache()
        self.config = {**cfg, "readings": len(self.cache)}
        self.info = info  # v2 (plan-08): typed safe names only, recoverable readings only
        if info is not None:
            self.version = "2"
            self.config |= {"bases": "typed_safe", "max_edit_ratio": self.MAX_EDIT_RATIO,
                            "min_key": MIN_NOISE_KEY}  # fmt: skip

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[str]:
        if self.info is not None and not seedable(row, self.info):
            return []
        out: list[str] = []
        for r in self.cache.get(row.alias, []):
            if compact_key(r) == compact_key(row.alias) or is_garbage(r, row.alias, self.min_ratio):
                continue  # identical up to case/punctuation (format@1 covers that) or garbage
            if self.info is not None and not recoverable(
                row.alias, r, self.MAX_EDIT_RATIO, self.info[row.alias].core
            ):
                continue
            if r not in out:
                out.append(r)
        return out
