import base64
import io
import json
import os
import random
import time
import zipfile
import urllib.error
import urllib.request
import http.client
import socket
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


NODE_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.getcwd()
GEN_ENDPOINT = "https://image.novelai.net/ai/generate-image"
USER_DATA_ENDPOINT = "https://api.novelai.net/user/data"

# 500 can be persistent when payload is rejected internally, so it retries only max_retries times.
RETRY_FOREVER_STATUS = {408, 409, 425, 429, 502, 503, 504}
RETRY_LIMITED_STATUS = {500}
FATAL_AUTH_STATUS = {401, 403}
FATAL_ACCOUNT_STATUS = {402}

MODEL_CHOICES = [
    "nai-diffusion-4-5-full",
    "nai-diffusion-4-5-curated",
    "nai-diffusion-4-full",
    "nai-diffusion-4-curated-preview",
    "nai-diffusion-3",
    "nai-diffusion-furry-3",
    "nai-diffusion-2",
]

SAMPLER_CHOICES = [
    "k_euler_ancestral",
    "k_euler",
    "k_dpmpp_2m",
    "k_dpmpp_sde",
    "k_dpmpp_2m_sde",
    "k_dpmpp_2s_ancestral",
    "k_dpm_2",
    "k_dpm_fast",
    "ddim_v3",
    "ddim",
]

SCHEDULER_CHOICES = [
    "karras",
    "native",
    "exponential",
    "polyexponential",
]

NOISE_SCHEDULE_CHOICES = [
    "native",
    "karras",
    "exponential",
    "polyexponential",
]

SEED_MODE_CHOICES = [
    "fixed",
    "random_each_run",
    "increment_each_run",
]

CHARACTER_POSITION_CHOICES = [
    "left",
    "center",
    "right",
    "top_left",
    "top_center",
    "top_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
    "custom",
]

CHARACTER_POSITION_MAP = {
    "left": (0.25, 0.50),
    "center": (0.50, 0.50),
    "right": (0.75, 0.50),
    "top_left": (0.25, 0.25),
    "top_center": (0.50, 0.25),
    "top_right": (0.75, 0.25),
    "bottom_left": (0.25, 0.75),
    "bottom_center": (0.50, 0.75),
    "bottom_right": (0.75, 0.75),
}

CHARACTER_GRID_COLS = ["A", "B", "C", "D", "E"]
CHARACTER_GRID_ROWS = ["1", "2", "3", "4", "5"]
CHARACTER_GRID_COL_MAP = {
    "A": 0.10,
    "B": 0.30,
    "C": 0.50,
    "D": 0.70,
    "E": 0.90,
}
CHARACTER_GRID_ROW_MAP = {
    "1": 0.10,
    "2": 0.30,
    "3": 0.50,
    "4": 0.70,
    "5": 0.90,
}

UC_PRESET_CHOICES = ["0", "1", "2", "3"]

UC_PRESET_TEXT = {
    "0": "blurry, lowres, error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, logo, dated, signature, multiple views, gigantic breasts",
    "1": "blurry, lowres, error, worst quality, bad quality, jpeg artifacts, very displeasing, logo, dated, signature",
    "2": "",
    "3": "",
}

DEFAULT_RETRY_VALUES = {
    "timeout": 180,
    "retry_delay": 10,
    "max_retries": 5,
    "retry_forever": True,
}

DEFAULT_PARAM_VALUES = {
    "width": 1024,
    "height": 1024,
    "model": "nai-diffusion-4-5-full",
    "seed": 0,
    "seed_mode": "random_each_run",
    "sampler": "k_euler_ancestral",
    "scheduler": "karras",
    "noise_schedule": "native",
    "steps": 28,
    "cfg_scale": 6.5,
    "cfg_rescale": 0.05,
    "uc_preset": "0",
    "quality_toggle": True,
    "prefer_brownian": True,
    "sm": False,
    "sm_dyn": False,
    "batch_size": 1,
    "legacy": False,
    "check_anlas": False,
}




ANLAS_LAST_BALANCE: Optional[int] = None
ANLAS_TOTAL_COST: int = 0

def update_anlas_tracker(current: Optional[int], *, previous_hint: Optional[int] = None, source: str = "", note: str = "") -> Tuple[int, int, str]:
    """Update global Anlas cost tracker. Returns (last_cost, total_cost, status_text)."""
    global ANLAS_LAST_BALANCE, ANLAS_TOTAL_COST
    if current is None:
        last = 0
        total = int(ANLAS_TOTAL_COST or 0)
        shown = ANLAS_LAST_BALANCE if ANLAS_LAST_BALANCE is not None else "?"
        status = f"Anlas: {shown} | Last Cost: {last} | Total Cost: {total}"
        if note:
            status += f" | {note}"
        if source:
            status += f" | Token Source: {source}"
        return last, total, status

    current_i = int(current)
    last = 0
    baseline = None
    if previous_hint is not None:
        baseline = int(previous_hint)
    elif ANLAS_LAST_BALANCE is not None:
        baseline = int(ANLAS_LAST_BALANCE)

    if baseline is not None and current_i < baseline:
        last = baseline - current_i
        ANLAS_TOTAL_COST += last

    ANLAS_LAST_BALANCE = current_i
    total = int(ANLAS_TOTAL_COST or 0)
    status = f"Anlas: {current_i} | Last Cost: {last} | Total Cost: {total}"
    if note:
        status += f" | {note}"
    if source:
        status += f" | Token Source: {source}"
    return int(last), total, status

def get_anlas_tracker_total() -> int:
    return int(ANLAS_TOTAL_COST or 0)


class NovelAIError(RuntimeError):
    pass


class NovelAIAuthError(ValueError):
    pass


def _clean_token(value: Optional[str]) -> str:
    if value is None:
        return ""
    token = str(value).strip().strip('"').strip("'").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def _read_env_file(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.exists(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip().strip('"').strip("'")
    except Exception as exc:
        print(f"[NovelAI] Could not read env file {path}: {exc}")
    return data


def _masked_token(token: str) -> str:
    if len(token) <= 10:
        return token[:2] + "..."
    return token[:6] + "..." + token[-4:]


def get_token(api_token: str = "") -> Tuple[str, str]:
    """Token priority: token.txt -> Comfy root .env -> process env -> node field."""
    token_txt = os.path.join(NODE_DIR, "token.txt")
    if os.path.exists(token_txt):
        try:
            with open(token_txt, "r", encoding="utf-8") as f:
                token = _clean_token(f.read())
            if token:
                return token, "token.txt"
        except Exception as exc:
            print(f"[NovelAI] Could not read token.txt: {exc}")

    env_candidates = [
        os.path.join(COMFY_ROOT, ".env"),
        os.path.join(os.path.dirname(NODE_DIR), ".env"),
        os.path.join(NODE_DIR, ".env"),
    ]
    for env_path in env_candidates:
        env = _read_env_file(env_path)
        for key in ("NAI_ACCESS_TOKEN", "NAI_API_TOKEN", "NOVELAI_API_TOKEN", "NOVELAI_TOKEN"):
            token = _clean_token(env.get(key, ""))
            if token:
                return token, env_path

    for key in ("NAI_ACCESS_TOKEN", "NAI_API_TOKEN", "NOVELAI_API_TOKEN", "NOVELAI_TOKEN"):
        token = _clean_token(os.environ.get(key, ""))
        if token:
            return token, f"env:{key}"

    token = _clean_token(api_token)
    if token:
        return token, "api_token field"

    raise NovelAIAuthError(
        "NovelAI token not found. Create token.txt in comfyui_novelai, "
        "or set NAI_ACCESS_TOKEN in ComfyUI/.env, or fill api_token."
    )


def make_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/x-zip-compressed,application/zip,image/png,application/json,*/*",
        "Origin": "https://novelai.net",
        "Referer": "https://novelai.net/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ComfyUI-NovelAI/1.0",
    }


def sanitize_payload_for_log(payload: Dict[str, Any]) -> str:
    try:
        import copy
        dbg = copy.deepcopy(payload)
        p = dbg.get("parameters", {})
        for key in ("image", "mask"):
            if isinstance(p.get(key), str):
                p[key] = p[key][:80] + "...(base64 truncated)"
        for key in ("reference_image_multiple", "reference_information_extracted_multiple", "reference_strength_multiple"):
            if key in p and isinstance(p[key], list):
                p[key] = ["...(list truncated)..." if isinstance(x, str) and len(x) > 80 else x for x in p[key]]
        return json.dumps(dbg, ensure_ascii=False, indent=2)[:5000]
    except Exception as exc:
        return f"<payload log failed: {exc}>"


def http_request(
    url: str,
    *,
    method: str = "POST",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 180,
) -> Tuple[int, bytes, Dict[str, str]]:
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            try:
                data = resp.read()
            except http.client.IncompleteRead as e:
                partial = getattr(e, "partial", b"") or b""
                raise NovelAIError(f"IncompleteRead while receiving response ({len(partial)} bytes read)")
            except (TimeoutError, socket.timeout, ConnectionResetError) as e:
                raise NovelAIError(f"Connection read error: {e}")
            return int(resp.status), data, dict(resp.headers)
    except urllib.error.HTTPError as e:
        return int(e.code), e.read(), dict(e.headers)
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionResetError) as e:
        raise NovelAIError(f"Connection error: {e}")
    except http.client.IncompleteRead as e:
        partial = getattr(e, "partial", b"") or b""
        raise NovelAIError(f"IncompleteRead while receiving response ({len(partial)} bytes read)")


def request_with_retry(
    url: str,
    payload: Dict[str, Any],
    token: str,
    *,
    timeout: int,
    retry_delay: int,
    max_retries: int,
    retry_forever: bool,
) -> bytes:
    headers = make_headers(token)
    attempt = 0
    while True:
        attempt += 1
        try:
            status, data, _ = http_request(
                url,
                method="POST",
                headers=headers,
                payload=payload,
                timeout=max(10, int(timeout)),
            )
        except NovelAIError as exc:
            if retry_forever or attempt <= max_retries:
                print(f"[NovelAI] {exc}. Retry {attempt} in {retry_delay}s...")
                time.sleep(max(1, int(retry_delay)))
                continue
            raise

        if 200 <= status < 300:
            return data

        text = data.decode("utf-8", errors="replace")[:1500]

        if status in FATAL_AUTH_STATUS:
            raise NovelAIAuthError(
                f"NovelAI API returned {status} Unauthorized/Forbidden. Token was refused. Body: {text}"
            )
        if status in FATAL_ACCOUNT_STATUS:
            raise NovelAIError(f"NovelAI API returned {status}. Account/payment/anlas issue. Body: {text}")
        if status == 400:
            print("[NovelAI] Payload rejected. Sanitized payload:")
            print(sanitize_payload_for_log(payload))
            raise NovelAIError(f"NovelAI API returned 400 Bad Request. Body: {text}")

        should_retry_forever = retry_forever and status in RETRY_FOREVER_STATUS
        should_retry_limited = status in (RETRY_FOREVER_STATUS | RETRY_LIMITED_STATUS) and attempt <= max_retries
        if should_retry_forever or should_retry_limited:
            print(f"[NovelAI] API {status}. Retry {attempt} in {retry_delay}s. Body: {text[:300]}")
            time.sleep(max(1, int(retry_delay)))
            continue

        if status == 500:
            print("[NovelAI] API 500 after retries. Sanitized payload:")
            print(sanitize_payload_for_log(payload))
        raise NovelAIError(f"NovelAI API returned {status}. Body: {text}")


def get_anlas_balance(token: str, *, timeout: int = 30) -> Tuple[Optional[int], str]:
    headers = make_headers(token)
    status, data, _ = http_request(USER_DATA_ENDPOINT, method="GET", headers=headers, timeout=timeout)
    if not (200 <= status < 300):
        return None, f"anlas check failed: HTTP {status}: {data.decode('utf-8', errors='replace')[:300]}"
    try:
        obj = json.loads(data.decode("utf-8"))
        steps = obj.get("subscription", {}).get("trainingStepsLeft", {})
        if isinstance(steps, dict):
            fixed = int(steps.get("fixedTrainingStepsLeft", 0) or 0)
            purchased = int(steps.get("purchasedTrainingSteps", 0) or 0)
            return fixed + purchased, "ok"
        return None, f"unexpected trainingStepsLeft shape: {steps!r}"
    except Exception as exc:
        return None, f"anlas parse failed: {exc}"


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    arr = image.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def image_tensor_to_base64_png(image: torch.Tensor, width: int, height: int) -> str:
    pil = tensor_to_pil(image)
    if pil.size != (int(width), int(height)):
        pil = pil.resize((int(width), int(height)), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def pil_to_base64_for_api(img: Image.Image) -> str:
    """Encode like the web client: PNG for alpha images, JPEG for normal RGB images."""
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    buf = io.BytesIO()
    if img.mode == "RGBA":
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def resize_and_pad_precise_reference_image(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    target_sizes = [(1024, 1536), (1472, 1472), (1536, 1024)]
    ow, oh = img.size
    if ow <= 0 or oh <= 0:
        raise NovelAIError("Precise Reference image has invalid dimensions.")
    ratio = ow / oh
    target_w, target_h = min(target_sizes, key=lambda size: abs((size[0] / size[1]) - ratio))
    scale = min(target_w / ow, target_h / oh)
    nw = max(1, int(round(ow * scale)))
    nh = max(1, int(round(oh * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    padded.paste(resized, ((target_w - nw) // 2, (target_h - nh) // 2))
    return padded


def precise_reference_tensor_to_base64(image: torch.Tensor) -> str:
    return pil_to_base64_for_api(resize_and_pad_precise_reference_image(tensor_to_pil(image)))


def mask_tensor_to_base64_png(mask: torch.Tensor, width: int, height: int, invert: bool = False) -> str:
    pil = tensor_to_pil(mask).convert("L")
    if pil.size != (int(width), int(height)):
        pil = pil.resize((int(width), int(height)), Image.Resampling.BILINEAR)
    if invert:
        pil = ImageOps.invert(pil)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_images_from_response(data: bytes) -> List[Image.Image]:
    images: List[Image.Image] = []
    if not data:
        raise NovelAIError("Empty response from NovelAI API.")
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    with zf.open(name) as f:
                        images.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
        if images:
            return images
        raise NovelAIError("NovelAI returned a ZIP but no images were found inside.")
    if data.startswith(b"\x89PNG") or data.startswith(b"\xff\xd8") or data.startswith(b"RIFF"):
        return [Image.open(io.BytesIO(data)).convert("RGB")]
    try:
        obj = json.loads(data.decode("utf-8"))
        candidates = []
        if isinstance(obj, dict):
            for key in ("image", "images", "output", "outputs"):
                if key in obj:
                    candidates.append(obj[key])
        for item in candidates:
            if isinstance(item, str):
                raw = base64.b64decode(item.split(",")[-1])
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str):
                        raw = base64.b64decode(sub.split(",")[-1])
                        images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                    elif isinstance(sub, dict):
                        for k in ("image", "data", "base64"):
                            if isinstance(sub.get(k), str):
                                raw = base64.b64decode(sub[k].split(",")[-1])
                                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                                break
        if images:
            return images
        raise NovelAIError(f"No image data found in JSON response: {str(obj)[:500]}")
    except json.JSONDecodeError:
        raise NovelAIError(f"Unknown response format. First bytes: {data[:80]!r}")


def stack_images(images: List[Image.Image]) -> torch.Tensor:
    if not images:
        raise NovelAIError("No images to output.")
    size = images[0].size
    fixed = []
    for img in images:
        if img.size != size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        fixed.append(pil_to_tensor(img))
    return torch.cat(fixed, dim=0)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _append_csv(a: str, b: str) -> str:
    a = (a or "").strip().strip(",")
    b = (b or "").strip().strip(",")
    if a and b:
        return f"{a}, {b}"
    return a or b


def normalize_character_prompts(character_prompts_json: str = "", character_prompts: Any = None) -> List[Dict[str, Any]]:
    source = character_prompts
    if character_prompts_json and str(character_prompts_json).strip():
        try:
            source = json.loads(character_prompts_json)
        except Exception as exc:
            print(f"[NovelAI] character_prompts_json ignored: {exc}")
    if not source:
        return []
    if isinstance(source, dict):
        source = [source]
    result = []
    for cp in source:
        if cp is None:
            continue

        ai_choice = False
        grid = None
        center = None
        position_mode = "manual"

        if isinstance(cp, dict):
            prompt = cp.get("prompt", cp.get("char_caption", "")) or ""
            uc = cp.get("uc", cp.get("negative", "")) or ""
            position_mode = str(cp.get("position_mode", "") or "").lower()
            ai_choice = bool(cp.get("ai_choice", False)) or position_mode in {"ai_choice", "ai_choices"}
            grid = cp.get("grid")
            if isinstance(cp.get("centers"), list):
                centers = cp.get("centers") or []
                center = centers[0] if centers else None
            if not isinstance(center, dict):
                center = cp.get("center") if isinstance(cp.get("center"), dict) else None
            if not isinstance(center, dict):
                center = {"x": cp.get("x", 0.5), "y": cp.get("y", 0.5)}
        else:
            prompt = getattr(cp, "prompt", "") or ""
            uc = getattr(cp, "uc", "") or ""
            position_mode = str(getattr(cp, "position_mode", "") or "").lower()
            ai_choice = bool(getattr(cp, "ai_choice", False)) or position_mode in {"ai_choice", "ai_choices"}
            grid = getattr(cp, "grid", None)
            center = getattr(cp, "center", None)
            if not isinstance(center, dict):
                center = {"x": getattr(cp, "x", 0.5), "y": getattr(cp, "y", 0.5)}

        try:
            x = float(center.get("x", 0.5))
            y = float(center.get("y", 0.5))
        except Exception:
            x, y = 0.5, 0.5

        item = {
            "prompt": str(prompt),
            "uc": str(uc),
            "center": {"x": _clamp01(x), "y": _clamp01(y)},
        }
        if ai_choice:
            item["ai_choice"] = True
            item["position_mode"] = "ai_choices"
        elif position_mode:
            item["position_mode"] = position_mode
        if grid:
            item["grid"] = grid
        result.append(item)
    return result

def _clamp01(value: Any, default: float = 0.5) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    return max(0.0, min(1.0, v))


def _character_position(preset: str, x: Any, y: Any) -> Dict[str, float]:
    preset = str(preset or "center")
    if preset == "custom":
        return {"x": _clamp01(x), "y": _clamp01(y)}
    px, py = CHARACTER_POSITION_MAP.get(preset, (0.5, 0.5))
    return {"x": float(px), "y": float(py)}


def _character_grid_position(col: Any, row: Any) -> Dict[str, float]:
    c = str(col or "C").upper()
    r = str(row or "3")
    return {
        "x": float(CHARACTER_GRID_COL_MAP.get(c, 0.5)),
        "y": float(CHARACTER_GRID_ROW_MAP.get(r, 0.5)),
    }


def build_character_prompts_from_slots(extra: Dict[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for i in range(1, 6):
        enabled = bool(extra.get(f"character_{i}_enable", False))
        prompt = str(extra.get(f"character_{i}_prompt", "") or "").strip()
        uc = str(extra.get(f"character_{i}_negative", "") or "").strip()
        if not enabled or not prompt:
            continue
        preset = str(extra.get(f"character_{i}_position", "center") or "center")
        center = _character_position(
            preset,
            extra.get(f"character_{i}_x", 0.5),
            extra.get(f"character_{i}_y", 0.5),
        )
        result.append({"prompt": prompt, "uc": uc, "center": center})
    return result


def character_slot_inputs() -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for i in range(1, 6):
        fields[f"character_{i}_enable"] = ("BOOLEAN", {"default": i == 1})
        fields[f"character_{i}_prompt"] = ("STRING", {"multiline": True, "default": ""})
        fields[f"character_{i}_negative"] = ("STRING", {"multiline": True, "default": ""})
        fields[f"character_{i}_position"] = (CHARACTER_POSITION_CHOICES, {"default": "center" if i == 1 else ("left" if i == 2 else "right")})
        fields[f"character_{i}_x"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
        fields[f"character_{i}_y"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
    return fields


def parameter_node_inputs() -> Dict[str, Any]:
    return {
        "width": ("INT", {"default": DEFAULT_PARAM_VALUES["width"], "min": 64, "max": 2048, "step": 64}),
        "height": ("INT", {"default": DEFAULT_PARAM_VALUES["height"], "min": 64, "max": 2048, "step": 64}),
        "model": (MODEL_CHOICES, {"default": DEFAULT_PARAM_VALUES["model"]}),
        "seed": ("INT", {"default": DEFAULT_PARAM_VALUES["seed"], "min": 0, "max": 9999999999}),
        "seed_mode": (SEED_MODE_CHOICES, {"default": DEFAULT_PARAM_VALUES["seed_mode"]}),
        "sampler": (SAMPLER_CHOICES, {"default": DEFAULT_PARAM_VALUES["sampler"]}),
        "scheduler": (SCHEDULER_CHOICES, {"default": DEFAULT_PARAM_VALUES["scheduler"]}),
        "noise_schedule": (NOISE_SCHEDULE_CHOICES, {"default": DEFAULT_PARAM_VALUES["noise_schedule"]}),
        "steps": ("INT", {"default": DEFAULT_PARAM_VALUES["steps"], "min": 1, "max": 100}),
        "cfg_scale": ("FLOAT", {"default": DEFAULT_PARAM_VALUES["cfg_scale"], "min": 0.0, "max": 50.0, "step": 0.1}),
        "cfg_rescale": ("FLOAT", {"default": DEFAULT_PARAM_VALUES["cfg_rescale"], "min": 0.0, "max": 1.0, "step": 0.01}),
        "uc_preset": (UC_PRESET_CHOICES, {"default": DEFAULT_PARAM_VALUES["uc_preset"]}),
        "quality_toggle": ("BOOLEAN", {"default": DEFAULT_PARAM_VALUES["quality_toggle"]}),
        "prefer_brownian": ("BOOLEAN", {"default": DEFAULT_PARAM_VALUES["prefer_brownian"]}),
        "sm": ("BOOLEAN", {"default": DEFAULT_PARAM_VALUES["sm"]}),
        "sm_dyn": ("BOOLEAN", {"default": DEFAULT_PARAM_VALUES["sm_dyn"]}),
        "batch_size": ("INT", {"default": DEFAULT_PARAM_VALUES["batch_size"], "min": 1, "max": 8}),
        "legacy": ("BOOLEAN", {"default": DEFAULT_PARAM_VALUES["legacy"]}),
        "check_anlas": ("BOOLEAN", {"default": DEFAULT_PARAM_VALUES["check_anlas"]}),
    }


def merge_parameter_values(config: Any, *, width: Optional[int] = None, height: Optional[int] = None) -> Dict[str, Any]:
    merged = dict(DEFAULT_PARAM_VALUES)
    if isinstance(config, dict):
        merged.update({k: v for k, v in config.items() if k in merged})
    if width is not None:
        merged["width"] = int(width)
    if height is not None:
        merged["height"] = int(height)
    return merged


def character_preview_text(characters: List[Dict[str, Any]]) -> str:
    if not characters:
        return "Characters: 0"
    lines = [f"Characters: {len(characters)}"]
    for idx, cp in enumerate(characters, start=1):
        if cp.get("ai_choice"):
            pos = "AI's Choice"
        else:
            center = cp.get("center", {})
            pos = f"({center.get('x', 0.5):.2f},{center.get('y', 0.5):.2f})"
        lines.append(
            f"{idx}. {cp.get('prompt', '')[:60]} | neg: {cp.get('uc', '')[:40]} | pos={pos}"
        )
    return "\n".join(lines)


def normalize_references(references: Any = None) -> List[Dict[str, Any]]:
    if not references:
        return []
    source = references
    if isinstance(source, dict):
        source = [source]
    result: List[Dict[str, Any]] = []
    for ref in source:
        if not isinstance(ref, dict):
            continue
        image_b64 = str(ref.get("image") or ref.get("image_b64") or "").strip()
        if not image_b64:
            continue
        result.append({
            "image": image_b64,
            "information_extracted": float(ref.get("information_extracted", 0.5)),
            "strength": float(ref.get("strength", 0.6)),
            "mode": str(ref.get("mode", "precise_reference") or "precise_reference"),
        })
    return result


def reference_preview_text(references: List[Dict[str, Any]]) -> str:
    if not references:
        return "References: 0"
    lines = [f"References: {len(references)}"]
    for idx, ref in enumerate(references, start=1):
        lines.append(
            f"{idx}. mode={ref.get('mode', 'precise_reference')} | info={float(ref.get('information_extracted', 0.5)):.2f} | strength={float(ref.get('strength', 0.6)):.2f}"
        )
    return "\n".join(lines)


def _model_is_v45(model: Any) -> bool:
    return str(model or "").startswith("nai-diffusion-4-5")


def apply_references_to_params(params: Dict[str, Any], references: Any = None, model: str = "") -> int:
    refs = normalize_references(references)

    # Always clear both legacy Vibe fields and V4.5 precise-reference fields first.
    params["reference_image_multiple"] = []
    params["reference_information_extracted_multiple"] = []
    params["reference_strength_multiple"] = []
    params.pop("director_reference_images", None)
    params.pop("director_reference_descriptions", None)
    params.pop("director_reference_strength_values", None)
    params.pop("director_reference_secondary_strength_values", None)
    params.pop("director_reference_information_extracted", None)

    if not refs:
        return 0

    modes = {str(r.get("mode", "precise_reference") or "precise_reference") for r in refs}
    unsupported = modes - {"precise_reference"}
    if unsupported:
        raise NovelAIError(
            "Unsupported reference mode removed from this node pack: "
            + ", ".join(sorted(unsupported))
            + ". Use NovelAI Precise Reference with V4.5 models only."
        )

    if not _model_is_v45(model):
        raise NovelAIError(
            "Precise Reference is V4.5-only in this node pack. "
            "Use nai-diffusion-4-5-full / nai-diffusion-4-5-curated, or disconnect Precise Reference."
        )

    params["director_reference_images"] = [r["image"] for r in refs]
    # API expects V4ConditionInput objects here, not plain strings.
    # base_caption selects what kind of precise reference is applied.
    params["director_reference_descriptions"] = [
        {
            "caption": {"base_caption": "character&style", "char_captions": []},
            "legacy_uc": False,
        }
        for _ in refs
    ]
    params["director_reference_strength_values"] = [float(r["strength"]) for r in refs]
    params["director_reference_secondary_strength_values"] = [1.0 - float(r["information_extracted"]) for r in refs]
    params["director_reference_information_extracted"] = [1.0 for _ in refs]
    return len(refs)


def perform_novelai_request(
    *,
    payload: Dict[str, Any],
    api_token: str,
    timeout: int,
    retry_delay: int,
    max_retries: int,
    retry_forever: bool,
    check_anlas: bool,
    estimated_cost: int = 0,
) -> Tuple[torch.Tensor, str, int, int, int, str]:
    token, source = get_token(api_token)
    print(f"[NovelAI] Using token from {source}: {_masked_token(token)}")

    before = None
    if check_anlas:
        before, msg = get_anlas_balance(token, timeout=min(timeout, 30))
        print(f"[NovelAI] Anlas before: {before if before is not None else msg}")

    data = request_with_retry(
        GEN_ENDPOINT,
        payload,
        token,
        timeout=timeout,
        retry_delay=retry_delay,
        max_retries=max_retries,
        retry_forever=retry_forever,
    )
    images = extract_images_from_response(data)
    image_tensor = stack_images(images)

    after = None
    if check_anlas:
        after, msg = get_anlas_balance(token, timeout=min(timeout, 30))
        print(f"[NovelAI] Anlas after: {after if after is not None else msg}")

    actual_cost = 0
    tracker_status = ""
    if before is not None and after is not None:
        actual_cost = max(0, int(before) - int(after))
        tracked_last, _, tracker_status = update_anlas_tracker(after, previous_hint=before, source=source, note="after generation")
        actual_cost = tracked_last if tracked_last > 0 else actual_cost
    else:
        _, _, tracker_status = update_anlas_tracker(after if after is not None else before, source=source, note="after generation")
    estimated_next_cost = actual_cost if actual_cost > 0 else int(estimated_cost)
    anlas_text = (
        f"{tracker_status}"
        f" | Estimated Cost: {estimated_next_cost}"
    )
    return image_tensor, anlas_text, int(before or 0), int(after or 0), int(actual_cost), source


def build_parameters(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    sampler: str,
    scheduler: str,
    noise_schedule: str,
    steps: int,
    cfg_scale: float,
    cfg_rescale: float,
    uc_preset: str,
    quality_toggle: bool,
    prefer_brownian: bool,
    sm: bool,
    sm_dyn: bool,
    batch_size: int,
    legacy: bool,
    character_prompts_json: str = "",
    character_prompts: Any = None,
) -> Dict[str, Any]:
    uc_preset_str = str(uc_preset)
    uc_index = _safe_int(uc_preset_str, 0)
    full_negative = _append_csv(negative_prompt, UC_PRESET_TEXT.get(uc_preset_str, ""))
    api_sampler = "ddim_v3" if sampler == "ddim" else sampler
    char_prompts = normalize_character_prompts(character_prompts_json, character_prompts)

    char_captions = []
    uc_char_captions = []
    uses_ai_choices = False
    for cp in char_prompts:
        center = cp.get("center") if isinstance(cp.get("center"), dict) else {"x": 0.5, "y": 0.5}
        safe_center = {"x": _clamp01(center.get("x", 0.5)), "y": _clamp01(center.get("y", 0.5))}
        if cp.get("prompt"):
            char_captions.append({"char_caption": cp["prompt"], "centers": [safe_center]})
        if cp.get("uc"):
            uc_char_captions.append({"char_caption": cp["uc"], "centers": [safe_center]})

    params: Dict[str, Any] = {
        "params_version": 1,
        "width": int(width),
        "height": int(height),
        "scale": float(cfg_scale),
        "cfg_scale": float(cfg_scale),
        "sampler": api_sampler,
        "sampler_name": api_sampler,
        "steps": int(steps),
        "seed": int(seed),
        "n_samples": max(1, int(batch_size)),
        "batch_size": 1,
        "n_iter": max(1, int(batch_size)),
        "ucPreset": uc_index,
        "qualityToggle": bool(quality_toggle),
        "sm": bool(sm),
        "sm_dyn": bool(sm_dyn),
        "dynamic_thresholding": False,
        "controlnet_strength": 1.0,
        "legacy": bool(legacy),
        "add_original_image": False,
        "cfg_rescale": float(cfg_rescale),
        "noise_schedule": noise_schedule,
        "scheduler": scheduler,
        "legacy_v3_extend": False,
        "uncond_scale": 1.0,
        "negative_prompt": full_negative,
        "prompt": prompt or "",
        "uc": full_negative,
        "reference_image_multiple": [],
        "reference_information_extracted_multiple": [],
        "reference_strength_multiple": [],
        "extra_noise_seed": int(seed),
        "characterPrompts": char_prompts,
        "v4_prompt": {
            "use_coords": True,
            "use_order": True,
            "caption": {"base_caption": prompt or "", "char_captions": char_captions},
        },
        "v4_negative_prompt": {
            "use_coords": False,
            "use_order": False,
            "caption": {"base_caption": full_negative, "char_captions": uc_char_captions},
        },
    }

    if api_sampler == "k_euler_ancestral" and scheduler != "native":
        params["deliberate_euler_ancestral_bug"] = False
    params["prefer_brownian"] = bool(prefer_brownian)
    return params


def choose_seed(seed: int, seed_mode: str, counter_name: str) -> int:
    if seed_mode == "random_each_run":
        return random.randint(0, 9999999999)
    if seed_mode == "increment_each_run":
        current = getattr(choose_seed, counter_name, 0)
        setattr(choose_seed, counter_name, current + 1)
        return int(seed) + current
    return int(seed)


def estimate_anlas_cost(*, width: int, height: int, steps: int, batch_size: int, img2img: bool, quality_toggle: bool) -> int:
    """Rough heuristic only. Real cost is measured via before/after balance when check_anlas=True."""
    pixels = int(width) * int(height)
    megapixels = pixels / 1048576.0
    base = 0
    # Very rough heuristic: common 1MP generations are often free/cheap; bigger jobs tend to cost more.
    if megapixels > 1.05:
        base += int(round((megapixels - 1.0) * 18))
    if int(steps) > 28:
        base += max(0, int((int(steps) - 28) / 4))
    if int(batch_size) > 1:
        base += (int(batch_size) - 1) * max(1, base or 2)
    if img2img:
        base += 0
    if bool(quality_toggle) and megapixels > 1.5:
        base += 2
    return max(0, int(base))


def generate_novelai(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    model: str,
    seed: int,
    seed_mode: str,
    sampler: str,
    scheduler: str,
    noise_schedule: str,
    steps: int,
    cfg_scale: float,
    cfg_rescale: float,
    uc_preset: str,
    quality_toggle: bool,
    prefer_brownian: bool,
    sm: bool,
    sm_dyn: bool,
    batch_size: int,
    legacy: bool,
    api_token: str,
    timeout: int,
    retry_delay: int,
    max_retries: int,
    retry_forever: bool,
    check_anlas: bool,
    img2img_image: Optional[torch.Tensor] = None,
    strength: float = 0.5,
    noise: float = 0.1,
    character_prompts_json: str = "",
    character_prompts: Any = None,
    parameters: Any = None,
    characters: Any = None,
    references: Any = None,
    **extra: Any,
) -> Tuple[torch.Tensor, str, str, int, int, int]:
    token, source = get_token(api_token)
    print(f"[NovelAI] Using token from {source}: {_masked_token(token)}")

    resolved = merge_parameter_values(parameters)
    width = int(resolved["width"])
    height = int(resolved["height"])
    model = str(resolved["model"])
    seed = int(resolved["seed"])
    seed_mode = str(resolved["seed_mode"])
    sampler = str(resolved["sampler"])
    scheduler = str(resolved["scheduler"])
    noise_schedule = str(resolved["noise_schedule"])
    steps = int(resolved["steps"])
    cfg_scale = float(resolved["cfg_scale"])
    cfg_rescale = float(resolved["cfg_rescale"])
    uc_preset = str(resolved["uc_preset"])
    quality_toggle = bool(resolved["quality_toggle"])
    prefer_brownian = bool(resolved["prefer_brownian"])
    sm = bool(resolved["sm"])
    sm_dyn = bool(resolved["sm_dyn"])
    batch_size = int(resolved["batch_size"])
    legacy = bool(resolved["legacy"])
    check_anlas = bool(resolved["check_anlas"])
    if img2img_image is not None and not isinstance(parameters, dict):
        try:
            if img2img_image.ndim == 4:
                h, w = img2img_image.shape[1], img2img_image.shape[2]
            else:
                h, w = img2img_image.shape[0], img2img_image.shape[1]
            width = int(w)
            height = int(h)
        except Exception:
            pass

    estimated_cost = estimate_anlas_cost(
        width=width,
        height=height,
        steps=steps,
        batch_size=batch_size,
        img2img=img2img_image is not None,
        quality_toggle=quality_toggle,
    )

    before = None
    if check_anlas:
        before, msg = get_anlas_balance(token, timeout=min(timeout, 30))
        print(f"[NovelAI] Anlas before: {before if before is not None else msg}")

    actual_seed = choose_seed(seed, seed_mode, "counter_i2i" if img2img_image is not None else "counter_t2i")
    slot_character_prompts = build_character_prompts_from_slots(extra)
    if characters is not None:
        effective_character_prompts_json = ""
        effective_character_prompts = characters
    elif slot_character_prompts:
        effective_character_prompts_json = ""
        effective_character_prompts = slot_character_prompts
    else:
        effective_character_prompts_json = character_prompts_json
        effective_character_prompts = character_prompts

    params = build_parameters(
        prompt=prompt or "",
        negative_prompt=negative_prompt or "",
        width=width,
        height=height,
        seed=actual_seed,
        sampler=sampler,
        scheduler=scheduler,
        noise_schedule=noise_schedule,
        steps=steps,
        cfg_scale=cfg_scale,
        cfg_rescale=cfg_rescale,
        uc_preset=uc_preset,
        quality_toggle=quality_toggle,
        prefer_brownian=prefer_brownian,
        sm=sm,
        sm_dyn=sm_dyn,
        batch_size=batch_size,
        legacy=legacy,
        character_prompts_json=effective_character_prompts_json,
        character_prompts=effective_character_prompts,
    )
    reference_count = apply_references_to_params(params, references, model=model)

    action = "generate"
    if img2img_image is not None:
        action = "img2img"
        params["image"] = image_tensor_to_base64_png(img2img_image, width, height)
        params["strength"] = float(strength)
        params["noise"] = float(noise)

    payload = {
        "input": prompt or "",
        "model": model,
        "action": action,
        "parameters": params,
    }

    print(
        f"[NovelAI] request action={action}, model={model}, size={int(width)}x{int(height)}, "
        f"seed={actual_seed}, sampler={sampler}, scheduler={scheduler}, noise_schedule={noise_schedule}"
    )

    data = request_with_retry(
        GEN_ENDPOINT,
        payload,
        token,
        timeout=timeout,
        retry_delay=retry_delay,
        max_retries=max_retries,
        retry_forever=retry_forever,
    )
    images = extract_images_from_response(data)
    image_tensor = stack_images(images)

    after = None
    if check_anlas:
        after, msg = get_anlas_balance(token, timeout=min(timeout, 30))
        print(f"[NovelAI] Anlas after: {after if after is not None else msg}")

    actual_cost = 0
    tracker_status = ""
    if before is not None and after is not None:
        actual_cost = max(0, int(before) - int(after))
        tracked_last, _, tracker_status = update_anlas_tracker(after, previous_hint=before, source=source, note="after generation")
        actual_cost = tracked_last if tracked_last > 0 else actual_cost
    else:
        _, _, tracker_status = update_anlas_tracker(after if after is not None else before, source=source, note="after generation")
    estimated_next_cost = actual_cost if actual_cost > 0 else estimated_cost
    anlas_text = (
        f"{tracker_status}"
        f" | Estimated Cost: {estimated_next_cost}"
    )

    info = {
        "mode": action,
        "model": model,
        "seed": actual_seed,
        "width": int(width),
        "height": int(height),
        "sampler": sampler,
        "scheduler": scheduler,
        "noise_schedule": noise_schedule,
        "steps": int(steps),
        "cfg_scale": float(cfg_scale),
        "cfg_rescale": float(cfg_rescale),
        "batch_size": int(batch_size),
        "character_count": len(normalize_character_prompts("", effective_character_prompts)) if effective_character_prompts is not None else len(normalize_character_prompts(effective_character_prompts_json, None)),
        "reference_count": int(reference_count),
        "strength": float(strength) if img2img_image is not None else None,
        "noise": float(noise) if img2img_image is not None else None,
        "anlas_before": before,
        "anlas_after": after,
        "estimated_cost": estimated_next_cost,
        "last_actual_cost": actual_cost,
        "anlas_text": anlas_text,
        "token_source": source,
    }
    return image_tensor, json.dumps(info, ensure_ascii=False, indent=2), anlas_text, int(before or 0), int(after or 0), int(actual_cost)


class NovelAIT2ILegacy:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "Independent NovelAI text-to-image node with retry and Anlas cost info."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "masterpiece, best quality, 1girl"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "lowres, bad anatomy, bad hands, text, error"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "model": (MODEL_CHOICES, {"default": "nai-diffusion-4-5-full"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
                "seed_mode": (SEED_MODE_CHOICES, {"default": "random_each_run"}),
                "sampler": (SAMPLER_CHOICES, {"default": "k_euler_ancestral"}),
                "scheduler": (SCHEDULER_CHOICES, {"default": "karras"}),
                "noise_schedule": (NOISE_SCHEDULE_CHOICES, {"default": "native"}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 50.0, "step": 0.1}),
                "cfg_rescale": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "uc_preset": (UC_PRESET_CHOICES, {"default": "0"}),
                "quality_toggle": ("BOOLEAN", {"default": True}),
                "prefer_brownian": ("BOOLEAN", {"default": True}),
                "sm": ("BOOLEAN", {"default": False}),
                "sm_dyn": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "legacy": ("BOOLEAN", {"default": False}),
                "check_anlas": ("BOOLEAN", {"default": False}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 600}),
                "retry_delay": ("INT", {"default": 10, "min": 1, "max": 300}),
                "max_retries": ("INT", {"default": 5, "min": 0, "max": 999}),
                "retry_forever": ("BOOLEAN", {"default": True}),
                **character_slot_inputs(),
            },
            "optional": {
                "character_prompts": ("LIST",),
                "character_prompts_json": ("STRING", {"multiline": True, "default": ""}),
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def generate(self, **kwargs):
        return generate_novelai(**kwargs)


class NovelAII2ILegacy:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "Independent NovelAI image-to-image node with retry and Anlas cost info."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "masterpiece, best quality"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "lowres, bad anatomy, bad hands, text, error"}),
                "strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "model": (MODEL_CHOICES, {"default": "nai-diffusion-4-5-full"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
                "seed_mode": (SEED_MODE_CHOICES, {"default": "random_each_run"}),
                "sampler": (SAMPLER_CHOICES, {"default": "k_euler_ancestral"}),
                "scheduler": (SCHEDULER_CHOICES, {"default": "karras"}),
                "noise_schedule": (NOISE_SCHEDULE_CHOICES, {"default": "native"}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 50.0, "step": 0.1}),
                "cfg_rescale": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "uc_preset": (UC_PRESET_CHOICES, {"default": "0"}),
                "quality_toggle": ("BOOLEAN", {"default": True}),
                "prefer_brownian": ("BOOLEAN", {"default": True}),
                "sm": ("BOOLEAN", {"default": False}),
                "sm_dyn": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "legacy": ("BOOLEAN", {"default": False}),
                "check_anlas": ("BOOLEAN", {"default": False}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 600}),
                "retry_delay": ("INT", {"default": 10, "min": 1, "max": 300}),
                "max_retries": ("INT", {"default": 5, "min": 0, "max": 999}),
                "retry_forever": ("BOOLEAN", {"default": True}),
                **character_slot_inputs(),
            },
            "optional": {
                "character_prompts": ("LIST",),
                "character_prompts_json": ("STRING", {"multiline": True, "default": ""}),
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def generate(self, image, **kwargs):
        kwargs["img2img_image"] = image
        return generate_novelai(**kwargs)


class NovelAIToken:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_token",)
    FUNCTION = "output"
    DESCRIPTION = "Outputs a STRING token for connection to NovelAI nodes. Use this so you only type the token once."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": "", "multiline": False}),
            }
        }

    def output(self, api_token=""):
        return (str(api_token or "").strip(),)


class NovelAIParameters:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("NAI_PARAMETERS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "build"
    DESCRIPTION = "Builds a reusable NovelAI parameters block for T2I/I2I nodes."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": parameter_node_inputs()}

    def build(self, **kwargs):
        config = merge_parameter_values(kwargs)
        return (config,)


def merge_retry_values(retry_settings: Any = None) -> Dict[str, Any]:
    merged = dict(DEFAULT_RETRY_VALUES)
    if isinstance(retry_settings, dict):
        merged.update({k: v for k, v in retry_settings.items() if k in merged})
    return merged


class NovelAIRetrySettings:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("NAI_RETRY_SETTINGS",)
    RETURN_NAMES = ("retry_settings",)
    FUNCTION = "build"
    DESCRIPTION = "Builds reusable timeout and retry settings for NovelAI T2I/I2I nodes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeout": ("INT", {"default": DEFAULT_RETRY_VALUES["timeout"], "min": 10, "max": 600}),
                "retry_delay": ("INT", {"default": DEFAULT_RETRY_VALUES["retry_delay"], "min": 1, "max": 300}),
                "max_retries": ("INT", {"default": DEFAULT_RETRY_VALUES["max_retries"], "min": 0, "max": 999}),
                "retry_forever": ("BOOLEAN", {"default": DEFAULT_RETRY_VALUES["retry_forever"]}),
            }
        }

    def build(self, **kwargs):
        return (merge_retry_values(kwargs),)


class NovelAICharactersLegacy:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("NAI_CHARACTERS",)
    RETURN_NAMES = ("characters",)
    FUNCTION = "build"
    DESCRIPTION = "Builds reusable NovelAI character prompts with up to 5 fixed slots. The new Single Character node is usually more convenient."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {**character_slot_inputs()},
            "optional": {
                "character_prompts_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def build(self, character_prompts_json="", **kwargs):
        slot_chars = build_character_prompts_from_slots(kwargs)
        if slot_chars:
            chars = slot_chars
        else:
            chars = normalize_character_prompts(character_prompts_json, None)
        return (chars,)


class NovelAICharacter:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("NAI_CHARACTERS",)
    RETURN_NAMES = ("characters",)
    FUNCTION = "build"
    DESCRIPTION = "Single character box with chain input/output. Connect multiple nodes to add more characters. Uses NovelAI-like A-E / 1-5 placement grid."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "position_col": (CHARACTER_GRID_COLS, {"default": "C"}),
                "position_row": (CHARACTER_GRID_ROWS, {"default": "3"}),
            },
            "optional": {
                "characters": ("NAI_CHARACTERS",),
            },
        }

    def build(self, enabled=True, prompt="", negative="", position_col="C", position_row="3", characters=None):
        current = []
        if isinstance(characters, list):
            current = [dict(x) for x in characters]
        elif isinstance(characters, dict):
            current = [dict(characters)]

        prompt = str(prompt or "").strip()
        negative = str(negative or "").strip()
        if enabled and prompt:
            center = _character_grid_position(position_col, position_row)
            current.append({
                "prompt": prompt,
                "uc": negative,
                "center": center,
                "grid": {"col": str(position_col), "row": str(position_row)},
            })
        return (current,)


class NovelAIPreciseReference:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("NAI_REFERENCES",)
    RETURN_NAMES = ("references",)
    FUNCTION = "build"
    DESCRIPTION = "💎 Precise Reference builder. Adds a reference image for NovelAI generation. This feature can spend Anlas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "information_extracted": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "references": ("NAI_REFERENCES",),
            },
        }

    def build(self, image, enabled=True, information_extracted=0.50, strength=0.60, references=None):
        current: List[Dict[str, Any]] = []
        if isinstance(references, list):
            current = [dict(x) for x in references if isinstance(x, dict)]
        elif isinstance(references, dict):
            current = [dict(references)]
        if enabled:
            if image.ndim == 4:
                h, w = int(image.shape[1]), int(image.shape[2])
            else:
                h, w = int(image.shape[0]), int(image.shape[1])
            current.append({
                "mode": "precise_reference",
                "image": precise_reference_tensor_to_base64(image),
                "information_extracted": float(information_extracted),
                "strength": float(strength),
            })
        return (current,)


class NovelAICharacterStack:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("NAI_CHARACTERS",)
    RETURN_NAMES = ("characters",)
    FUNCTION = "build"
    DESCRIPTION = "Combines up to 8 character inputs. Position Mode: position random = NovelAI-like AI Choices; position manual = keep Character grid positions."

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 9):
            optional[f"character_{i}"] = ("NAI_CHARACTERS",)
        return {
            "required": {
                "position_mode": (["position random", "position manual"], {"default": "position random"}),
            },
            "optional": optional,
        }

    def build(self, position_mode="position random", **kwargs):
        combined = []
        for i in range(1, 9):
            key = f"character_{i}"
            value = kwargs.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        combined.append(dict(item))
            elif isinstance(value, dict):
                combined.append(dict(value))

        normalized = normalize_character_prompts("", combined)
        mode = str(position_mode or "position random").strip().lower()

        if mode in {"position random", "random", "ai_choices", "ai_choice", "random_grid"}:
            grid_slots = [(c, r) for r in CHARACTER_GRID_ROWS for c in CHARACTER_GRID_COLS]
            random.shuffle(grid_slots)
            converted = []
            for idx, cp in enumerate(normalized):
                col, row = grid_slots[idx % len(grid_slots)]
                item = dict(cp)
                item.pop("ai_choice", None)
                item["position_mode"] = "position random"
                item["grid"] = {"col": col, "row": row}
                item["center"] = _character_grid_position(col, row)
                converted.append(item)
            return (converted,)

        converted = []
        for cp in normalized:
            item = dict(cp)
            item.pop("ai_choice", None)
            item["position_mode"] = "position manual"
            converted.append(item)
        return (converted,)


class NovelAIT2ICompact:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "anlas", "last_actual_cost", "actual_cost_total", "status_text")
    FUNCTION = "generate"
    DESCRIPTION = "Compact NovelAI text-to-image node that receives parameters, retry settings, characters and references from separate builder nodes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "masterpiece, best quality, 1girl"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "lowres, bad anatomy, bad hands, text, error"}),
            },
            "optional": {
                "parameters": ("NAI_PARAMETERS",),
                "retry_settings": ("NAI_RETRY_SETTINGS",),
                "characters": ("NAI_CHARACTERS",),
                "references": ("NAI_REFERENCES",),
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def generate(self, prompt, negative_prompt, parameters=None, retry_settings=None, characters=None, references=None, api_token=""):
        base = merge_parameter_values(parameters)
        retry = merge_retry_values(retry_settings)
        result = generate_novelai(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=base["width"],
            height=base["height"],
            model=base["model"],
            seed=base["seed"],
            seed_mode=base["seed_mode"],
            sampler=base["sampler"],
            scheduler=base["scheduler"],
            noise_schedule=base["noise_schedule"],
            steps=base["steps"],
            cfg_scale=base["cfg_scale"],
            cfg_rescale=base["cfg_rescale"],
            uc_preset=base["uc_preset"],
            quality_toggle=base["quality_toggle"],
            prefer_brownian=base["prefer_brownian"],
            sm=base["sm"],
            sm_dyn=base["sm_dyn"],
            batch_size=base["batch_size"],
            legacy=base["legacy"],
            api_token=api_token,
            timeout=retry["timeout"],
            retry_delay=retry["retry_delay"],
            max_retries=retry["max_retries"],
            retry_forever=retry["retry_forever"],
            check_anlas=True,
            parameters=parameters,
            characters=characters,
            references=references,
        )
        return (result[0], int(result[4]), int(result[5]), get_anlas_tracker_total(), str(result[2]))


class NovelAII2ICompact:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "anlas", "last_actual_cost", "actual_cost_total", "status_text")
    FUNCTION = "generate"
    DESCRIPTION = "Compact NovelAI image-to-image node that receives parameters, retry settings, characters and references from separate builder nodes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "masterpiece, best quality"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "lowres, bad anatomy, bad hands, text, error"}),
                "strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "parameters": ("NAI_PARAMETERS",),
                "retry_settings": ("NAI_RETRY_SETTINGS",),
                "characters": ("NAI_CHARACTERS",),
                "references": ("NAI_REFERENCES",),
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def generate(self, image, prompt, negative_prompt, strength, noise, parameters=None, retry_settings=None, characters=None, references=None, api_token=""):
        base = merge_parameter_values(parameters)
        retry = merge_retry_values(retry_settings)
        result = generate_novelai(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=base["width"],
            height=base["height"],
            model=base["model"],
            seed=base["seed"],
            seed_mode=base["seed_mode"],
            sampler=base["sampler"],
            scheduler=base["scheduler"],
            noise_schedule=base["noise_schedule"],
            steps=base["steps"],
            cfg_scale=base["cfg_scale"],
            cfg_rescale=base["cfg_rescale"],
            uc_preset=base["uc_preset"],
            quality_toggle=base["quality_toggle"],
            prefer_brownian=base["prefer_brownian"],
            sm=base["sm"],
            sm_dyn=base["sm_dyn"],
            batch_size=base["batch_size"],
            legacy=base["legacy"],
            api_token=api_token,
            timeout=retry["timeout"],
            retry_delay=retry["retry_delay"],
            max_retries=retry["max_retries"],
            retry_forever=retry["retry_forever"],
            check_anlas=base["check_anlas"],
            img2img_image=image,
            strength=strength,
            noise=noise,
            parameters=parameters,
            characters=characters,
            references=references,
        )
        return (result[0], int(result[4]), int(result[5]), get_anlas_tracker_total(), str(result[2]))


class NovelAIInpaint:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "💎 NovelAI inpaint/infill node with image + mask. Can spend Anlas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "masterpiece, best quality"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "lowres, bad anatomy, bad hands, text, error"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "parameters": ("NAI_PARAMETERS",),
                "retry_settings": ("NAI_RETRY_SETTINGS",),
                "characters": ("NAI_CHARACTERS",),
                "references": ("NAI_REFERENCES",),
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def generate(self, image, mask, prompt, negative_prompt, invert_mask=False, strength=0.50, noise=0.10, parameters=None, retry_settings=None, characters=None, references=None, api_token=""):
        base = merge_parameter_values(parameters)
        retry = merge_retry_values(retry_settings)
        width = int(base["width"])
        height = int(base["height"])
        if image is not None:
            try:
                if image.ndim == 4:
                    height, width = int(image.shape[1]), int(image.shape[2])
                else:
                    height, width = int(image.shape[0]), int(image.shape[1])
            except Exception:
                pass
        actual_seed = choose_seed(int(base["seed"]), str(base["seed_mode"]), "counter_inpaint")
        params = build_parameters(
            prompt=prompt or "",
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            seed=actual_seed,
            sampler=str(base["sampler"]),
            scheduler=str(base["scheduler"]),
            noise_schedule=str(base["noise_schedule"]),
            steps=int(base["steps"]),
            cfg_scale=float(base["cfg_scale"]),
            cfg_rescale=float(base["cfg_rescale"]),
            uc_preset=str(base["uc_preset"]),
            quality_toggle=bool(base["quality_toggle"]),
            prefer_brownian=bool(base["prefer_brownian"]),
            sm=bool(base["sm"]),
            sm_dyn=bool(base["sm_dyn"]),
            batch_size=int(base["batch_size"]),
            legacy=bool(base["legacy"]),
            character_prompts_json="",
            character_prompts=characters,
        )
        reference_count = apply_references_to_params(params, references, model=str(base["model"]))
        params["image"] = image_tensor_to_base64_png(image, width, height)
        params["mask"] = mask_tensor_to_base64_png(mask, width, height, bool(invert_mask))
        params["strength"] = float(strength)
        params["noise"] = float(noise)
        payload = {
            "input": prompt or "",
            "model": str(base["model"]),
            "action": "infill",
            "parameters": params,
        }
        image_tensor, anlas_text, before, after, actual_cost, source = perform_novelai_request(
            payload=payload,
            api_token=api_token,
            timeout=int(retry["timeout"]),
            retry_delay=int(retry["retry_delay"]),
            max_retries=int(retry["max_retries"]),
            retry_forever=bool(retry["retry_forever"]),
            check_anlas=True,
            estimated_cost=estimate_anlas_cost(width=width, height=height, steps=int(base["steps"]), batch_size=1, img2img=True, quality_toggle=bool(base["quality_toggle"])) + 1,
        )
        info = {
            "mode": "infill",
            "model": str(base["model"]),
            "seed": actual_seed,
            "width": width,
            "height": height,
            "sampler": str(base["sampler"]),
            "scheduler": str(base["scheduler"]),
            "noise_schedule": str(base["noise_schedule"]),
            "steps": int(base["steps"]),
            "cfg_scale": float(base["cfg_scale"]),
            "cfg_rescale": float(base["cfg_rescale"]),
            "character_count": len(normalize_character_prompts("", characters)) if characters is not None else 0,
            "reference_count": int(reference_count),
            "strength": float(strength),
            "noise": float(noise),
            "anlas_before": before,
            "anlas_after": after,
            "last_actual_cost": actual_cost,
            "anlas_text": anlas_text,
            "token_source": source,
        }
        return image_tensor, json.dumps(info, ensure_ascii=False, indent=2), anlas_text, before, after, actual_cost


class NovelAIEnhance:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "💎 NovelAI enhance node. Can spend Anlas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "masterpiece, best quality"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "lowres, bad anatomy, bad hands, text, error"}),
                "strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "parameters": ("NAI_PARAMETERS",),
                "retry_settings": ("NAI_RETRY_SETTINGS",),
                "characters": ("NAI_CHARACTERS",),
                "references": ("NAI_REFERENCES",),
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def generate(self, image, prompt, negative_prompt, strength=0.50, noise=0.10, parameters=None, retry_settings=None, characters=None, references=None, api_token=""):
        base = merge_parameter_values(parameters)
        retry = merge_retry_values(retry_settings)
        width = int(base["width"])
        height = int(base["height"])
        if image is not None:
            try:
                if image.ndim == 4:
                    height, width = int(image.shape[1]), int(image.shape[2])
                else:
                    height, width = int(image.shape[0]), int(image.shape[1])
            except Exception:
                pass
        actual_seed = choose_seed(int(base["seed"]), str(base["seed_mode"]), "counter_enhance")
        params = build_parameters(
            prompt=prompt or "",
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            seed=actual_seed,
            sampler=str(base["sampler"]),
            scheduler=str(base["scheduler"]),
            noise_schedule=str(base["noise_schedule"]),
            steps=int(base["steps"]),
            cfg_scale=float(base["cfg_scale"]),
            cfg_rescale=float(base["cfg_rescale"]),
            uc_preset=str(base["uc_preset"]),
            quality_toggle=bool(base["quality_toggle"]),
            prefer_brownian=bool(base["prefer_brownian"]),
            sm=bool(base["sm"]),
            sm_dyn=bool(base["sm_dyn"]),
            batch_size=1,
            legacy=bool(base["legacy"]),
            character_prompts_json="",
            character_prompts=characters,
        )
        reference_count = apply_references_to_params(params, references, model=str(base["model"]))
        params["image"] = image_tensor_to_base64_png(image, width, height)
        params["strength"] = float(strength)
        params["noise"] = float(noise)
        payload = {
            "input": prompt or "",
            "model": str(base["model"]),
            "action": "enhance",
            "parameters": params,
        }
        image_tensor, anlas_text, before, after, actual_cost, source = perform_novelai_request(
            payload=payload,
            api_token=api_token,
            timeout=int(retry["timeout"]),
            retry_delay=int(retry["retry_delay"]),
            max_retries=int(retry["max_retries"]),
            retry_forever=bool(retry["retry_forever"]),
            check_anlas=True,
            estimated_cost=estimate_anlas_cost(width=width, height=height, steps=int(base["steps"]), batch_size=1, img2img=True, quality_toggle=bool(base["quality_toggle"])) + 1,
        )
        info = {
            "mode": "enhance",
            "model": str(base["model"]),
            "seed": actual_seed,
            "width": width,
            "height": height,
            "sampler": str(base["sampler"]),
            "scheduler": str(base["scheduler"]),
            "noise_schedule": str(base["noise_schedule"]),
            "steps": int(base["steps"]),
            "cfg_scale": float(base["cfg_scale"]),
            "cfg_rescale": float(base["cfg_rescale"]),
            "character_count": len(normalize_character_prompts("", characters)) if characters is not None else 0,
            "reference_count": int(reference_count),
            "strength": float(strength),
            "noise": float(noise),
            "anlas_before": before,
            "anlas_after": after,
            "last_actual_cost": actual_cost,
            "anlas_text": anlas_text,
            "token_source": source,
        }
        return image_tensor, json.dumps(info, ensure_ascii=False, indent=2), anlas_text, before, after, actual_cost


class NovelAIUpscale:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "💎 NovelAI upscale node. Can spend Anlas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": (["2", "4"], {"default": "2"}),
            },
            "optional": {
                "parameters": ("NAI_PARAMETERS",),
                "retry_settings": ("NAI_RETRY_SETTINGS",),
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def generate(self, image, scale_factor="2", parameters=None, retry_settings=None, api_token=""):
        base = merge_parameter_values(parameters)
        retry = merge_retry_values(retry_settings)
        width = int(base["width"])
        height = int(base["height"])
        if image is not None:
            try:
                if image.ndim == 4:
                    height, width = int(image.shape[1]), int(image.shape[2])
                else:
                    height, width = int(image.shape[0]), int(image.shape[1])
            except Exception:
                pass
        factor = int(scale_factor)
        target_w = width * factor
        target_h = height * factor
        params = {
            "width": target_w,
            "height": target_h,
            "image": image_tensor_to_base64_png(image, width, height),
            "scale": factor,
        }
        payload = {
            "input": "",
            "model": str(base["model"]),
            "action": "upscale",
            "parameters": params,
        }
        image_tensor, anlas_text, before, after, actual_cost, source = perform_novelai_request(
            payload=payload,
            api_token=api_token,
            timeout=int(retry["timeout"]),
            retry_delay=int(retry["retry_delay"]),
            max_retries=int(retry["max_retries"]),
            retry_forever=bool(retry["retry_forever"]),
            check_anlas=True,
            estimated_cost=max(1, factor * factor),
        )
        info = {
            "mode": "upscale",
            "model": str(base["model"]),
            "input_width": width,
            "input_height": height,
            "width": target_w,
            "height": target_h,
            "scale_factor": factor,
            "anlas_before": before,
            "anlas_after": after,
            "last_actual_cost": actual_cost,
            "anlas_text": anlas_text,
            "token_source": source,
        }
        return image_tensor, json.dumps(info, ensure_ascii=False, indent=2), anlas_text, before, after, actual_cost


def run_director_tool(
    *,
    action: str,
    image,
    parameters: Any = None,
    retry_settings: Any = None,
    api_token: str = "",
    prompt: str = "",
    negative_prompt: str = "",
    characters: Any = None,
    references: Any = None,
    extra_params: Optional[Dict[str, Any]] = None,
    estimated_cost: int = 1,
) -> Tuple[torch.Tensor, str, str, int, int, int]:
    base = merge_parameter_values(parameters)
    retry = merge_retry_values(retry_settings)
    width = int(base["width"])
    height = int(base["height"])
    if image is not None:
        try:
            if image.ndim == 4:
                height, width = int(image.shape[1]), int(image.shape[2])
            else:
                height, width = int(image.shape[0]), int(image.shape[1])
        except Exception:
            pass

    actual_seed = choose_seed(int(base["seed"]), str(base["seed_mode"]), f"counter_{action}")
    params = build_parameters(
        prompt=prompt or "",
        negative_prompt=negative_prompt or "",
        width=width,
        height=height,
        seed=actual_seed,
        sampler=str(base["sampler"]),
        scheduler=str(base["scheduler"]),
        noise_schedule=str(base["noise_schedule"]),
        steps=int(base["steps"]),
        cfg_scale=float(base["cfg_scale"]),
        cfg_rescale=float(base["cfg_rescale"]),
        uc_preset=str(base["uc_preset"]),
        quality_toggle=bool(base["quality_toggle"]),
        prefer_brownian=bool(base["prefer_brownian"]),
        sm=bool(base["sm"]),
        sm_dyn=bool(base["sm_dyn"]),
        batch_size=1,
        legacy=bool(base["legacy"]),
        character_prompts_json="",
        character_prompts=characters,
    )
    reference_count = apply_references_to_params(params, references, model=str(base["model"]))
    params["image"] = image_tensor_to_base64_png(image, width, height)
    if extra_params:
        params.update(extra_params)

    payload = {
        "input": prompt or "",
        "model": str(base["model"]),
        "action": action,
        "parameters": params,
    }
    image_tensor, anlas_text, before, after, actual_cost, source = perform_novelai_request(
        payload=payload,
        api_token=api_token,
        timeout=int(retry["timeout"]),
        retry_delay=int(retry["retry_delay"]),
        max_retries=int(retry["max_retries"]),
        retry_forever=bool(retry["retry_forever"]),
        check_anlas=True,
        estimated_cost=int(estimated_cost),
    )
    info = {
        "mode": action,
        "model": str(base["model"]),
        "seed": actual_seed,
        "width": width,
        "height": height,
        "sampler": str(base["sampler"]),
        "scheduler": str(base["scheduler"]),
        "noise_schedule": str(base["noise_schedule"]),
        "steps": int(base["steps"]),
        "cfg_scale": float(base["cfg_scale"]),
        "cfg_rescale": float(base["cfg_rescale"]),
        "character_count": len(normalize_character_prompts("", characters)) if characters is not None else 0,
        "reference_count": int(reference_count),
        "anlas_before": before,
        "anlas_after": after,
        "last_actual_cost": actual_cost,
        "anlas_text": anlas_text,
        "token_source": source,
    }
    if extra_params:
        info["tool_parameters"] = extra_params
    return image_tensor, json.dumps(info, ensure_ascii=False, indent=2), anlas_text, before, after, actual_cost


class NovelAIRemoveBackground:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "💎 Remove Background director tool. Can spend Anlas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "result_mode": (["generated", "masked", "blend"], {"default": "generated"}),
            },
            "optional": {
                "parameters": ("NAI_PARAMETERS",),
                "retry_settings": ("NAI_RETRY_SETTINGS",),
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def generate(self, image, result_mode="generated", parameters=None, retry_settings=None, api_token=""):
        return run_director_tool(
            action="remove_background",
            image=image,
            parameters=parameters,
            retry_settings=retry_settings,
            api_token=api_token,
            extra_params={"result_mode": str(result_mode)},
            estimated_cost=1,
        )


class NovelAILineArt:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "💎 Line Art director tool. Can spend Anlas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"parameters": ("NAI_PARAMETERS",), "retry_settings": ("NAI_RETRY_SETTINGS",), "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True})},
        }

    def generate(self, image, parameters=None, retry_settings=None, api_token=""):
        return run_director_tool(action="lineart", image=image, parameters=parameters, retry_settings=retry_settings, api_token=api_token, estimated_cost=1)


class NovelAISketch:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "💎 Sketch director tool. Can spend Anlas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"parameters": ("NAI_PARAMETERS",), "retry_settings": ("NAI_RETRY_SETTINGS",), "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True})},
        }

    def generate(self, image, parameters=None, retry_settings=None, api_token=""):
        return run_director_tool(action="sketch", image=image, parameters=parameters, retry_settings=retry_settings, api_token=api_token, estimated_cost=1)


class NovelAIColorize:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "💎 Colorize director tool. Can spend Anlas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "defry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            },
            "optional": {
                "parameters": ("NAI_PARAMETERS",),
                "retry_settings": ("NAI_RETRY_SETTINGS",),
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def generate(self, image, prompt="", defry=0.0, parameters=None, retry_settings=None, api_token=""):
        return run_director_tool(
            action="colorize",
            image=image,
            parameters=parameters,
            retry_settings=retry_settings,
            api_token=api_token,
            prompt=prompt,
            extra_params={"defry": float(defry)},
            estimated_cost=1,
        )


EMOTION_CHOICES = [
    "neutral", "happy", "sad", "angry", "surprised", "embarrassed", "smug", "shy", "crying", "laughing", "confused", "determined"
]


class NovelAIEmotion:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "💎 Emotion director tool. Can spend Anlas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "emotion": (EMOTION_CHOICES, {"default": "neutral"}),
                "emotion_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "parameters": ("NAI_PARAMETERS",),
                "retry_settings": ("NAI_RETRY_SETTINGS",),
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def generate(self, image, emotion="neutral", emotion_level=0.5, prompt="", parameters=None, retry_settings=None, api_token=""):
        return run_director_tool(
            action="emotion",
            image=image,
            parameters=parameters,
            retry_settings=retry_settings,
            api_token=api_token,
            prompt=prompt,
            extra_params={"emotion": str(emotion), "emotion_level": float(emotion_level)},
            estimated_cost=1,
        )


class NovelAIDeclutter:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "info_json", "anlas_text", "anlas_before", "anlas_after", "actual_cost")
    FUNCTION = "generate"
    DESCRIPTION = "💎 Declutter director tool. Can spend Anlas."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"parameters": ("NAI_PARAMETERS",), "retry_settings": ("NAI_RETRY_SETTINGS",), "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True})},
        }

    def generate(self, image, parameters=None, retry_settings=None, api_token=""):
        return run_director_tool(action="declutter", image=image, parameters=parameters, retry_settings=retry_settings, api_token=api_token, estimated_cost=1)


class NovelAIAnlas:
    CATEGORY = "NovelAI"
    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("anlas", "last_actual_cost", "actual_cost_total", "status_text")
    FUNCTION = "check"
    DESCRIPTION = "Checks NovelAI Anlas balance. Remembers the previous balance internally and tracks generation cost automatically."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 30, "min": 5, "max": 120}),
            },
            "optional": {
                "api_token": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
            },
        }

    def check(self, trigger=True, timeout=30, api_token=""):
        if not trigger:
            current = int(ANLAS_LAST_BALANCE or 0)
            total = int(ANLAS_TOTAL_COST or 0)
            status = f"Anlas: {current if current > 0 else '?'} | Last Cost: 0 | Total Cost: {total} | check disabled"
            return (current, 0, total, status)

        token, source = get_token(api_token)
        value, msg = get_anlas_balance(token, timeout=int(timeout))
        if value is None:
            current = int(ANLAS_LAST_BALANCE or 0)
            total = int(ANLAS_TOTAL_COST or 0)
            status = f"Anlas unavailable ({msg}) | Last Cost: 0 | Total Cost: {total}"
            print(f"[NovelAI] {status}")
            return (current, 0, total, status)

        last_cost, total, status = update_anlas_tracker(int(value), source=source, note="manual check")
        print(f"[NovelAI] {status}")
        return (int(value), int(last_cost), int(total), status)


NODE_CLASS_MAPPINGS = {
    "NovelAIToken": NovelAIToken,
    "NovelAIParameters": NovelAIParameters,
    "NovelAIRetrySettings": NovelAIRetrySettings,
    "NovelAICharacter": NovelAICharacter,
    "NovelAIPreciseReference": NovelAIPreciseReference,
    "NovelAICharacterStack": NovelAICharacterStack,
    "NovelAIT2I": NovelAIT2ICompact,
    "NovelAII2I": NovelAII2ICompact,
    "NovelAIInpaint": NovelAIInpaint,
    "NovelAIEnhance": NovelAIEnhance,
    "NovelAIUpscale": NovelAIUpscale,
    "NovelAIRemoveBackground": NovelAIRemoveBackground,
    "NovelAILineArt": NovelAILineArt,
    "NovelAISketch": NovelAISketch,
    "NovelAIColorize": NovelAIColorize,
    "NovelAIEmotion": NovelAIEmotion,
    "NovelAIDeclutter": NovelAIDeclutter,
    "NovelAIAnlas": NovelAIAnlas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NovelAIToken": "NovelAI Token",
    "NovelAIParameters": "NovelAI Parameters",
    "NovelAIRetrySettings": "NovelAI Retry Settings",
    "NovelAICharacter": "NovelAI Character (V4.5)",
    "NovelAIPreciseReference": "NovelAI 💎 Precise Reference (V4.5)",
    "NovelAICharacterStack": "NovelAI Character Stack (V4.5)",
    "NovelAIT2I": "NovelAI T2I",
    "NovelAII2I": "NovelAI I2I",
    "NovelAIInpaint": "NovelAI 💎 Inpaint",
    "NovelAIEnhance": "NovelAI 💎 Enhance",
    "NovelAIUpscale": "NovelAI 💎 Upscale",
    "NovelAIRemoveBackground": "NovelAI 💎 Remove Background (Director Tool)",
    "NovelAILineArt": "NovelAI 💎 Line Art (Director Tool)",
    "NovelAISketch": "NovelAI 💎 Sketch (Director Tool)",
    "NovelAIColorize": "NovelAI 💎 Colorize (Director Tool)",
    "NovelAIEmotion": "NovelAI 💎 Emotion (Director Tool)",
    "NovelAIDeclutter": "NovelAI 💎 Declutter (Director Tool)",
    "NovelAIAnlas": "NovelAI Anlas",
}
