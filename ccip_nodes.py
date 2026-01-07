from typing import Any, Tuple, Optional, List
import numpy as np
from PIL import Image
import os
try:
    import folder_paths
except Exception:
    folder_paths = None


class CCIPModelLoader:
    @staticmethod
    def _models_base_from_folder_paths() -> Optional[str]:
        if folder_paths is None:
            return None
        try:
            base = os.path.join(folder_paths.models_dir, 'ccip')
            return os.path.abspath(base)
        except Exception:
            return None

    @classmethod
    def INPUT_TYPES(cls):
        base_dir = cls._models_base_from_folder_paths()

        subfolders: List[str] = []
        if base_dir and os.path.exists(base_dir):
            try:
                subfolders = [f for f in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, f))]
            except Exception:
                subfolders = []

        choices = subfolders

        return {
            "required": {
                "model_folder": (choices, {"default": choices[0] if choices else ""}),
                "device": ("STRING", {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("CCIP_MODEL",)
    FUNCTION = "load"
    CATEGORY = "CCIP"

    def load(self, model_choice: Optional[str] = None, model_folder: Optional[str] = None) -> Tuple[dict]:
        if model_folder is not None:
            model_choice = model_folder
        if not model_choice:
            raise ValueError('No model folder selected; remote option removed. Please choose a local model folder.')

        base = self._models_base_from_folder_paths()
        if base is None:
            raise FileNotFoundError('folder_paths.models_dir not available; cannot load local models. Please ensure folder_paths is installed and configured.')

        sel = os.path.join(base, model_choice)
        if not os.path.isdir(sel):
            raise FileNotFoundError(f'Selected model folder not found under {base}: {model_choice}')

        return ({"model": sel, "type": "local"},)


def _to_pil_image(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, str):
        return Image.open(img)
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            if arr.shape[0] in (1,):
                arr = np.squeeze(arr, 0)
            else:
                arr = arr.transpose(1, 2, 0)
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported image type: {type(img)}")


class CCIPExtractFeature:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("CCIP_MODEL",),
            },
            "optional": {
                "size": ("INT", {"default": 384}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "extract"
    CATEGORY = "CCIP"

    def extract(self, image, model, size: int = 384):
        from ccip_lib.ccip import ccip_extract_feature

        pil = _to_pil_image(image)
        emb = ccip_extract_feature(pil, size=size, model=model["model"] if isinstance(model, dict) else model)
        emb = np.asarray(emb)
        return (emb,)


class CCIPDifference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "model": ("CCIP_MODEL",),
            },
            "optional": {
                "size": ("INT", {"default": 384}),
            }
        }

    RETURN_TYPES = ("NUMBER",)
    FUNCTION = "diff"
    CATEGORY = "CCIP"

    def diff(self, image_a, image_b, model, size: int = 384):
        from ccip_lib.ccip import ccip_difference

        a = _to_pil_image(image_a)
        b = _to_pil_image(image_b)
        d = ccip_difference(a, b, size=size, model=model["model"] if isinstance(model, dict) else model)
        return (float(d),)


class CCIPSame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "model": ("CCIP_MODEL",),
            },
            "optional": {
                "use_default_threshold": ("BOOLEAN", {"default": True}),
                "threshold": ("FLOAT", {"default": 0.18}),
                "size": ("INT", {"default": 384}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "same"
    CATEGORY = "CCIP"

    def same(self, image_a, image_b, model, use_default_threshold: bool = True, threshold: float = 0.18, size: int = 384):
        from ccip_lib.ccip import ccip_same, ccip_default_threshold

        a = _to_pil_image(image_a)
        b = _to_pil_image(image_b)
        thr = None
        if use_default_threshold:
            try:
                thr = ccip_default_threshold(model["model"] if isinstance(model, dict) else model)
            except Exception:
                thr = threshold
        else:
            thr = threshold

        res = ccip_same(a, b, threshold=thr, size=size, model=model["model"] if isinstance(model, dict) else model)
        return (bool(res),)

NODE_CLASSES = [
    CCIPModelLoader,
    CCIPExtractFeature,
    CCIPDifference,
    CCIPSame,
]

NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in NODE_CLASSES}

def load_nodes():
    return NODE_CLASS_MAPPINGS

__all__ = ['NODE_CLASSES', 'NODE_CLASS_MAPPINGS', 'load_nodes']
