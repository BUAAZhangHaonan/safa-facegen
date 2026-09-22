"""Import the pinned, repository-local upstream modules without global RF names."""
from pathlib import Path
import importlib
import importlib.util
import sys
import types

ROOT = Path(__file__).resolve().parents[3]


def module(alias: str, relative: str, name: str):
    if alias not in sys.modules:
        package = types.ModuleType(alias)
        package.__path__ = [str(ROOT / relative)]
        sys.modules[alias] = package
    return importlib.import_module(f"{alias}.{name}")


def ldm_imports():
    path = str(ROOT / "vendor/latent_diffusion")
    if path not in sys.path:
        sys.path.insert(0, path)


def rf_config():
    return module("_safa_rf", "vendor/rectified_flow/ImageGeneration",
                  "configs.rectified_flow.celeba_hq_pytorch_rf_gaussian").get_config()


def rf_backbone():
    config = rf_config()
    return module("_safa_rf", "vendor/rectified_flow/ImageGeneration", "models.ncsnpp").NCSNpp(config)


def lcd_math():
    return module("_safa_lcm", "vendor/latent_consistency", "lcd_math")
