"""Load the pinned vendor package without generic 'models' import collisions."""
import importlib.util
import os
from pathlib import Path
import sys


def load_official():
    name = "safa_facegen.meanflow._official"
    if name in sys.modules:
        return sys.modules[name]
    default = Path(__file__).resolve().parents[3] / "vendor" / "meanflow"
    root = Path(os.environ.get("SAFA_MEANFLOW_VENDOR", str(default)))
    if not (root / "UPSTREAM.json").is_file():
        raise FileNotFoundError(f"Pinned MeanFlow vendor not found: {root}")
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_modules():
    import importlib
    root = load_official().__name__
    return (importlib.import_module(root + ".meanflow"),
            importlib.import_module(root + ".models.models_dit"))


def create_model(model_id):
    from .spec import FIXED_RECIPE, OFFICIAL_NAMES, get_spec
    get_spec(model_id)
    mf, _ = official_modules()
    model = mf.MeanFlow(model_str=OFFICIAL_NAMES[model_id], model_config={},
                       guidance_eq=FIXED_RECIPE["guidance"],
                       noise_dist=FIXED_RECIPE["time_sampling"]["distribution"],
                       class_dropout_prob=FIXED_RECIPE["class_dropout_prob"],
                       data_proportion=FIXED_RECIPE["data_proportion"],
                       P_mean=FIXED_RECIPE["time_sampling"]["mean"],
                       P_std=FIXED_RECIPE["time_sampling"]["std"],
                       norm_p=FIXED_RECIPE["norm_p"], norm_eps=FIXED_RECIPE["norm_eps"])
    if model.dtype != mf.jnp.float32 or model.num_classes != FIXED_RECIPE["null_label"]:
        raise ValueError("Official model precision/NULL index differs from the registered recipe")
    return model
