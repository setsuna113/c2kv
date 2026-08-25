# Guarded test bootstrap: stub heavy optional imports (datasets,
# inference.mdocdataset) that the module chain pulls in but never uses on the
# CPU test paths.  Every stub is behind a try/import guard, so on machines
# with the real dependencies installed (e.g. the NPU server env) this file is
# a no-op.  torch is NOT stubbed: torch-dependent tests must declare
# `pytest.importorskip("torch")` so they skip cleanly where torch is absent.
import importlib.machinery
import pathlib
import sys
import types


def _stub_module(name: str) -> types.ModuleType:
    """Build a stub module that survives importlib introspection.

    A bare ``types.ModuleType`` leaves ``__spec__`` as None; transformers
    probes ``datasets.__spec__`` during its lazy-import checks and raises
    ``ValueError: datasets.__spec__ is None``.  Attaching a real (loader-less)
    ModuleSpec keeps the stub indistinguishable from an ordinary module for
    that check.
    """
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


if "datasets" not in sys.modules:
    try:
        import datasets  # noqa: F401
    except ImportError:
        stub = _stub_module("datasets")
        stub.Dataset = type("Dataset", (), {})
        stub.__version__ = "0.0.0.stub"
        sys.modules["datasets"] = stub

try:
    import inference.mdocdataset  # noqa: F401
except ImportError:
    mdoc = _stub_module("inference.mdocdataset")
    mdoc.load_mdoc_dataset = None
    mdoc.QA_SYSTEM_PROMPT = "stub"
    pkg = _stub_module("inference")
    pkg.mdocdataset = mdoc
    sys.modules.setdefault("inference", pkg)
    sys.modules["inference.mdocdataset"] = mdoc

# python/models is a real package and its __init__.py imports torch eagerly.
# pytest imports a test module living there as ``models.test_*``, which runs
# that __init__ FIRST — so on a torch-free box the file ERRORs at collection
# and its module-level `pytest.importorskip("torch")` never gets to run.
# Registering a loader-less namespace shim for the package (path only, no
# __init__ execution) lets the test module import, hit its importorskip and
# SKIP.  Guarded like every other stub here: a no-op wherever torch exists.
try:
    import torch  # noqa: F401
except ImportError:
    _models_dir = pathlib.Path(__file__).resolve().parent / "python" / "models"
    if _models_dir.is_dir() and "models" not in sys.modules:
        _models_pkg = types.ModuleType("models")
        _models_pkg.__path__ = [str(_models_dir)]
        _models_spec = importlib.machinery.ModuleSpec("models", loader=None, is_package=True)
        _models_spec.submodule_search_locations = _models_pkg.__path__
        _models_pkg.__spec__ = _models_spec
        sys.modules["models"] = _models_pkg
