Upstream: Gsunshine/meanflow, commit d70cb55d298ee03c53bf6da67bec281082e4e2d9.
Original file SHA256 values are in UPSTREAM.json. LICENSE is unchanged (MIT).

Local changes:
* Python package-relative imports in meanflow.py and models/models_dit.py.
* Positional jnp.clip bounds in meanflow.py for JAX >= 0.6 compatibility.
* Package __init__.py files.

The network, objective, JVP, adaptive weighting and original initialization are
unchanged. Training orchestration, data access, checkpoint serialization and the
PyTorch inference port are in src/safa_facegen/meanflow, not mixed into upstream.
