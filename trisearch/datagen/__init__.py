"""Optional dataset generation (requires SageMath with the TOPCOM engine).

Pipeline: sample point clouds → convex hull → keep simplicial polytopes that are
not combinatorially isomorphic to one already kept → enumerate triangulations
with TOPCOM → Hugging Face dataset rows.
"""


def require_sage():
    try:
        import sage.all  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Data generation needs SageMath with TOPCOM (e.g. `conda install -c conda-forge sage topcom`). "
            "Training and evaluation do not."
        ) from exc
