"""PepDyn GCN baselines for LMDB protein-peptide dynamics data."""

from pathlib import Path
import os


# Work around MKL/OpenMP conflicts seen in some conda environments.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

# Keep matplotlib cache on a writable path for non-interactive jobs.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-pepdyn"))
