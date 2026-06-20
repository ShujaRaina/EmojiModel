from . import dit
try:
  # dimamba depends on CUDA-only kernels (mamba_ssm / causal_conv1d);
  # skip it gracefully on CPU-only installs.
  from . import dimamba
except ImportError:
  dimamba = None
from . import ema
from . import autoregressive
