"""
Unified rCM training module for LTX distillation.

This module currently reuses the existing LTX2DMD implementation as the
training core, because the codebase historically concentrated the full Stage-1
distillation stack in `dmd.py`. With SCM enabled in config, this alias now
represents the full rCM training path (SCM + DMD); with SCM disabled, it
represents the DMD-only baseline.
"""

from ltx_distillation.dmd import LTX2DMD


class LTX2RCM(LTX2DMD):
    """Semantic alias for the full rCM training core on top of LTX."""

    pass
