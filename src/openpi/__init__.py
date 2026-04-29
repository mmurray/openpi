# JAX 0.8 compatibility patch for numpy 1.26.
#
# Isaac Sim 5.1 pins numpy==1.26.0 (its C extensions are ABI-bound to it).
# JAX 0.8 declares numpy>=2.0 but jaxlib 0.8 is actually binary-compatible
# with numpy 1.26 — the only hard incompatibility is one Python-level call
# in `jax._src.literals`:
#
#     return np.asarray(self.val, dtype=dtype, copy=copy)
#
# numpy 1.26's `np.asarray` doesn't accept `copy` (added in numpy 2.0).
# This patch replaces `TypedNdArray.__array__` to avoid that kwarg so that
# JAX 0.8, numpy 1.26, and Isaac Sim can coexist in one env.
#
# The patch is gated: it is only applied when the installed numpy actually
# lacks the `copy` kwarg on `np.asarray`, so users on numpy >= 2.0 are
# unaffected.
def _maybe_apply_jax_numpy_compat_patch():
    import numpy as _np

    # Probe: does np.asarray accept copy= ?
    try:
        _np.asarray(_np.uint32(0), copy=True)
        return  # numpy >= 2.0 path works fine — no patch needed
    except TypeError:
        pass  # numpy 1.x or numpy 2.x with the scalar-ABI bug → patch

    try:
        from jax._src import literals
    except ImportError:
        return  # JAX not installed or internal layout changed → no-op

    if not hasattr(literals, "TypedNdArray"):
        return  # patch target moved in a future JAX release → no-op

    def _patched_array(self, dtype=None, copy=None):
        arr = _np.asarray(self.val, dtype=dtype)
        if copy is True:
            arr = arr.copy()
        return arr

    literals.TypedNdArray.__array__ = _patched_array


_maybe_apply_jax_numpy_compat_patch()
del _maybe_apply_jax_numpy_compat_patch
