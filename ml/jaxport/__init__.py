"""Framework-independent reference implementation of this project's models.

`ml/jaxport` re-expresses the four small `nn.Module`s the programme actually
trains — PixelMAE, TemporalTransformer, SectionHead (tier 1) — in JAX/Flax
NNX, together with a one-way torch → JAX weight converter, so that a
published checkpoint can be read and run without adopting the PyTorch
tooling. It is a REFERENCE implementation, not a migration: the torch stack
under `ml/` remains the operational one, every published number is produced
by it, and nothing in the operational tree imports from this package. The
value of a second, independent implementation of the same arithmetic is that
it can be scored against the first (`ml/plans/JAX_PORT.md` §5).

NOTE ON THE PACKAGE NAME. This directory is `jaxport`, deliberately NOT
`jax`. `ml/` is placed on `sys.path` by nearly every script in this repo
(`sys.path.insert(0, dirname(abspath(__file__)))`), so a directory named
`ml/jax/` would shadow the installed `jax` library for the whole process and
break `import jax` everywhere — including inside this package. Do not rename
it.
"""
