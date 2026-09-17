"""Every family-1 store's adapter, discovered and checked (E-082).

Each module in this directory (other than names starting with `_`, which are
shared helpers) exposes `ADAPTER`, a subclass of
`build_family10_stores.SourceAdapter`. Importing this package imports every
module in NAME ORDER and builds

    REGISTRY = {store: adapter_class}

and REFUSES — raises, so nothing downstream can run on a half-registry — when:

  * a module does not import, or has no `ADAPTER`;
  * two adapters claim the same `store`;
  * an adapter's `family`, `distribution` or `licence` is missing or not one
    of the contract's values;
  * `licence["redistribution"] == "no"` and `distribution == "public"` — the
    two-track rule: data whose terms forbid redistribution is built only onto
    the private track;
  * `first_year < 1914` with `time_dtype != "int64"` (schema 3 is required
    where int32 seconds since 1982 cannot reach).

A broken adapter therefore breaks the CLI for every store, loudly, which is
the intended trade: a registry that silently skipped a module would build a
family with a store missing and nothing to say so.
"""
import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ML = os.path.dirname(os.path.dirname(_HERE))
if _ML not in sys.path:
    sys.path.insert(0, _ML)

import build_family10_stores as f10b                            # noqa: E402

FAMILIES = {
    # code -> (family name in store.json, family_version, path slug)
    "1gf": ("family1_gf", "1.gf", "family1_gf"),
    "1tf": ("family1_tf", "1.0.tf", "family1_tf"),
    "09tf": ("family09_tf", "0.9.tf", "family09_tf"),
}
DISTRIBUTIONS = ("public", "private")
REDISTRIBUTION = ("yes", "attribution", "no")
DERIVED_WORKS = ("free", "non-commercial", "restricted")


class RegistryError(ValueError):
    """A family-1 adapter that the registry refuses to list."""


def module_names(directory=_HERE):
    """The adapter modules in `directory`, sorted; helpers (`_x.py`) skipped."""
    out = []
    for n in sorted(os.listdir(directory)):
        if n.endswith(".py") and not n.startswith("_"):
            out.append(n[:-3])
    return out


def validate(cls, where="?"):
    """Raise `RegistryError` unless `cls` satisfies the contract."""
    if not (isinstance(cls, type) and issubclass(cls, f10b.SourceAdapter)):
        raise RegistryError(f"{where}: ADAPTER is {cls!r}, not a subclass of "
                            f"build_family10_stores.SourceAdapter")
    store = getattr(cls, "store", "")
    if not store or not isinstance(store, str):
        raise RegistryError(f"{where}: ADAPTER has no `store` name")
    fam = getattr(cls, "family", None)
    if fam not in FAMILIES:
        raise RegistryError(f"{where} ({store}): family {fam!r} is not one of "
                            f"{sorted(FAMILIES)}")
    dist = getattr(cls, "distribution", None)
    if dist not in DISTRIBUTIONS:
        raise RegistryError(f"{where} ({store}): distribution {dist!r} is not "
                            f"one of {DISTRIBUTIONS}")
    lic = getattr(cls, "licence", None)
    if not isinstance(lic, dict) or not lic.get("name"):
        raise RegistryError(f"{where} ({store}): `licence` must be a dict "
                            f"with at least `name`, `redistribution` and "
                            f"`derived_works`")
    red = lic.get("redistribution")
    if red not in REDISTRIBUTION:
        raise RegistryError(f"{where} ({store}): licence.redistribution "
                            f"{red!r} is not one of {REDISTRIBUTION}")
    der = lic.get("derived_works")
    if der not in DERIVED_WORKS:
        raise RegistryError(f"{where} ({store}): licence.derived_works "
                            f"{der!r} is not one of {DERIVED_WORKS}")
    if red == "no" and dist != "private":
        raise RegistryError(
            f"{where} ({store}): licence.redistribution is 'no' and "
            f"distribution is {dist!r}. Data whose terms forbid "
            f"redistribution is built on the PRIVATE track only "
            f"(E-082 §1.3) — set distribution = 'private'.")
    try:
        f10b.time_dtype_name(cls)
    except ValueError as e:
        raise RegistryError(f"{where} ({store}): {e}") from None
    creds = getattr(cls, "credentials", ())
    if isinstance(creds, str) or not all(isinstance(c, str) for c in creds):
        raise RegistryError(f"{where} ({store}): credentials must be a tuple "
                            f"of environment variable NAMES")
    return store


def discover(package=__name__, directory=_HERE):
    """Import every adapter module in name order -> {store: class}."""
    reg = {}
    where_of = {}
    for name in module_names(directory):
        mod = importlib.import_module(f"{package}.{name}")
        cls = getattr(mod, "ADAPTER", None)
        if cls is None:
            raise RegistryError(f"{package}.{name} defines no ADAPTER — a "
                                f"helper module must be named _{name}.py")
        store = validate(cls, f"{package}.{name}")
        if store in reg:
            raise RegistryError(
                f"store {store!r} is claimed twice: by {where_of[store]} and "
                f"by {package}.{name}")
        reg[store] = cls
        where_of[store] = f"{package}.{name}"
    return reg


REGISTRY = discover()
