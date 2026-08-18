"""License and provenance helpers for PyTDC-hosted assets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class AssetLicense:
    """License metadata for a dataset, model, or resource exposed by PyTDC."""

    name: str
    asset_type: str
    license: str
    source_url: Optional[str] = None
    allowed_uses: Tuple[str, ...] = field(default_factory=tuple)
    restrictions: Tuple[str, ...] = field(default_factory=tuple)
    citation: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable view of the license metadata."""
        return asdict(self)


UNKNOWN_LICENSE = AssetLicense(
    name="unknown",
    asset_type="unknown",
    license="unknown",
    restrictions=(
        "PyTDC does not have recorded license metadata for this asset.",
        "Check the upstream source before redistribution, commercial use, or model training.",
    ),
)

_LICENSES: Dict[str, AssetLicense] = {}


def _normalize(name: str) -> str:
    return name.casefold().replace("_", "-").strip()


def register_asset_license(
    info: AssetLicense, aliases: Iterable[str] = ()) -> None:
    """Register license metadata under its name and optional aliases."""
    keys = (info.name, *aliases)
    for key in keys:
        _LICENSES[_normalize(key)] = info


def get_asset_license(name: str, default_unknown: bool = True) -> AssetLicense:
    """Return license metadata for an asset name.

    If ``default_unknown`` is true, unknown assets return conservative metadata
    that tells callers to consult upstream terms. Otherwise, unknown assets raise
    ``KeyError``.
    """
    key = _normalize(name)
    if key in _LICENSES:
        return _LICENSES[key]
    if default_unknown:
        return AssetLicense(
            name=name,
            asset_type=UNKNOWN_LICENSE.asset_type,
            license=UNKNOWN_LICENSE.license,
            restrictions=UNKNOWN_LICENSE.restrictions,
        )
    raise KeyError("No license metadata recorded for {}".format(name))


def retrieve_license_info(name: str,
                          default_unknown: bool = True) -> Dict[str, object]:
    """Return a dictionary of license and provenance metadata for an asset."""
    return get_asset_license(name, default_unknown).to_dict()


register_asset_license(
    AssetLicense(
        name="cellxgene-census",
        asset_type="resource",
        license="MIT",
        source_url="https://github.com/chanzuckerberg/cellxgene-census",
        allowed_uses=("research", "commercial", "redistribution"),
        citation="CELLxGENE Discover Census",
        notes=
        ("This entry describes the client package license. Users should also "
         "review CELLxGENE dataset-level terms for the specific data release "
         "they query."),
    ),
    aliases=("cellxgene_census", "cellxgene"),
)
