"""Context-safe data utilities for E3-aware ProMeta experiments.

The historical data path keyed protein sequences and graphs by compound ID.
That is unsafe because one compound may occur in more than one target context.
This module treats ``(Compound ID, Target, canonical E3)`` as the biological
identity of a row and gives every such context a deterministic row ID.

The module is intentionally independent of the current training code.  It can
therefore be introduced alongside the released pipeline and used to construct
paired manifests before a new E3-aware model is wired in.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data


AmbiguityPolicy = Literal["raise", "drop"]
MissingPolicy = Literal["allow", "raise", "drop"]
MultipleMatchPolicy = Literal["union", "first", "raise"]
MissingKeyPolicy = Literal["drop", "raise"]


class DataValidationError(ValueError):
    """Base class for invalid or internally inconsistent input data."""


class AmbiguousContextError(DataValidationError):
    """Raised when one canonical context resolves to conflicting values."""


class MissingContextError(DataValidationError):
    """Raised when required context-linked data are absent."""


class SplitLeakageError(DataValidationError):
    """Raised when a strict source/target split shares a forbidden identity."""


class InvalidMoleculeError(DataValidationError):
    """Raised when RDKit cannot parse a required molecular structure."""


class AmbiguousSubstructureError(DataValidationError):
    """Raised when a component has multiple matches under a strict policy."""


@dataclass(frozen=True)
class ContextColumns:
    """Column names used by the project data tables."""

    compound_id: str = "Compound ID"
    target: str = "Target"
    e3: str = "E3 ligase"
    label: str = "Label_bin"
    smiles: str = "Smiles"
    poi_sequence: str = "POI_seq"
    e3_sequence: str = "E3_seq"
    warhead_smiles: str = "warhead smiles"
    e3_ligand_smiles: str = "e3 ligand smiles"
    linker_smiles: str = "linker smiles"
    e3_uniprot: str = "E3 ligase Uniprot"


DEFAULT_COLUMNS = ContextColumns()


# Alias matching is case-insensitive.  Unknown names are upper-cased so that
# harmless capitalization differences cannot create distinct E3 identities.
DEFAULT_E3_ALIASES: dict[str, str] = {
    "crbn": "CRBN",
    "vhl": "VHL",
    "keap1": "KEAP1",
    "iap": "cIAP1",
    "ciap1": "cIAP1",
}


@dataclass(frozen=True, order=True)
class ContextKey:
    """Canonical biological identity of one PROTAC assay context."""

    compound_id: str
    target: str
    e3: str

    def as_tuple(self) -> tuple[str, str, str]:
        return self.compound_id, self.target, self.e3


@dataclass(frozen=True)
class ContextSequences:
    poi_sequence: str | None
    e3_sequence: str | None


@dataclass(frozen=True)
class ContextComponents:
    warhead_smiles: str | None
    e3_ligand_smiles: str | None
    linker_smiles: str | None
    e3_uniprot: str | None


@dataclass(frozen=True)
class ContextIndexAudit:
    """Row accounting for a context-linked auxiliary table."""

    total_rows: int
    valid_key_rows: int
    indexed_contexts: int
    dropped_missing_key_rows: int
    dropped_ambiguous_contexts: int


@dataclass(frozen=True)
class E3LigandMatchAudit:
    """Result of matching an E3-ligand component onto the full PROTAC."""

    status: str
    n_matches: int
    query_atom_count: int
    masked_atom_count: int
    detail: str = ""

    @property
    def matched(self) -> bool:
        return self.status.startswith("matched_")


@dataclass(frozen=True)
class ContextGraphIndex:
    """Graphs keyed by stable context row ID plus an explicit audit table."""

    graphs: dict[str, Data]
    audit: pd.DataFrame


@dataclass(frozen=True)
class CrossE3Split:
    """A strict source-E3 training frame and target-E3 evaluation frame."""

    source_e3: str
    target_e3: str
    source: pd.DataFrame
    target: pd.DataFrame
    split_id: str


@dataclass(frozen=True)
class Episode:
    """One materialized support/query episode selected by a manifest."""

    episode_id: str
    target: str
    e3: str
    support: pd.DataFrame
    query: pd.DataFrame

    @property
    def support_row_ids(self) -> tuple[str, ...]:
        return tuple(self.support["row_id"].astype(str))

    @property
    def query_row_ids(self) -> tuple[str, ...]:
        return tuple(self.query["row_id"].astype(str))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _required_text(value: Any, field: str) -> str:
    if _is_missing(value):
        raise DataValidationError(f"{field} is missing")
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text or text.casefold() in {"nan", "none", "null"}:
        raise DataValidationError(f"{field} is missing")
    return text


def _optional_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    return text


def normalize_compound_id(value: Any) -> str:
    """Normalize IDs without turning integer-valued floats into ``"554.0"``."""

    if _is_missing(value):
        raise DataValidationError("Compound ID is missing")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _required_text(value, "Compound ID")


def normalize_target(value: Any) -> str:
    return _required_text(value, "Target")


def canonicalize_e3(
    value: Any,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Return a deterministic E3 name, including the project aliases.

    ``KEAP1``/``Keap1`` become ``KEAP1`` and ``IAP``/``cIAP1`` become
    ``cIAP1``.  Additional aliases may be provided using any capitalization.
    """

    text = _required_text(value, "E3 ligase")
    lookup = dict(DEFAULT_E3_ALIASES)
    if aliases:
        for alias, canonical in aliases.items():
            lookup[_required_text(alias, "E3 alias").casefold()] = _required_text(
                canonical, "canonical E3"
            )
    return lookup.get(text.casefold(), text.upper())


def normalize_protein_sequence(value: Any) -> str | None:
    """Normalize sequence case/whitespace while retaining uncommon residues."""

    text = _optional_text(value)
    if text is None:
        return None
    return re.sub(r"\s+", "", text).upper()


def canonicalize_smiles(value: Any) -> str:
    """Canonicalize an isomeric SMILES string for strict leakage checks."""

    text = _required_text(value, "Smiles")
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise InvalidMoleculeError(f"RDKit cannot parse Smiles={text!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def context_key(
    compound_id: Any,
    target: Any,
    e3: Any,
    aliases: Mapping[str, str] | None = None,
) -> ContextKey:
    return ContextKey(
        normalize_compound_id(compound_id),
        normalize_target(target),
        canonicalize_e3(e3, aliases),
    )


def stable_context_id(key: ContextKey) -> str:
    """Create an order-independent stable ID from the full canonical context."""

    payload = json.dumps(key.as_tuple(), ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"ctx_{digest}"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def _key_frame(
    frame: pd.DataFrame,
    columns: ContextColumns,
    aliases: Mapping[str, str] | None,
) -> pd.DataFrame:
    _require_columns(frame, (columns.compound_id, columns.target, columns.e3))
    out = frame.copy()
    out[columns.compound_id] = out[columns.compound_id].map(normalize_compound_id)
    out[columns.target] = out[columns.target].map(normalize_target)
    out["canonical_e3"] = out[columns.e3].map(
        lambda value: canonicalize_e3(value, aliases)
    )
    out["context_key"] = [
        ContextKey(compound_id, target, e3)
        for compound_id, target, e3 in zip(
            out[columns.compound_id], out[columns.target], out["canonical_e3"]
        )
    ]
    out["row_id"] = out["context_key"].map(stable_context_id)
    return out


def _auxiliary_key_frame(
    frame: pd.DataFrame,
    columns: ContextColumns,
    aliases: Mapping[str, str] | None,
    missing_key: MissingKeyPolicy,
) -> tuple[pd.DataFrame, int]:
    """Normalize auxiliary-table keys with auditable incomplete-row handling."""

    if missing_key not in {"drop", "raise"}:
        raise ValueError(f"Unknown missing-key policy: {missing_key}")
    _require_columns(frame, (columns.compound_id, columns.target, columns.e3))
    valid_positions: list[int] = []
    keys: list[ContextKey] = []
    for position, (source_index, row) in enumerate(frame.iterrows()):
        try:
            key = context_key(
                row[columns.compound_id],
                row[columns.target],
                row[columns.e3],
                aliases,
            )
        except DataValidationError as error:
            if missing_key == "raise":
                raise DataValidationError(
                    f"Invalid context key in auxiliary row {source_index!r}: {error}"
                ) from error
            continue
        valid_positions.append(position)
        keys.append(key)
    out = frame.iloc[valid_positions].copy()
    out[columns.compound_id] = [key.compound_id for key in keys]
    out[columns.target] = [key.target for key in keys]
    out["canonical_e3"] = [key.e3 for key in keys]
    out["context_key"] = keys
    out["row_id"] = [stable_context_id(key) for key in keys]
    return out, len(frame) - len(valid_positions)


def prepare_contexts(
    frame: pd.DataFrame,
    *,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
    require_label: bool = True,
    require_unique: bool = True,
) -> pd.DataFrame:
    """Normalize a context table and attach canonical keys and stable row IDs.

    A context duplicate is rejected by default.  This prevents an assay row
    from appearing in both support and query under an indistinguishable ID.
    """

    out = _key_frame(frame, columns, aliases)
    if require_label:
        _require_columns(out, (columns.label,))
    if columns.label in out.columns:
        numeric = pd.to_numeric(out[columns.label], errors="raise")
        if numeric.isna().any() or not numeric.isin((0, 1)).all():
            raise DataValidationError(f"{columns.label} must contain only 0/1")
        out[columns.label] = numeric.astype(int)
    if require_unique:
        duplicated = out["context_key"].duplicated(keep=False)
        if duplicated.any():
            examples = sorted(
                {key.as_tuple() for key in out.loc[duplicated, "context_key"]}
            )[:5]
            raise AmbiguousContextError(
                f"Duplicate canonical contexts are not valid assay row IDs; "
                f"examples={examples}"
            )
    # Repeated IDs are expected when callers explicitly allow duplicate
    # contexts.  A real collision means one digest represents *different*
    # canonical keys, which remains an error under either policy.
    distinct_id_key_pairs = out[["row_id", "context_key"]].drop_duplicates()
    if distinct_id_key_pairs["row_id"].duplicated().any():
        raise DataValidationError("Stable context ID collision detected")
    return out.reset_index(drop=True)


def _unique_values(
    values: Iterable[Any],
    normalizer: Any,
) -> list[str]:
    normalized = []
    for value in values:
        item = normalizer(value)
        if item is not None and item not in normalized:
            normalized.append(item)
    return normalized


def build_context_sequence_index(
    sequence_table: pd.DataFrame,
    *,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
    on_ambiguous: AmbiguityPolicy = "raise",
    missing_key: MissingKeyPolicy = "drop",
    return_audit: bool = False,
) -> (
    dict[ContextKey, ContextSequences]
    | tuple[dict[ContextKey, ContextSequences], ContextIndexAudit]
):
    """Resolve POI/E3 sequences by full context, never by compound alone."""

    if on_ambiguous not in {"raise", "drop"}:
        raise ValueError(f"Unknown ambiguity policy: {on_ambiguous}")
    _require_columns(
        sequence_table,
        (
            columns.compound_id,
            columns.target,
            columns.e3,
            columns.poi_sequence,
            columns.e3_sequence,
        ),
    )
    keyed, dropped_missing = _auxiliary_key_frame(
        sequence_table, columns, aliases, missing_key
    )
    index: dict[ContextKey, ContextSequences] = {}
    dropped_ambiguous = 0
    for key, group in keyed.groupby("context_key", sort=False):
        poi_values = _unique_values(group[columns.poi_sequence], normalize_protein_sequence)
        e3_values = _unique_values(group[columns.e3_sequence], normalize_protein_sequence)
        if len(poi_values) > 1 or len(e3_values) > 1:
            message = (
                f"Conflicting sequences for context {key.as_tuple()}: "
                f"n_poi={len(poi_values)}, n_e3={len(e3_values)}"
            )
            if on_ambiguous == "raise":
                raise AmbiguousContextError(message)
            dropped_ambiguous += 1
            continue
        index[key] = ContextSequences(
            poi_values[0] if poi_values else None,
            e3_values[0] if e3_values else None,
        )
    audit = ContextIndexAudit(
        total_rows=len(sequence_table),
        valid_key_rows=len(keyed),
        indexed_contexts=len(index),
        dropped_missing_key_rows=dropped_missing,
        dropped_ambiguous_contexts=dropped_ambiguous,
    )
    return (index, audit) if return_audit else index


def attach_context_sequences(
    contexts: pd.DataFrame,
    sequence_index: Mapping[ContextKey, ContextSequences],
    *,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
    missing: MissingPolicy = "allow",
) -> pd.DataFrame:
    """Attach exact-context sequences and explicit missingness indicators."""

    if missing not in {"allow", "raise", "drop"}:
        raise ValueError(f"Unknown missing policy: {missing}")
    out = prepare_contexts(contexts, columns=columns, aliases=aliases)
    poi_values: list[str | None] = []
    e3_values: list[str | None] = []
    index_missing: list[bool] = []
    for key in out["context_key"]:
        resolved = sequence_index.get(key)
        index_missing.append(resolved is None)
        poi_values.append(None if resolved is None else resolved.poi_sequence)
        e3_values.append(None if resolved is None else resolved.e3_sequence)
    out[columns.poi_sequence] = poi_values
    out[columns.e3_sequence] = e3_values
    out["sequence_context_missing"] = index_missing
    out["poi_sequence_missing"] = out[columns.poi_sequence].isna()
    out["e3_sequence_missing"] = out[columns.e3_sequence].isna()
    bad = (
        out["sequence_context_missing"]
        | out["poi_sequence_missing"]
        | out["e3_sequence_missing"]
    )
    if bad.any() and missing == "raise":
        examples = out.loc[bad, "context_key"].map(ContextKey.as_tuple).tolist()[:5]
        raise MissingContextError(
            f"Missing POI/E3 sequence data for {int(bad.sum())} contexts; "
            f"examples={examples}"
        )
    if missing == "drop":
        out = out.loc[~bad].copy()
    return out.reset_index(drop=True)


def _component_value(value: Any) -> str | None:
    return _optional_text(value)


def build_context_component_index(
    component_table: pd.DataFrame,
    *,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
    on_ambiguous: AmbiguityPolicy = "raise",
    missing_key: MissingKeyPolicy = "drop",
    return_audit: bool = False,
) -> (
    dict[ContextKey, ContextComponents]
    | tuple[dict[ContextKey, ContextComponents], ContextIndexAudit]
):
    """Resolve optional warhead/E3-ligand/linker fields by exact context."""

    if on_ambiguous not in {"raise", "drop"}:
        raise ValueError(f"Unknown ambiguity policy: {on_ambiguous}")
    _require_columns(component_table, (columns.compound_id, columns.target, columns.e3))
    keyed, dropped_missing = _auxiliary_key_frame(
        component_table, columns, aliases, missing_key
    )
    component_columns = (
        columns.warhead_smiles,
        columns.e3_ligand_smiles,
        columns.linker_smiles,
        columns.e3_uniprot,
    )
    index: dict[ContextKey, ContextComponents] = {}
    dropped_ambiguous = 0
    for key, group in keyed.groupby("context_key", sort=False):
        resolved: list[str | None] = []
        ambiguous = False
        for column in component_columns:
            values = (
                _unique_values(group[column], _component_value)
                if column in group.columns
                else []
            )
            if len(values) > 1:
                ambiguous = True
                if on_ambiguous == "raise":
                    raise AmbiguousContextError(
                        f"Conflicting {column!r} values for context {key.as_tuple()}"
                    )
            resolved.append(values[0] if values else None)
        if ambiguous and on_ambiguous == "drop":
            dropped_ambiguous += 1
            continue
        index[key] = ContextComponents(*resolved)
    audit = ContextIndexAudit(
        total_rows=len(component_table),
        valid_key_rows=len(keyed),
        indexed_contexts=len(index),
        dropped_missing_key_rows=dropped_missing,
        dropped_ambiguous_contexts=dropped_ambiguous,
    )
    return (index, audit) if return_audit else index


def attach_context_components(
    contexts: pd.DataFrame,
    component_index: Mapping[ContextKey, ContextComponents],
    *,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Attach optional component fields without changing biological row IDs."""

    out = prepare_contexts(contexts, columns=columns, aliases=aliases)
    fields = {
        columns.warhead_smiles: "warhead_smiles",
        columns.e3_ligand_smiles: "e3_ligand_smiles",
        columns.linker_smiles: "linker_smiles",
        columns.e3_uniprot: "e3_uniprot",
    }
    for column, attribute in fields.items():
        out[column] = [
            None
            if component_index.get(key) is None
            else getattr(component_index[key], attribute)
            for key in out["context_key"]
        ]
    out["component_context_missing"] = [
        key not in component_index for key in out["context_key"]
    ]
    return out


def assemble_context_table(
    contexts: pd.DataFrame,
    *,
    sequence_table: pd.DataFrame | None = None,
    component_table: pd.DataFrame | None = None,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
    missing_sequences: MissingPolicy = "allow",
) -> pd.DataFrame:
    """Build one context-safe table from assay, sequence, and component rows."""

    out = prepare_contexts(contexts, columns=columns, aliases=aliases)
    if sequence_table is not None:
        sequence_index = build_context_sequence_index(
            sequence_table, columns=columns, aliases=aliases
        )
        out = attach_context_sequences(
            out,
            sequence_index,
            columns=columns,
            aliases=aliases,
            missing=missing_sequences,
        )
    if component_table is not None:
        component_index = build_context_component_index(
            component_table, columns=columns, aliases=aliases
        )
        out = attach_context_components(
            out, component_index, columns=columns, aliases=aliases
        )
    return out


ATOM_NUMBERS = (6, 7, 8, 9, 15, 16, 17, 35, 53)
FORMAL_CHARGES = (-2, -1, 0, 1, 2)
CHIRAL_TAGS = (
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
)
BOND_TYPES = (
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
)
BOND_STEREOS = (
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
)

ATOM_FEATURE_NAMES = tuple(
    [f"element_{number}" for number in ATOM_NUMBERS]
    + ["element_other"]
    + [f"formal_charge_{charge}" for charge in FORMAL_CHARGES]
    + ["formal_charge_other", "aromatic"]
    + ["chiral_unspecified", "chiral_cw", "chiral_ccw", "chiral_other"]
)
BOND_FEATURE_NAMES = tuple(
    ["bond_single", "bond_double", "bond_triple", "bond_aromatic", "bond_other"]
    + ["conjugated", "in_ring"]
    + [
        "stereo_none",
        "stereo_any",
        "stereo_z",
        "stereo_e",
        "stereo_cis",
        "stereo_trans",
        "stereo_other",
    ]
)


def _one_hot_with_other(value: Any, choices: Sequence[Any]) -> list[float]:
    return [float(value == choice) for choice in choices] + [
        float(value not in choices)
    ]


def atom_features(atom: rdchem.Atom) -> list[float]:
    """Element, formal-charge, aromaticity, and chirality features."""

    return (
        _one_hot_with_other(atom.GetAtomicNum(), ATOM_NUMBERS)
        + _one_hot_with_other(atom.GetFormalCharge(), FORMAL_CHARGES)
        + [float(atom.GetIsAromatic())]
        + _one_hot_with_other(atom.GetChiralTag(), CHIRAL_TAGS)
    )


def bond_features(bond: rdchem.Bond) -> list[float]:
    """Bond order, conjugation, ring, and stereo features."""

    return (
        _one_hot_with_other(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot_with_other(bond.GetStereo(), BOND_STEREOS)
    )


def match_e3_ligand(
    molecule: rdchem.Mol,
    e3_ligand_smiles: Any,
    *,
    use_chirality: bool = True,
    multiple_match_policy: MultipleMatchPolicy = "union",
) -> tuple[torch.BoolTensor, E3LigandMatchAudit]:
    """Match an E3-ligand component and return both mask and audit status.

    Failed or absent matches produce an all-false mask *and* a non-success
    status.  Multiple matches are never hidden: the default unions all matched
    atoms and records ``matched_multiple_union``.
    """

    if multiple_match_policy not in {"union", "first", "raise"}:
        raise ValueError(f"Unknown multiple-match policy: {multiple_match_policy}")
    mask = torch.zeros(molecule.GetNumAtoms(), dtype=torch.bool)
    component = _optional_text(e3_ligand_smiles)
    if component is None:
        return mask, E3LigandMatchAudit("missing_component", 0, 0, 0)
    query = Chem.MolFromSmiles(component)
    if query is None or query.GetNumAtoms() == 0:
        return mask, E3LigandMatchAudit(
            "invalid_component", 0, 0, 0, f"Cannot parse {component!r}"
        )
    matches = molecule.GetSubstructMatches(
        query, uniquify=True, useChirality=use_chirality
    )
    if not matches:
        return mask, E3LigandMatchAudit(
            "no_match", 0, query.GetNumAtoms(), 0
        )
    if len(matches) > 1 and multiple_match_policy == "raise":
        raise AmbiguousSubstructureError(
            f"E3 ligand has {len(matches)} matches in the full molecule"
        )
    selected = matches[:1] if multiple_match_policy == "first" else matches
    for match in selected:
        mask[list(match)] = True
    if len(matches) == 1:
        status = "matched_unique"
    else:
        status = f"matched_multiple_{multiple_match_policy}"
    return mask, E3LigandMatchAudit(
        status,
        len(matches),
        query.GetNumAtoms(),
        int(mask.sum().item()),
    )


def molecule_to_graph(
    molecule: rdchem.Mol,
    *,
    e3_ligand_smiles: Any = None,
    use_chirality: bool = True,
    multiple_match_policy: MultipleMatchPolicy = "union",
) -> tuple[Data, E3LigandMatchAudit]:
    """Convert an RDKit molecule to an edge-aware PyG graph."""

    if molecule is None or molecule.GetNumAtoms() == 0:
        raise InvalidMoleculeError("Molecule is empty")
    nodes = torch.tensor(
        [atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.float32
    )
    edge_pairs: list[list[int]] = []
    edge_values: list[list[float]] = []
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_pairs.extend(([begin, end], [end, begin]))
        edge_values.extend((features, features))
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_values, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_FEATURE_NAMES)), dtype=torch.float32)
    ligand_mask, audit = match_e3_ligand(
        molecule,
        e3_ligand_smiles,
        use_chirality=use_chirality,
        multiple_match_policy=multiple_match_policy,
    )
    graph = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)
    graph.e3_ligand_mask = ligand_mask
    graph.e3_ligand_match_status = audit.status
    graph.e3_ligand_match_ok = audit.matched
    graph.e3_ligand_match_count = audit.n_matches
    return graph, audit


def smiles_to_graph(
    smiles: Any,
    *,
    e3_ligand_smiles: Any = None,
    use_chirality: bool = True,
    multiple_match_policy: MultipleMatchPolicy = "union",
) -> tuple[Data, E3LigandMatchAudit]:
    text = _required_text(smiles, "Smiles")
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise InvalidMoleculeError(f"RDKit cannot parse Smiles={text!r}")
    graph, audit = molecule_to_graph(
        molecule,
        e3_ligand_smiles=e3_ligand_smiles,
        use_chirality=use_chirality,
        multiple_match_policy=multiple_match_policy,
    )
    graph.smiles = text
    return graph, audit


def build_context_graph_index(
    contexts: pd.DataFrame,
    *,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
    invalid_molecule: Literal["raise", "drop"] = "raise",
    require_e3_ligand_match: bool = False,
    multiple_match_policy: MultipleMatchPolicy = "union",
) -> ContextGraphIndex:
    """Build graphs keyed by context row ID and return a match audit table."""

    if invalid_molecule not in {"raise", "drop"}:
        raise ValueError(f"Unknown invalid-molecule policy: {invalid_molecule}")
    _require_columns(contexts, (columns.smiles,))
    frame = prepare_contexts(contexts, columns=columns, aliases=aliases)
    graphs: dict[str, Data] = {}
    audit_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        ligand = row.get(columns.e3_ligand_smiles)
        try:
            graph, audit = smiles_to_graph(
                row[columns.smiles],
                e3_ligand_smiles=ligand,
                multiple_match_policy=multiple_match_policy,
            )
            molecule_status = "parsed"
        except InvalidMoleculeError as error:
            if invalid_molecule == "raise":
                raise
            audit_rows.append(
                {
                    "row_id": row["row_id"],
                    "Compound ID": row[columns.compound_id],
                    "Target": row[columns.target],
                    "canonical_e3": row["canonical_e3"],
                    "molecule_status": "invalid",
                    "e3_ligand_match_status": "not_attempted",
                    "e3_ligand_match_count": 0,
                    "e3_ligand_masked_atoms": 0,
                    "detail": str(error),
                }
            )
            continue
        if require_e3_ligand_match and not audit.matched:
            raise MissingContextError(
                f"E3 ligand match failed for {row['row_id']}: {audit.status}"
            )
        graph.row_id = str(row["row_id"])
        graph.cid = str(row[columns.compound_id])
        graph.target = str(row[columns.target])
        graph.e3 = str(row["canonical_e3"])
        graph.y = torch.tensor([int(row[columns.label])], dtype=torch.long)
        for output_name, column in (
            ("warhead_smiles", columns.warhead_smiles),
            ("e3_ligand_smiles", columns.e3_ligand_smiles),
            ("linker_smiles", columns.linker_smiles),
            ("e3_uniprot", columns.e3_uniprot),
        ):
            value = row.get(column)
            setattr(graph, output_name, _optional_text(value))
        graphs[graph.row_id] = graph
        audit_rows.append(
            {
                "row_id": graph.row_id,
                "Compound ID": graph.cid,
                "Target": graph.target,
                "canonical_e3": graph.e3,
                "molecule_status": molecule_status,
                "e3_ligand_match_status": audit.status,
                "e3_ligand_match_count": audit.n_matches,
                "e3_ligand_masked_atoms": audit.masked_atom_count,
                "detail": audit.detail,
            }
        )
    return ContextGraphIndex(graphs, pd.DataFrame(audit_rows))


def _canonical_smiles_values(values: pd.Series) -> set[str]:
    return {
        canonicalize_smiles(value)
        for value in values
        if _optional_text(value) is not None
    }


def split_source_target_e3(
    contexts: pd.DataFrame,
    source_e3: Any,
    target_e3: Any,
    *,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
    require_disjoint_compounds: bool = True,
    require_disjoint_smiles: bool = True,
) -> CrossE3Split:
    """Create a strict source/target-E3 split with leakage checks."""

    source_name = canonicalize_e3(source_e3, aliases)
    target_name = canonicalize_e3(target_e3, aliases)
    if source_name == target_name:
        raise DataValidationError("Source and target E3 must differ")
    frame = prepare_contexts(contexts, columns=columns, aliases=aliases)
    source = frame.loc[frame["canonical_e3"].eq(source_name)].copy()
    target = frame.loc[frame["canonical_e3"].eq(target_name)].copy()
    if source.empty or target.empty:
        raise MissingContextError(
            f"Empty source/target partition: source={len(source)}, target={len(target)}"
        )
    if require_disjoint_compounds:
        overlap = set(source[columns.compound_id]) & set(target[columns.compound_id])
        if overlap:
            raise SplitLeakageError(
                f"Source and target E3 share {len(overlap)} Compound IDs; "
                f"examples={sorted(overlap)[:5]}"
            )
    if require_disjoint_smiles and columns.smiles in frame.columns:
        overlap = _canonical_smiles_values(source[columns.smiles]) & _canonical_smiles_values(
            target[columns.smiles]
        )
        if overlap:
            raise SplitLeakageError(
                f"Source and target E3 share {len(overlap)} canonical molecules"
            )
    split_payload = {
        "source_e3": source_name,
        "target_e3": target_name,
        "source_rows": sorted(source["row_id"]),
        "target_rows": sorted(target["row_id"]),
    }
    split_digest = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return CrossE3Split(
        source_name,
        target_name,
        source.reset_index(drop=True),
        target.reset_index(drop=True),
        f"split_{split_digest}",
    )


MANIFEST_COLUMNS = (
    "manifest_id",
    "episode_id",
    "seed",
    "canonical_e3",
    "Target",
    "role",
    "label",
    "sample_order",
    "row_id",
    "query_row_id",
)


def _derived_seed(*parts: Any) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def build_episode_manifest(
    target_contexts: pd.DataFrame,
    *,
    seed: int,
    k_shot: int = 2,
    q_query: int = 3,
    episodes_per_target: int = 1,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
    on_ineligible: Literal["skip", "raise"] = "skip",
) -> pd.DataFrame:
    """Sample class-balanced Target episodes with stable, reusable row IDs.

    Randomness is derived separately for every target/class/episode from stable
    names.  Consequently input row order, graph construction, or an E3
    intervention cannot alter which rows are selected.
    """

    if k_shot < 1 or q_query < 1 or episodes_per_target < 1:
        raise ValueError("k_shot, q_query, and episodes_per_target must be positive")
    if on_ineligible not in {"skip", "raise"}:
        raise ValueError(f"Unknown ineligible policy: {on_ineligible}")
    frame = prepare_contexts(target_contexts, columns=columns, aliases=aliases)
    e3_values = sorted(frame["canonical_e3"].unique())
    if len(e3_values) != 1:
        raise DataValidationError(
            f"An episodic target frame must contain exactly one E3, got {e3_values}"
        )
    e3 = e3_values[0]
    rows: list[dict[str, Any]] = []
    required = k_shot + q_query
    for target, group in frame.groupby(columns.target, sort=True):
        by_label = {
            label: sorted(group.loc[group[columns.label].eq(label), "row_id"])
            for label in (1, 0)
        }
        if any(len(row_ids) < required for row_ids in by_label.values()):
            if on_ineligible == "raise":
                counts = {label: len(ids) for label, ids in by_label.items()}
                raise MissingContextError(
                    f"Target {target!r} is not K{k_shot}Q{q_query} eligible: {counts}"
                )
            continue
        for episode_index in range(episodes_per_target):
            episode_payload = (e3, str(target), int(seed), k_shot, q_query, episode_index)
            episode_id = f"ep_{hashlib.sha256(json.dumps(episode_payload).encode()).hexdigest()[:20]}"
            role_orders = {"support": 0, "query": 0}
            for label in (1, 0):
                selected = list(by_label[label])
                rng = random.Random(
                    _derived_seed(seed, e3, str(target), episode_index, label)
                )
                rng.shuffle(selected)
                support_ids = selected[:k_shot]
                query_ids = selected[k_shot : k_shot + q_query]
                for role, row_ids in (("support", support_ids), ("query", query_ids)):
                    for row_id in row_ids:
                        rows.append(
                            {
                                "episode_id": episode_id,
                                "seed": int(seed),
                                "canonical_e3": e3,
                                "Target": str(target),
                                "role": role,
                                "label": label,
                                "sample_order": role_orders[role],
                                "row_id": row_id,
                                "query_row_id": row_id if role == "query" else None,
                            }
                        )
                        role_orders[role] += 1
    if not rows:
        raise MissingContextError(
            f"No target is K{k_shot}Q{q_query} eligible for E3={e3}"
        )
    manifest_payload = [
        (
            row["episode_id"],
            row["role"],
            row["label"],
            row["sample_order"],
            row["row_id"],
        )
        for row in rows
    ]
    manifest_id = "manifest_" + hashlib.sha256(
        json.dumps(manifest_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    for row in rows:
        row["manifest_id"] = manifest_id
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def query_row_ids(manifest: pd.DataFrame) -> tuple[str, ...]:
    _require_columns(manifest, ("role", "row_id", "query_row_id"))
    query = manifest.loc[manifest["role"].eq("query")]
    if query["query_row_id"].isna().any():
        raise DataValidationError("Query manifest rows must have query_row_id")
    if not query["row_id"].astype(str).eq(query["query_row_id"].astype(str)).all():
        raise DataValidationError("query_row_id must equal the stable context row_id")
    return tuple(query["query_row_id"].astype(str))


def materialize_episodes(
    contexts: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
) -> list[Episode]:
    """Materialize exactly the rows in a frozen episode manifest."""

    _require_columns(manifest, MANIFEST_COLUMNS)
    frame = prepare_contexts(contexts, columns=columns, aliases=aliases)
    lookup = frame.set_index("row_id", drop=False)
    requested = set(manifest["row_id"].astype(str))
    missing = requested - set(lookup.index.astype(str))
    if missing:
        raise MissingContextError(
            f"Manifest rows are absent from contexts: {sorted(missing)[:5]}"
        )
    episodes: list[Episode] = []
    for episode_id, selection in manifest.groupby("episode_id", sort=False):
        targets = selection["Target"].astype(str).unique()
        e3s = selection["canonical_e3"].astype(str).unique()
        if len(targets) != 1 or len(e3s) != 1:
            raise DataValidationError(f"Manifest episode {episode_id} mixes contexts")
        selected_rows: dict[str, pd.DataFrame] = {}
        for role in ("support", "query"):
            role_manifest = selection.loc[selection["role"].eq(role)].sort_values(
                "sample_order"
            )
            materialized = lookup.loc[role_manifest["row_id"].astype(str)].copy()
            materialized = materialized.reset_index(drop=True)
            expected_labels = role_manifest["label"].astype(int).tolist()
            if materialized[columns.label].astype(int).tolist() != expected_labels:
                raise DataValidationError(
                    f"Labels changed after manifest creation for episode {episode_id}"
                )
            selected_rows[role] = materialized
        episodes.append(
            Episode(
                str(episode_id),
                targets[0],
                e3s[0],
                selected_rows["support"],
                selected_rows["query"],
            )
        )
    return episodes


def build_e3_sequence_catalog(
    sequence_index: Mapping[ContextKey, ContextSequences],
) -> dict[str, str]:
    """Collapse context sequences to one sequence per E3, rejecting conflicts."""

    observed: dict[str, set[str]] = {}
    for key, sequences in sequence_index.items():
        if sequences.e3_sequence is not None:
            observed.setdefault(key.e3, set()).add(sequences.e3_sequence)
    conflicting = {e3: values for e3, values in observed.items() if len(values) > 1}
    if conflicting:
        summary = {e3: len(values) for e3, values in conflicting.items()}
        raise AmbiguousContextError(f"E3 sequence catalog is ambiguous: {summary}")
    return {e3: next(iter(values)) for e3, values in observed.items()}


def make_e3_mapping(
    e3_values: Iterable[Any],
    *,
    mode: Literal["identity", "shuffle", "wrong"] = "shuffle",
    seed: int = 0,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build identity, seeded shuffled, or deterministic wrong-E3 mappings.

    Both non-identity modes are derangements: no E3 maps to itself.  ``wrong``
    is a sorted cyclic rotation, while ``shuffle`` is a seeded derangement.
    """

    values = sorted({canonicalize_e3(value, aliases) for value in e3_values})
    if not values:
        raise DataValidationError("No E3 values were provided")
    if mode == "identity":
        return dict(zip(values, values))
    if mode not in {"shuffle", "wrong"}:
        raise ValueError(f"Unknown E3 mapping mode: {mode}")
    if len(values) < 2:
        raise DataValidationError(f"{mode} mapping requires at least two E3s")
    if mode == "wrong":
        donors = values[1:] + values[:1]
    else:
        rng = random.Random(seed)
        donors = list(values)
        for _ in range(1000):
            rng.shuffle(donors)
            if all(source != donor for source, donor in zip(values, donors)):
                break
        else:  # Defensive deterministic fallback; normally reached only never.
            donors = values[1:] + values[:1]
    mapping = dict(zip(values, donors))
    if any(source == donor for source, donor in mapping.items()):
        # Rejection can repeatedly return fixed points for small unlucky streams.
        donors = values[1:] + values[:1]
        mapping = dict(zip(values, donors))
    return mapping


def apply_e3_mapping(
    contexts: pd.DataFrame,
    mapping: Mapping[Any, Any],
    *,
    e3_sequence_catalog: Mapping[Any, str] | None = None,
    columns: ContextColumns = DEFAULT_COLUMNS,
    aliases: Mapping[str, str] | None = None,
    model_e3_column: str = "model_e3",
    model_e3_sequence_column: str = "model_E3_seq",
) -> pd.DataFrame:
    """Apply a model-input E3 mapping without altering context identity.

    The biological ``canonical_e3`` and ``row_id`` remain unchanged, which is
    essential for paired full/shuffled/wrong-E3 comparisons.
    """

    frame = prepare_contexts(contexts, columns=columns, aliases=aliases)
    canonical_mapping = {
        canonicalize_e3(source, aliases): canonicalize_e3(donor, aliases)
        for source, donor in mapping.items()
    }
    missing_sources = set(frame["canonical_e3"]) - set(canonical_mapping)
    if missing_sources:
        raise MissingContextError(
            f"E3 mapping omits biological E3s: {sorted(missing_sources)}"
        )
    frame[model_e3_column] = frame["canonical_e3"].map(canonical_mapping)
    frame["e3_intervention_changed"] = frame[model_e3_column].ne(
        frame["canonical_e3"]
    )
    if e3_sequence_catalog is not None:
        catalog: dict[str, str] = {}
        for e3, sequence in e3_sequence_catalog.items():
            canonical = canonicalize_e3(e3, aliases)
            normalized = normalize_protein_sequence(sequence)
            if normalized is None:
                continue
            if canonical in catalog and catalog[canonical] != normalized:
                raise AmbiguousContextError(
                    f"Conflicting sequence catalog entries for E3={canonical}"
                )
            catalog[canonical] = normalized
        missing_donors = set(frame[model_e3_column]) - set(catalog)
        if missing_donors:
            raise MissingContextError(
                f"No sequence for mapped E3s: {sorted(missing_donors)}"
            )
        frame[model_e3_sequence_column] = frame[model_e3_column].map(catalog)
    return frame


__all__ = [
    "ATOM_FEATURE_NAMES",
    "BOND_FEATURE_NAMES",
    "ContextColumns",
    "ContextComponents",
    "ContextGraphIndex",
    "ContextIndexAudit",
    "ContextKey",
    "ContextSequences",
    "CrossE3Split",
    "DEFAULT_COLUMNS",
    "DEFAULT_E3_ALIASES",
    "DataValidationError",
    "AmbiguousContextError",
    "AmbiguousSubstructureError",
    "E3LigandMatchAudit",
    "Episode",
    "InvalidMoleculeError",
    "MissingContextError",
    "SplitLeakageError",
    "apply_e3_mapping",
    "assemble_context_table",
    "attach_context_components",
    "attach_context_sequences",
    "atom_features",
    "bond_features",
    "build_context_component_index",
    "build_context_graph_index",
    "build_context_sequence_index",
    "build_e3_sequence_catalog",
    "build_episode_manifest",
    "canonicalize_smiles",
    "canonicalize_e3",
    "context_key",
    "make_e3_mapping",
    "match_e3_ligand",
    "materialize_episodes",
    "molecule_to_graph",
    "normalize_compound_id",
    "normalize_protein_sequence",
    "prepare_contexts",
    "query_row_ids",
    "smiles_to_graph",
    "split_source_target_e3",
    "stable_context_id",
]
