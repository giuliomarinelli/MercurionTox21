from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    Crippen,
    rdMolDescriptors,
    Lipinski,
    AllChem
)

try:
    from rdkit.Chem import inchi
    INCHI_AVAILABLE = True
except ImportError:  # RDKit senza supporto InChI
    inchi = None
    INCHI_AVAILABLE = False


# =========================
# ERRORI
# =========================

class InvalidSmilesError(ValueError):
    pass


class CanonicalizationError(RuntimeError):
    pass


# =========================
# MODEL PROPRIETÀ
# =========================

@dataclass
class MoleculePropertiesDTO:
    mwFreebase: Optional[float]
    alogp: Optional[float]
    hba: Optional[int]
    hbd: Optional[int]
    psa: Optional[float]
    rtb: Optional[int]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


# =========================
# HELPERS DI BASE
# =========================

def _mol_from_smiles(smiles: str) -> Chem.Mol:
    smiles = (smiles or "").strip()
    if not smiles:
        raise InvalidSmilesError("SMILES vuota")

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise InvalidSmilesError(f"SMILES non valida: {smiles}")

    # coerente con wasm: togliamo H espliciti
    mol = Chem.RemoveHs(mol)
    return mol


# =========================
# PROPRIETÀ MOLECOLARI
# =========================

def get_molecule_properties(smiles: str) -> MoleculePropertiesDTO:
    """
    Restituisce le proprietà base usate in Mercurion.
    """
    mol = _mol_from_smiles(smiles)

    def _safe(fn, default=None):
        try:
            return fn(mol)
        except Exception:
            return default

    mw = _safe(Descriptors.MolWt)
    if mw is None:
        mw = _safe(Descriptors.ExactMolWt)

    props = MoleculePropertiesDTO(
        mwFreebase=float(mw) if mw is not None else None,
        alogp=_safe(Crippen.MolLogP),
        hba=_safe(rdMolDescriptors.CalcNumHBA),
        hbd=_safe(rdMolDescriptors.CalcNumHBD),
        psa=_safe(rdMolDescriptors.CalcTPSA),
        rtb=_safe(Lipinski.NumRotatableBonds),
    )

    return props


# =========================
# CANONICAL SMILES STABILE
# =========================

def to_canonical_smiles(
    smiles: str,
    *,
    isomeric: bool = True,
    kekule: bool = False,
) -> str:
    """
    Canonicalizzazione “stabile” con un minimo di round-trip.
    Lancia CanonicalizationError se qualcosa va storto.
    """
    mol = _mol_from_smiles(smiles)

    try:
        c1 = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=isomeric,
            kekuleSmiles=kekule,
        )
    except Exception:
        # fallback molto conservativo
        try:
            c1 = Chem.MolToSmiles(mol, canonical=True)
        except Exception as exc:
            raise CanonicalizationError("Canonicalizzazione fallita") from exc

    if not c1:
        raise CanonicalizationError("Canonicalizzazione fallita (stringa vuota)")

    # round-trip per stabilizzare un minimo
    mol2 = _mol_from_smiles(c1)
    try:
        c2 = Chem.MolToSmiles(
            mol2,
            canonical=True,
            isomericSmiles=isomeric,
            kekuleSmiles=kekule,
        )
    except Exception:
        c2 = Chem.MolToSmiles(mol2, canonical=True)

    return c2 or c1


# =========================
# STRUCTURE KEY INDIPENDENTE
# =========================

def get_structure_key(smiles: str) -> str:
    """
    Indicatore univoco di struttura, indipendente dalla forma delle SMILES.

    Ordine di preferenza:
    1) InChIKey
    2) InChI
    3) Morgan fingerprint (bitstring)
    4) Fallback: canonical SMILES
    """
    mol = _mol_from_smiles(smiles)

    # 1) InChIKey
    if INCHI_AVAILABLE:
        try:
            key = inchi.MolToInchiKey(mol)  # type: ignore[attr-defined]
            if key and key.strip():
                return f"INCHIKEY:{key}"
        except Exception:
            pass

        # 2) InChI completo
        try:
            inchi_str = inchi.MolToInchi(mol)  # type: ignore[attr-defined]
            if inchi_str and inchi_str.strip():
                return f"INCHI:{inchi_str}"
        except Exception:
            pass

    # 3) Morgan FP
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_str = fp.ToBitString()
        if fp_str and fp_str.strip("0"):
            return f"MFP2:{fp_str}"
    except Exception:
        pass

    # 4) Fallback: canonical SMILES “semplice”
    try:
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if can and can.strip():
            return f"CANON:{can}"
    except Exception:
        pass

    raise CanonicalizationError("Impossibile derivare una structure key")


# =========================
# CONFRONTO STRUTTURE
# =========================

def are_same_structure(smiles_a: str, smiles_b: str) -> bool:
    """
    Confronto robusto: non guarda le SMILES ma la structure key.
    """
    if not smiles_a or not smiles_b:
        return False

    try:
        ka = get_structure_key(smiles_a)
        kb = get_structure_key(smiles_b)
    except (InvalidSmilesError, CanonicalizationError):
        return False

    return ka == kb
