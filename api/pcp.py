from .rdkit import InvalidSmilesError, to_canonical_smiles
import pubchempy as pcp
from typing import cast

def get_iupac_name_from_smiles(smiles: str) -> str:
    if not smiles:
        raise InvalidSmilesError("SMILES string is empty")
    canon = to_canonical_smiles(smiles)
    compounds = cast(list[pcp.Compound], pcp.get_compounds(canon, 'smiles'))
    if not compounds:
        raise InvalidSmilesError(f"Could not find compound for canonical SMILES: {canon}")
    iupac_name = compounds[0].iupac_name
    if iupac_name is None:
        iupac_name = ""
    return iupac_name