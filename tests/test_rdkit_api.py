import unittest
from unittest.mock import patch

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from api.rdkit import (
    InvalidSmilesError,
    are_same_structure,
    get_molecule_properties,
    get_structure_key,
    to_canonical_smiles,
)


class RdkitApiTests(unittest.TestCase):
    def test_descriptor_values_match_original_rdkit_functions(self):
        for smiles in ("CCO", "CC(=O)NCC", "[NH4+].[Cl-]", "[13CH3]C"):
            with self.subTest(smiles=smiles):
                mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
                props = get_molecule_properties(smiles).to_dict()
                expected = {
                    "mwFreebase": getattr(Descriptors, "MolWt")(mol),
                    "alogp": getattr(Crippen, "MolLogP")(mol),
                    "hba": rdMolDescriptors.CalcNumHBA(mol),
                    "hbd": rdMolDescriptors.CalcNumHBD(mol),
                    "psa": rdMolDescriptors.CalcTPSA(mol),
                    "rtb": getattr(Lipinski, "NumRotatableBonds")(mol),
                }
                self.assertEqual(props, expected)

    def test_molecular_weight_falls_back_to_exact_weight(self):
        with patch("api.rdkit._mol_weight", side_effect=RuntimeError("unavailable")):
            props = get_molecule_properties("CCO")
        assert props.mwFreebase is not None
        self.assertAlmostEqual(props.mwFreebase, 46.041864812)

    def test_equivalent_smiles_match_with_and_without_inchi(self):
        for available in (True, False):
            with self.subTest(inchi=available), patch("api.rdkit.INCHI_AVAILABLE", available):
                self.assertTrue(are_same_structure("CCO", "OCC"))
                self.assertFalse(are_same_structure("CCO", "CCC"))

    def test_stereoisomers_remain_distinct_without_inchi(self):
        with patch("api.rdkit.INCHI_AVAILABLE", False):
            self.assertFalse(are_same_structure("C[C@H](O)F", "C[C@@H](O)F"))
            self.assertTrue(get_structure_key("CCO").startswith("CANON:"))

    def test_inchi_failure_uses_canonical_smiles(self):
        with patch("api.rdkit.inchi") as inchi:
            inchi.MolToInchiKey.side_effect = RuntimeError("unavailable")
            inchi.MolToInchi.side_effect = RuntimeError("unavailable")
            self.assertEqual(get_structure_key("OCC"), "CANON:CCO")

    def test_canonical_smiles_preserve_stereochemistry(self):
        a = to_canonical_smiles("C[C@H](O)F")
        b = to_canonical_smiles("C[C@@H](O)F")
        self.assertNotEqual(a, b)
        self.assertEqual(to_canonical_smiles("OCC"), "CCO")

    def test_empty_smiles_raise_domain_error(self):
        for operation in (get_molecule_properties, to_canonical_smiles, get_structure_key):
            with self.subTest(operation=operation.__name__), self.assertRaises(InvalidSmilesError):
                operation(" ")


if __name__ == "__main__":
    unittest.main()
