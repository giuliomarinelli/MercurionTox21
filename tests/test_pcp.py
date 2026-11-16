from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api.pcp import CompoundNotFoundError, get_iupac_name_from_smiles
from api.rdkit import InvalidSmilesError


class PubChemTests(unittest.TestCase):
    def test_client_receives_canonical_smiles_and_returns_iupac_name(self):
        with patch("api.pcp.pcp.get_compounds", return_value=[SimpleNamespace(iupac_name="ethanol")]) as client:
            self.assertEqual(get_iupac_name_from_smiles("OCC"), "ethanol")
            client.assert_called_once_with("CCO", "smiles")

    def test_absent_compound_is_distinct_from_absent_name(self):
        with patch("api.pcp.pcp.get_compounds", return_value=[]):
            with self.assertRaises(CompoundNotFoundError):
                get_iupac_name_from_smiles("CCO")
        for name in (None, ""):
            with self.subTest(name=name), patch("api.pcp.pcp.get_compounds", return_value=[SimpleNamespace(iupac_name=name)]):
                self.assertEqual(get_iupac_name_from_smiles("CCO"), "")

    def test_invalid_smiles_never_contact_the_client(self):
        with patch("api.pcp.pcp.get_compounds") as client:
            for smiles in ("", " ", "not-a-smiles"):
                with self.subTest(smiles=smiles), self.assertRaises(InvalidSmilesError):
                    get_iupac_name_from_smiles(smiles)
            client.assert_not_called()

    def test_client_failure_is_propagated_instead_of_fabricating_a_name(self):
        with patch("api.pcp.pcp.get_compounds", side_effect=TimeoutError("synthetic lookup failure")):
            with self.assertRaises(TimeoutError):
                get_iupac_name_from_smiles("CCO")


if __name__ == "__main__":
    unittest.main()
