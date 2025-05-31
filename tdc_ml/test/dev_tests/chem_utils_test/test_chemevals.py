import os
import sys

import unittest
import shutil


class TestEvals(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        self.smi_short = ['CCO', 'CCN', 'CCC', 'CCCl',
                          'CCBr']  # ethanol, ethylamine, propane, chloroethane
        self.smi_long = [
            "CCO", "CCN", "CCCl", "CCBr", "CCC", "CC(C)C", "CCCC", "CC(C)O",
            "CC(C)N", "CC(C)Cl", "C1CCCCC1", "C1=CC=CC=C1", "C1=CC=CN=C1",
            "c1ccncc1", "c1ccccc1O", "c1ccccc1N", "c1ccccc1Cl", "c1ccccc1Br",
            "c1ccccc1C", "c1ccccc1CC", "COC", "COCC", "CN(C)C", "CC(=O)O",
            "C(C(=O)O)N", "NC(=O)C", "CC(=O)N", "CCOC", "CCN(CC)CC",
            "CC(C)(C)C", "c1cc(c(cc1O)O)C(=O)O", "O=C(O)Cc1ccccc1", "N#CC",
            "C#N", "O=C=O", "ClCCl", "BrCBr", "F", "Cl", "Br", "c1ccccc1C(=O)O",
            "c1ccccc1C(=O)N", "c1ccccc1C#N", "c1ccccc1S", "c1ccccc1CO",
            "c1ccccc1CN", "c1ccccc1OC", "c1ccccc1NC", "c1ccccc1NO",
            "c1ccccc1CCO", "CC(C)C(C)(C)C", "CC(C)OC(=O)C", "CNC(C)C",
            "CN(C)C(=O)C", "CC(C)CC(=O)O", "C(C(=O)O)(N)CC", "CC(C)C(=O)O",
            "CC(C)CO", "CC(C)CN", "CC(C)Cl", "OC(=O)c1ccccc1", "OC(=O)c1ccncc1",
            "OC(=O)c1ccc(Cl)cc1", "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)COC(=O)C",
            "COC1=CC=CC=C1OC", "COC1=CC=CC=C1", "CN1CCCC1C(=O)O",
            "CC1=CC=CC=C1O", "C1=CC=C(C=C1)C(=O)O", "O=C(C)Oc1ccccc1C(=O)O",
            "O=C(NC)C", "O=C(NCc1ccccc1)C", "CC(C)CC", "CC1=CC=CN=C1",
            "C1=CN=CN1", "CC1=NC=CC=C1", "CC1=CC=C(O)C=C1", "COC(=O)C",
            "COC(=O)OC", "CCN(CC)C(=O)C", "CC(C)C(=O)N", "CC(C)CC",
            "C1=CC=CC(=C1)C(=O)O", "CC1=CC=CC=C1C", "CC(=O)C", "CC(=O)OC",
            "CCOC(=O)C", "CCOC(=O)OC", "C=CC(=O)OC", "C1=CC2=CC=CC=C2C=C1",
            "C1=CC2=CC=CN=C2C=C1", "C1=NC=C(C=C1)C(=O)O", "CC(C)(C)CO",
            "C(C(C(=O)O)N)CO", "CC(C)C(=O)O", "CC(C)(C)C(=O)O", "CCCCCCCC(=O)O",
            "CCCCCCCCCCCC(=O)O", "CCCCCCCCCCCCCCCC(=O)O"
        ]
        super().__init__(methodName)

    def setUp(self):
        print(os.getcwd())
        pass

    def test_ncircles(self):
        from tdc_ml.chem_utils.evaluator import ncircles, ncircles_recursive

        # test base case
        k_seq, C_seq = ncircles(self.smi_short)
        k_rec, C_rec = ncircles_recursive(self.smi_short, L=0, m=2)

        assert k_rec == k_seq

        # test on long sequences
        k, centres = ncircles_recursive(self.smi_long, L=3, m=5, t=0.5)

        assert k == 73
        assert len(centres) == k

    def test_ham_div(self):
        from tdc_ml.chem_utils.evaluator import hamiltonian_diversity
        # test on short
        ham_div = hamiltonian_diversity(self.smi_short)

        assert ham_div >= 0
        assert f'{ham_div:.2f}' == '3.14'

        ham_div = hamiltonian_diversity(self.smi_long)

        assert ham_div >= 0
        assert f'{ham_div:.2f}' == '56.88'

    def test_vendi(self):
        from tdc_ml.chem_utils.evaluator import mol_vendi

        score = mol_vendi(self.smi_short)

        assert score >= 0
        assert f'{score:.2f}' == '4.00'

        score_big = mol_vendi(self.smi_long)

        assert score_big >= 0
        assert f'{score_big:.2f}' == '52.01'

    def test_posecheck(self):
        import tempfile

        from rdkit import Chem
        from rdkit.Chem import AllChem

        from tdc_ml.chem_utils.evaluator import load_posecheck

        MINIMAL_PDB = (
            "MODEL        1\n"
            "ATOM      1  N   GLY A   1      11.104  13.207   2.056  1.00 20.00           N\n"
            "ATOM      2  CA  GLY A   1      12.567  13.487   2.048  1.00 20.00           C\n"
            "ATOM      3  C   GLY A   1      13.005  14.746   2.857  1.00 20.00           C\n"
            "ATOM      4  O   GLY A   1      14.185  14.982   2.986  1.00 20.00           O\n"
            "TER\n"
            "ENDMDL\n"
            "END\n")

        with tempfile.NamedTemporaryFile("w", suffix=".pdb",
                                         delete=False) as tmp:
            tmp.write(MINIMAL_PDB)
            tmp.flush()
            pdb_path = tmp.name

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        pc = load_posecheck()
        pc.load_protein_from_pdb(pdb_path)
        pc.load_ligands_from_mols([mol])
        res = pc.run()

        assert isinstance(res, dict)
        assert 'clashes' in res
        assert 'strain' in res
        assert 'interactions' in res

    def teardown(self):
        print(os.getcwd())

        if os.path.exists(os.path.join(os.getcwd(), 'data')):
            shutil.rmtree(os.path.join(os.getcwd(), 'data'))
