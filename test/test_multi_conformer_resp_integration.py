import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:
    from rdkit import Chem
    from rdkit import Geometry as Geom
    from radonpy.sim import psi4_wrapper
    _QM_AVAILABLE = True
except ImportError:
    _QM_AVAILABLE = False


def make_upstream_ethanol_conformers():
    symbols = ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H')
    coordinates = (
        (
            (0.00000000, 0.00000000, 0.00000000),
            (1.48805540, -0.00728176, 0.39653260),
            (2.04971655, 1.37648153, 0.25604810),
            (3.06429978, 1.37151670, 0.52641124),
            (1.58679428, -0.33618761, 1.43102358),
            (2.03441010, -0.68906454, -0.25521028),
            (-0.40814044, -1.00553466, 0.10208540),
            (-0.54635470, 0.68178278, 0.65174288),
            (-0.09873888, 0.32890585, -1.03449097),
        ),
        (
            (0.00000000, 0.00000000, 0.00000000),
            (1.48013500, -0.00724300, 0.39442200),
            (2.00696300, 1.29224100, 0.26232800),
            (2.91547900, 1.25572900, 0.50972300),
            (1.61500700, -0.32678000, 1.45587700),
            (2.07197500, -0.68695100, -0.26493400),
            (-0.32500012, 1.02293415, -0.30034094),
            (-0.18892141, -0.68463906, -0.85893815),
            (-0.64257065, -0.32709111, 0.84987482),
        ),
    )

    editable = Chem.RWMol()
    for symbol in symbols:
        editable.AddAtom(Chem.Atom(symbol))
    for atom1, atom2 in ((0, 1), (1, 2), (2, 3), (1, 4), (1, 5), (0, 6), (0, 7), (0, 8)):
        editable.AddBond(atom1, atom2, Chem.BondType.SINGLE)
    mol = editable.GetMol()
    Chem.SanitizeMol(mol)

    for conformer_coordinates in coordinates:
        conformer = Chem.Conformer(len(symbols))
        for atom_id, position in enumerate(conformer_coordinates):
            conformer.SetAtomPosition(atom_id, Geom.Point3D(*position))
        mol.AddConformer(conformer, assignId=True)
    return mol


@unittest.skipUnless(_QM_AVAILABLE, 'Psi4, RESP, and RDKit are required for this integration test')
class TestMultiConformerRespIntegration(unittest.TestCase):
    def run_resp(self, mol, conf_ids, weights, work_dir, name):
        wrapper = psi4_wrapper.Psi4w(
            mol,
            work_dir=work_dir,
            tmp_dir=work_dir,
            name=name,
            method='HF',
            basis='6-31G(d)',
            num_threads=1,
            memory=1000,
        )
        charges = wrapper.resp_multi(conf_ids, weights=weights)
        self.assertFalse(wrapper.error_flag)
        return charges

    def test_upstream_ethanol_two_conformer_case(self):
        mol = make_upstream_ethanol_conformers()
        with tempfile.TemporaryDirectory() as work_dir1, tempfile.TemporaryDirectory() as work_dir2:
            charges_forward = self.run_resp(mol, [0, 1], [1.0, 1.0], work_dir1, 'ethanol_forward')
            charges_reverse = self.run_resp(mol, [1, 0], [1.0, 1.0], work_dir2, 'ethanol_reverse')

        self.assertTrue(np.all(np.isfinite(charges_forward)))
        self.assertAlmostEqual(float(np.sum(charges_forward)), 0.0, places=7)
        np.testing.assert_allclose(charges_forward, charges_reverse, rtol=0.0, atol=1.0e-7)

    def test_upstream_reference_charges_with_upstream_options(self):
        # Run this check in a fresh process because Psi4 owns global state and can
        # crash when a raw upstream calculation follows initialized wrappers.
        runner = os.path.join(os.path.dirname(__file__), 'resp_reference_runner.py')
        completed = subprocess.run(
            [sys.executable, runner],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)


if __name__ == '__main__':
    unittest.main()
