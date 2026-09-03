import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from rdkit import Chem
from rdkit import Geometry as Geom

from radonpy.core import const
from radonpy.sim import psi4_wrapper


class TestPsi4wInitialization(unittest.TestCase):
    def setUp(self):
        mol = Chem.AddHs(Chem.MolFromSmiles('O'))
        coordinates = (
            ((0.0, 0.0, 0.0), (0.9, 0.0, 0.0), (-0.2, 0.8, 0.0)),
            ((2.0, 3.0, 4.0), (3.1, 3.0, 4.0), (1.8, 4.2, 4.3)),
        )
        for conformer_coordinates in coordinates:
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for atom_id, position in enumerate(conformer_coordinates):
                conformer.SetAtomPosition(atom_id, Geom.Point3D(*position))
            mol.AddConformer(conformer, assignId=True)

        self.coordinates = np.array(coordinates)
        self.wrapper = psi4_wrapper.Psi4w(
            mol,
            confId=1,
            name='default_name',
            charge=-1,
            multiplicity=2,
            num_threads=1,
        )

    def assert_molecule_coordinates(self, molecule, expected):
        actual = molecule.geometry().to_array() * const.bohr2ang
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-7)

    def test_mol2psi4_uses_default_conformer_name_charge_and_multiplicity(self):
        molecule = self.wrapper._mol2psi4('no_reorient', 'no_com')

        self.assert_molecule_coordinates(molecule, self.coordinates[1])
        self.assertEqual(molecule.name(), 'default_name')
        self.assertEqual(molecule.molecular_charge(), -1)
        self.assertEqual(molecule.multiplicity(), 2)

    def test_mol2psi4_accepts_conformer_and_name_without_changing_defaults(self):
        molecule = self.wrapper._mol2psi4(
            'no_reorient', 'no_com', confId=0, name='explicit_name'
        )

        self.assert_molecule_coordinates(molecule, self.coordinates[0])
        self.assertEqual(molecule.name(), 'explicit_name')
        self.assertEqual(self.wrapper.confId, 1)
        self.assertEqual(self.wrapper.name, 'default_name')

    def test_init_calls_internal_steps_in_order_and_returns_molecule(self):
        calls = []
        expected = object()

        with mock.patch.object(
            self.wrapper, '_prepare_psi4', side_effect=lambda: calls.append('prepare')
        ), mock.patch.object(
            self.wrapper,
            '_mol2psi4',
            side_effect=lambda *args: calls.append(('mol', args)) or expected,
        ), mock.patch.object(
            self.wrapper,
            '_configure_psi4',
            side_effect=lambda output=None: calls.append(('configure', output)),
        ):
            actual = self.wrapper._init_psi4('symmetry c1', output='output.log')

        self.assertIs(actual, expected)
        self.assertEqual(
            calls,
            ['prepare', ('mol', ('symmetry c1',)), ('configure', 'output.log')],
        )

    def test_fin_restores_working_directory_and_none_does_not_set_output(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as work_dir:
            self.wrapper.work_dir = work_dir
            self.wrapper.tmp_dir = work_dir
            try:
                with mock.patch.object(
                    psi4_wrapper.psi4.core, 'set_output_file'
                ) as set_output_file:
                    self.wrapper._init_psi4('no_reorient', 'no_com')
                    self.assertEqual(os.getcwd(), work_dir)
                    set_output_file.assert_not_called()

                self.wrapper._fin_psi4()
                self.assertEqual(os.getcwd(), original_cwd)
            finally:
                os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
