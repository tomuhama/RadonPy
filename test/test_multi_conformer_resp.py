import copy
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from rdkit import Chem
from rdkit import Geometry as Geom

from radonpy.core import calc
from radonpy.sim import psi4_wrapper, qm


def make_two_conformer_mol():
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
    return mol


class FakeChargeWrapper:
    def __init__(self, mol):
        self.mol = Chem.Mol(mol)
        self.error_flag = False
        self.calls = []

    def resp(self):
        self.calls.append(('resp',))
        for i, atom in enumerate(self.mol.GetAtoms()):
            atom.SetDoubleProp('ESP', -0.1 - i)
            atom.SetDoubleProp('RESP', 0.1 + i)

    def resp_multi(self, confId_list, weights=None):
        self.calls.append(('resp_multi', confId_list, weights))
        for i, atom in enumerate(self.mol.GetAtoms()):
            atom.SetDoubleProp('ESP', -0.2 - i)
            atom.SetDoubleProp('RESP', 0.2 + i)

    def mulliken_charge(self, recalc=False):
        self.calls.append(('mulliken', recalc))
        for i, atom in enumerate(self.mol.GetAtoms()):
            atom.SetDoubleProp('MullikenCharge', 0.3 + i)

    def lowdin_charge(self, recalc=False):
        self.calls.append(('lowdin', recalc))
        for i, atom in enumerate(self.mol.GetAtoms()):
            atom.SetDoubleProp('LowdinCharge', 0.4 + i)


class FakePsi4Molecule:
    def __init__(self, name, symbols=('O', 'H', 'H'), charge=0, multiplicity=1):
        self._name = name
        self._symbols = tuple(symbols)
        self._charge = charge
        self._multiplicity = multiplicity

    def name(self):
        return self._name

    def natom(self):
        return len(self._symbols)

    def symbol(self, index):
        return self._symbols[index]

    def molecular_charge(self):
        return self._charge

    def multiplicity(self):
        return self._multiplicity


class TestAssignChargesRouting(unittest.TestCase):
    def setUp(self):
        self.mol = make_two_conformer_mol()

    def test_qm_wrapper_forwards_multi_arguments_unchanged(self):
        conf_ids = [1, 0]
        weights = [2.0, 0.5]
        with mock.patch.object(qm.calc, 'assign_charges', return_value=True) as assign:
            result = qm.assign_charges(
                self.mol,
                charge='multi-conformer RESP',
                confId_list=conf_ids,
                weights=weights,
                opt=False,
            )

        self.assertTrue(result)
        self.assertIs(assign.call_args.kwargs['confId_list'], conf_ids)
        self.assertIs(assign.call_args.kwargs['weights'], weights)

    def test_resp_uses_only_resp_and_multi_uses_only_resp_multi(self):
        wrapper = FakeChargeWrapper(self.mol)
        with mock.patch.object(calc, 'QMw', return_value=wrapper):
            self.assertTrue(calc.assign_charges(self.mol, charge='RESP', opt=False))
        self.assertEqual(wrapper.calls, [('resp',)])

        wrapper = FakeChargeWrapper(self.mol)
        with mock.patch.object(calc, 'QMw', return_value=wrapper):
            self.assertTrue(
                calc.assign_charges(
                    self.mol,
                    charge='multi-conformer RESP',
                    confId_list=[1, 0],
                    weights=[2.0, 0.5],
                    opt=False,
                )
            )
        self.assertEqual(wrapper.calls, [('resp_multi', [1, 0], [2.0, 0.5])])
        for atom in self.mol.GetAtoms():
            self.assertEqual(atom.GetDoubleProp('AtomicCharge'), atom.GetDoubleProp('RESP'))

    def test_other_qm_charge_routes_are_unchanged(self):
        expected = {
            'ESP': ('resp',),
            'Mulliken': ('mulliken', True),
            'Lowdin': ('lowdin', True),
        }
        for charge, expected_call in expected.items():
            with self.subTest(charge=charge):
                mol = make_two_conformer_mol()
                wrapper = FakeChargeWrapper(mol)
                with mock.patch.object(calc, 'QMw', return_value=wrapper):
                    self.assertTrue(calc.assign_charges(mol, charge=charge, opt=False))
                self.assertEqual(wrapper.calls, [expected_call])

    def test_multi_rejects_optimization_and_non_psi4_solver_without_calculation(self):
        with mock.patch.object(calc, 'QMw') as wrapper_factory:
            self.assertFalse(
                calc.assign_charges(
                    self.mol,
                    charge='multi-conformer RESP',
                    confId_list=[0, 1],
                    opt=True,
                )
            )
            self.assertFalse(
                calc.assign_charges(
                    self.mol,
                    charge='multi-conformer RESP',
                    confId_list=[0, 1],
                    opt=False,
                    qm_solver='nwchem',
                )
            )
        wrapper_factory.assert_not_called()

    def test_multi_rejects_invalid_conformer_inputs_without_calculation(self):
        invalid_inputs = (
            None,
            [],
            [0],
            [0, 9],
            [0, 0],
            [0, '1'],
            [0, True],
        )
        with mock.patch.object(calc, 'QMw') as wrapper_factory:
            for conf_ids in invalid_inputs:
                with self.subTest(conf_ids=conf_ids):
                    self.assertFalse(
                        calc.assign_charges(
                            self.mol,
                            charge='multi-conformer RESP',
                            confId_list=conf_ids,
                            opt=False,
                        )
                    )
        wrapper_factory.assert_not_called()

    def test_multi_rejects_invalid_weights_without_calculation(self):
        invalid_weights = (
            [1.0],
            [1.0, -0.1],
            [1.0, np.nan],
            [1.0, np.inf],
            [0.0, 0.0],
            [1.0, object()],
        )
        with mock.patch.object(calc, 'QMw') as wrapper_factory:
            for weights in invalid_weights:
                with self.subTest(weights=weights):
                    self.assertFalse(
                        calc.assign_charges(
                            self.mol,
                            charge='multi-conformer RESP',
                            confId_list=[0, 1],
                            weights=weights,
                            opt=False,
                        )
                    )
        wrapper_factory.assert_not_called()

    def test_multi_defaults_to_equal_weights_without_normalizing(self):
        wrapper = FakeChargeWrapper(self.mol)
        with mock.patch.object(calc, 'QMw', return_value=wrapper):
            self.assertTrue(
                calc.assign_charges(
                    self.mol,
                    charge='multi-conformer RESP',
                    confId_list=[1, 0],
                    opt=False,
                )
            )
        self.assertEqual(wrapper.calls, [('resp_multi', [1, 0], [1.0, 1.0])])


class TestPsi4wMultiResp(unittest.TestCase):
    def setUp(self):
        self.wrapper = psi4_wrapper.Psi4w(
            make_two_conformer_mol(),
            confId=1,
            name='multi_test',
            num_threads=1,
        )

    def test_joint_two_stage_fit_reuses_all_conformer_files(self):
        stage1 = np.array(((-0.3, 0.15, 0.15), (-0.4, 0.2, 0.2)))
        stage2 = np.array(((-0.2, 0.1, 0.1), (-0.5, 0.25, 0.25)))
        pmols = [FakePsi4Molecule('multi_test_conf1'), FakePsi4Molecule('multi_test_conf0')]
        resp_calls = []

        def fake_resp(molecules, options):
            resp_calls.append((molecules, copy.deepcopy(options)))
            if len(resp_calls) == 1:
                for i, molecule in enumerate(molecules):
                    open('%i_%s_grid.dat' % (i + 1, molecule.name()), 'w').close()
                    open('%i_%s_grid_esp.dat' % (i + 1, molecule.name()), 'w').close()
                return stage1
            return stage2

        def set_constraints(molecule, charges, options):
            options['constraint_charge'] = [[float(charges[0]), [1]]]
            options['constraint_group'] = [[2, 3]]

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as work_dir:
            os.chdir(work_dir)
            try:
                with mock.patch.object(self.wrapper, '_prepare_psi4') as prepare, \
                     mock.patch.object(self.wrapper, '_configure_psi4') as configure, \
                     mock.patch.object(self.wrapper, '_fin_psi4') as finish, \
                     mock.patch.object(self.wrapper, '_mol2psi4', side_effect=pmols) as mol2psi4, \
                     mock.patch.object(psi4_wrapper.resp, 'resp', new=fake_resp), \
                     mock.patch.object(psi4_wrapper.resp.stage2_helper, 'set_stage2_constraint', side_effect=set_constraints) as stage2_helper, \
                     mock.patch.object(psi4_wrapper.psi4.core, 'set_output_file') as set_output:
                    result = self.wrapper.resp_multi([1, 0], weights=[2.0, 0.5])
            finally:
                os.chdir(original_cwd)

        np.testing.assert_allclose(result, stage2[1])
        prepare.assert_called_once_with()
        configure.assert_called_once_with(output='./multi_test_psi4_multi_resp.log')
        finish.assert_called_once_with()
        self.assertEqual(
            mol2psi4.call_args_list,
            [
                mock.call(confId=1, name='multi_test_conf1'),
                mock.call(confId=0, name='multi_test_conf0'),
            ],
        )
        self.assertEqual(self.wrapper.confId, 1)
        self.assertEqual(len(resp_calls), 2)
        self.assertIs(resp_calls[0][0], resp_calls[1][0])
        self.assertEqual(len(resp_calls[0][0]), 2)
        self.assertEqual(resp_calls[0][1]['WEIGHT'], [2.0, 0.5])
        self.assertEqual(resp_calls[1][1]['WEIGHT'], [2.0, 0.5])
        self.assertEqual(
            resp_calls[1][1]['grid'],
            ['./1_multi_test_conf1_grid.dat', './2_multi_test_conf0_grid.dat'],
        )
        self.assertEqual(
            resp_calls[1][1]['esp'],
            ['./1_multi_test_conf1_grid_esp.dat', './2_multi_test_conf0_grid_esp.dat'],
        )
        self.assertEqual(resp_calls[1][1]['RESP_A'], 0.001)
        self.assertEqual(resp_calls[1][1]['RADIUS'], {})
        stage2_helper.assert_called_once()
        self.assertIs(stage2_helper.call_args.args[0], pmols[0])
        np.testing.assert_allclose(stage2_helper.call_args.args[1], stage1[1])
        set_output.assert_called_once_with('./multi_test_psi4_multi_resp.log', True)
        for i, atom in enumerate(self.wrapper.mol.GetAtoms()):
            self.assertEqual(atom.GetDoubleProp('ESP'), stage2[0, i])
            self.assertEqual(atom.GetDoubleProp('RESP'), stage2[1, i])

    def test_missing_stage_one_file_fails_without_writing_properties(self):
        stage1 = np.array(((-0.3, 0.15, 0.15), (-0.4, 0.2, 0.2)))
        pmols = [FakePsi4Molecule('multi_test_conf0'), FakePsi4Molecule('multi_test_conf1')]

        def fake_resp(molecules, options):
            return stage1

        with mock.patch.object(self.wrapper, '_prepare_psi4'), \
             mock.patch.object(self.wrapper, '_configure_psi4'), \
             mock.patch.object(self.wrapper, '_fin_psi4') as finish, \
             mock.patch.object(self.wrapper, '_mol2psi4', side_effect=pmols), \
             mock.patch.object(psi4_wrapper.resp, 'resp', new=fake_resp), \
             mock.patch.object(psi4_wrapper.resp.stage2_helper, 'set_stage2_constraint'):
            result = self.wrapper.resp_multi([0, 1])

        self.assertTrue(self.wrapper.error_flag)
        self.assertTrue(np.all(np.isnan(result)))
        finish.assert_called_once_with()
        for atom in self.wrapper.mol.GetAtoms():
            self.assertFalse(atom.HasProp('ESP'))
            self.assertFalse(atom.HasProp('RESP'))

    def test_inconsistent_psi4_molecules_fail_before_fitting_and_finalize_once(self):
        def fake_resp(molecules, options):
            raise AssertionError('RESP fitting must not start')

        inconsistent_second_molecules = (
            FakePsi4Molecule('multi_test_conf1', symbols=('O', 'H')),
            FakePsi4Molecule('multi_test_conf1', symbols=('H', 'O', 'H')),
            FakePsi4Molecule('multi_test_conf1', charge=1),
            FakePsi4Molecule('multi_test_conf1', multiplicity=2),
        )
        for second_molecule in inconsistent_second_molecules:
            with self.subTest(second_molecule=second_molecule), \
                 mock.patch.object(self.wrapper, '_prepare_psi4'), \
                 mock.patch.object(self.wrapper, '_configure_psi4') as configure, \
                 mock.patch.object(self.wrapper, '_fin_psi4') as finish, \
                 mock.patch.object(
                     self.wrapper,
                     '_mol2psi4',
                     side_effect=[FakePsi4Molecule('multi_test_conf0'), second_molecule],
                 ), \
                 mock.patch.object(psi4_wrapper.resp, 'resp', new=fake_resp):
                result = self.wrapper.resp_multi([0, 1])

            self.assertTrue(self.wrapper.error_flag)
            self.assertTrue(np.all(np.isnan(result)))
            configure.assert_not_called()
            finish.assert_called_once_with()

    def test_resp_fitting_failure_finalizes_once_without_writing_properties(self):
        pmols = [FakePsi4Molecule('multi_test_conf0'), FakePsi4Molecule('multi_test_conf1')]

        def fake_resp(molecules, options):
            raise RuntimeError('fit failed')

        with mock.patch.object(self.wrapper, '_prepare_psi4'), \
             mock.patch.object(self.wrapper, '_configure_psi4'), \
             mock.patch.object(self.wrapper, '_fin_psi4') as finish, \
             mock.patch.object(self.wrapper, '_mol2psi4', side_effect=pmols), \
             mock.patch.object(psi4_wrapper.resp, 'resp', new=fake_resp):
            result = self.wrapper.resp_multi([0, 1])

        self.assertTrue(self.wrapper.error_flag)
        self.assertTrue(np.all(np.isnan(result)))
        finish.assert_called_once_with()
        for atom in self.wrapper.mol.GetAtoms():
            self.assertFalse(atom.HasProp('ESP'))
            self.assertFalse(atom.HasProp('RESP'))

    def test_direct_method_input_failures_do_not_initialize_psi4(self):
        invalid_cases = (
            (None, None),
            ([], None),
            ([0], None),
            ([0, 9], None),
            ([0, 0], None),
            ([0, '1'], None),
            ([0, False], None),
            ([0, 1], [1.0]),
            ([0, 1], [-1.0, 1.0]),
            ([0, 1], [np.nan, 1.0]),
            ([0, 1], [np.inf, 1.0]),
            ([0, 1], [0.0, 0.0]),
        )
        with mock.patch.object(self.wrapper, '_prepare_psi4') as prepare, \
             mock.patch.object(self.wrapper, '_fin_psi4') as finish:
            for conf_ids, weights in invalid_cases:
                with self.subTest(conf_ids=conf_ids, weights=weights):
                    result = self.wrapper.resp_multi(conf_ids, weights=weights)
                    self.assertTrue(self.wrapper.error_flag)
                    self.assertTrue(np.all(np.isnan(result)))
        prepare.assert_not_called()
        finish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
