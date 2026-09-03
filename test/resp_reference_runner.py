"""Run the upstream RESP ethanol reference calculation in an isolated process."""

import os
import tempfile

import numpy as np
import psi4
import resp


GEOMETRY1 = """C    0.00000000  0.00000000  0.00000000
C    1.48805540 -0.00728176  0.39653260
O    2.04971655  1.37648153  0.25604810
H    3.06429978  1.37151670  0.52641124
H    1.58679428 -0.33618761  1.43102358
H    2.03441010 -0.68906454 -0.25521028
H   -0.40814044 -1.00553466  0.10208540
H   -0.54635470  0.68178278  0.65174288
H   -0.09873888  0.32890585 -1.03449097
"""

GEOMETRY2 = """C    0.00000000  0.00000000  0.00000000
C    1.48013500 -0.00724300  0.39442200
O    2.00696300  1.29224100  0.26232800
H    2.91547900  1.25572900  0.50972300
H    1.61500700 -0.32678000  1.45587700
H    2.07197500 -0.68695100 -0.26493400
H   -0.32500012  1.02293415 -0.30034094
H   -0.18892141 -0.68463906 -0.85893815
H   -0.64257065 -0.32709111  0.84987482
"""

REFERENCE1 = np.array([
    -0.149134, 0.274292, -0.630868, 0.377965, -0.011016,
    -0.009444, 0.058576, 0.044797, 0.044831,
])
REFERENCE2 = np.array([
    -0.079853, 0.253918, -0.630868, 0.377965, -0.007711,
    -0.007711, 0.031420, 0.031420, 0.031420,
])


def main():
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as work_dir:
        os.chdir(work_dir)
        try:
            psi4.core.set_output_file('psi4_resp_reference.log', False)
            molecule1 = psi4.geometry(GEOMETRY1)
            molecule1.update_geometry()
            molecule1.set_name('conformer1')
            molecule2 = psi4.geometry(GEOMETRY2)
            molecule2.update_geometry()
            molecule2.set_name('conformer2')
            molecules = [molecule1, molecule2]

            # These options intentionally match upstream test_resp_2. RadonPy's
            # production defaults use VDW_POINT_DENSITY=20.0, so its charges
            # should not be compared directly with this density=1.0 reference.
            options = {
                'VDW_SCALE_FACTORS': [1.4, 1.6, 1.8, 2.0],
                'VDW_POINT_DENSITY': 1.0,
                'RESP_A': 0.0005,
                'RESP_B': 0.1,
                'RESTRAINT': True,
                'IHFREE': False,
                'WEIGHT': [1, 1],
            }
            charges1 = resp.resp(molecules, options)
            options['resp_a'] = 0.001
            resp.set_stage2_constraint(molecules[0], charges1[1], options)
            options['grid'] = [
                '%i_%s_grid.dat' % (i + 1, molecule.name())
                for i, molecule in enumerate(molecules)
            ]
            options['esp'] = [
                '%i_%s_grid_esp.dat' % (i + 1, molecule.name())
                for i, molecule in enumerate(molecules)
            ]
            charges2 = resp.resp(molecules, options)
        finally:
            psi4.core.clean()
            psi4.core.clean_options()
            os.chdir(original_cwd)

    error1 = np.max(np.abs(np.asarray(charges1[1]) - REFERENCE1))
    error2 = np.max(np.abs(np.asarray(charges2[1]) - REFERENCE2))
    print('stage1 max abs error: %.12g' % error1)
    print('stage2 max abs error: %.12g' % error2)
    np.testing.assert_allclose(charges1[1], REFERENCE1, rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(charges2[1], REFERENCE2, rtol=0.0, atol=1.0e-5)


if __name__ == '__main__':
    main()
