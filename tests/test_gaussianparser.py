#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import numpy as np

from nomad.datamodel import EntryArchive
from gaussianparser.gaussian_parser import GaussianParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return GaussianParser()


def test_scf_spinpol(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Al_scf/Al.out', archive, None)

    sec_runs = archive.section_run
    assert len(sec_runs) == 1
    assert sec_runs[0].x_gaussian_program_implementation == 'EM64L-G09RevB.01'
    assert sec_runs[0].x_gaussian_number_of_processors == '8'
    assert sec_runs[0].section_basis_set_atom_centered[0].basis_set_atom_centered_short_name == 'AUG-CC-PVTZ'
    assert len(sec_runs[0].x_gaussian_section_orbital_symmetries[0].x_gaussian_alpha_symmetries) == 50

    sec_methods = sec_runs[0].section_method
    assert len(sec_methods) == 1
    assert len(sec_methods[0].section_XC_functionals) == 1
    assert sec_methods[0].section_XC_functionals[0].XC_functional_name == 'HYB_GGA_XC_B3LYP'

    sec_systems = sec_runs[0].section_system
    assert len(sec_systems) == 1
    assert sec_systems[0].configuration_periodic_dimensions == [False, False, False]
    assert sec_systems[0].x_gaussian_total_charge == -1
    assert sec_systems[0].atom_labels == ['Al']

    sec_sccs = sec_runs[0].section_single_configuration_calculation
    assert len(sec_sccs) == 1
    assert sec_sccs[0].energy_total.magnitude == approx(-1.05675722e-15)
    assert len(sec_sccs[0].x_gaussian_section_hybrid_coeffs) == 1
    assert np.shape(sec_sccs[0].eigenvalues[0].band_energies[0].occupations) == (50,)
    assert np.shape(sec_sccs[0].eigenvalues[0].band_energies[0].value) == (50,)
    assert sec_sccs[0].eigenvalues[0].band_energies[0].occupations[7] == 0
    assert sec_sccs[0].eigenvalues[0].band_energies[0].value[-5].magnitude == approx(4.64011991e-18)
    assert sec_sccs[0].x_gaussian_section_molecular_multipoles[0].x_gaussian_molecular_multipole_values[4] == approx(-9.36527896e-39)
    assert len(sec_sccs[0].section_scf_iteration) == 1
    assert sec_sccs[0].section_scf_iteration[0].x_gaussian_energy_scf.magnitude == approx(-1.05675722e-15)
    assert sec_sccs[0].single_configuration_calculation_converged


def test_scf_multirun(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Al_multistep/m61b5.out', archive, None)

    sec_runs = archive.section_run
    assert len(sec_runs) == 2
    assert len(sec_runs[0].section_single_configuration_calculation) == 6
    assert len(sec_runs[1].section_single_configuration_calculation) == 1
    assert len(sec_runs[0].section_single_configuration_calculation[2].section_scf_iteration) == 11
    assert len(sec_runs[1].section_single_configuration_calculation[0].section_scf_iteration) == 1
    assert len(sec_runs[0].section_system) == 6
    assert len(sec_runs[1].section_system) == 1
    assert len(sec_runs[0].section_method) == 1
    assert len(sec_runs[1].section_method) == 1

    sec_scc = sec_runs[0].section_single_configuration_calculation[4]
    assert sec_scc.atom_forces_raw[0][2].magnitude == approx(-9.69697756e-14)
    assert sec_scc.section_scf_iteration[3].x_gaussian_delta_energy_total_scf_iteration.magnitude == approx(-8.82412332e-27)

    sec_thermo = sec_runs[1].x_gaussian_section_thermochem[0]
    assert sec_thermo.x_gaussian_temperature == approx(298.15)
    assert sec_thermo.x_gaussian_moments[1] == approx(8.59409221e-45)
    assert sec_thermo.x_gaussian_thermal_correction_free_energy == approx(-1.00274129e-19)


def test_mp(parser):
    archive = EntryArchive()
    parser.parse('tests/data/NO_mp/onno.out', archive, None)

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 17
    approx(sec_sccs[0].x_gaussian_section_moller_plesset[0].x_gaussian_mp2_correction_energy.magnitude, -3.17820357e-18)
    approx(sec_sccs[-1].energy_total.magnitude, -1.12849219e-15)
    approx(sec_sccs[3].x_gaussian_section_coupled_cluster[0].x_gaussian_ccsd_correction_energy.magnitude, -3.08257224e-18)


def test_freq(parser):
    archive = EntryArchive()
    parser.parse('tests/data/CHO_freq/prono.out', archive, None)

    sec_runs = archive.section_run
    assert len(sec_runs) == 2

    assert len(sec_runs[1].x_gaussian_section_frequencies) == 1
    assert np.shape(sec_runs[1].x_gaussian_section_frequencies[0].x_gaussian_frequencies) == (33,)
    assert np.shape(sec_runs[1].x_gaussian_section_frequencies[0].x_gaussian_red_masses) == (33,)
    assert np.shape(sec_runs[1].x_gaussian_section_frequencies[0].x_gaussian_normal_mode_values) == (33, 13, 3)
    assert sec_runs[1].x_gaussian_section_frequencies[0].x_gaussian_normal_mode_values[28][6][1] == 0.19
