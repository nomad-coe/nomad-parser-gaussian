from __future__ import division
from builtins import str
from builtins import range
from builtins import object
import setup_paths
from nomadcore.simple_parser import mainFunction, SimpleMatcher as SM
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.caching_backend import CachingLevel
from nomadcore.unit_conversion.unit_conversion import convert_unit
import os, sys, json, logging
import numpy as np
import ase
import re

############################################################
# This is the parser for the output file of Gaussian.
############################################################

logger = logging.getLogger("nomad.GaussianParser")

# description of the output
mainFileDescription = SM(
    name = 'root',
    weak = True,
    forwardMatch = True, 
    startReStr = "",
    subMatchers = [
        SM(name = 'newRun',
           startReStr = r"\s*Cite this work as:",
#           endReStr = r"\s*Normal termination of Gaussian",
           repeats = True,
           required = True,
           forwardMatch = True,
           fixedStartValues={ 'program_name': 'Gaussian', 'program_basis_set_type': 'gaussians' },
           sections   = ['section_run'],
           subMatchers = [
               SM(name = 'header',
                  startReStr = r"\s*Cite this work as:",
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*Cite this work as:"),
                      SM(r"\s*Gaussian [0-9]+, Revision [A-Za-z0-9.]*,"),
                      SM(r"\s\*\*\*\*\*\*\*\*\*\*\*\**"),
                      SM(r"\s*Gaussian\s*(?P<program_version>[0-9]+):\s*(?P<x_gaussian_program_implementation>[A-Za-z0-9-.]+)\s*(?P<x_gaussian_program_release_date>[0-9][0-9]?\-[A-Z][a-z][a-z]\-[0-9]+)"),
                      SM(r"\s*(?P<x_gaussian_program_execution_date>[0-9][0-9]?\-[A-Z][a-z][a-z]\-[0-9]+)"),
                      ]
             ),
               SM(name = 'globalparams',
                  startReStr = r"\s*%\w*=",
                  subFlags = SM.SubFlags.Unordered,
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*%[Cc]hk=(?P<x_gaussian_chk_file>[A-Za-z0-9.]*)"),
                      SM(r"\s*%[Mm]em=(?P<x_gaussian_memory>[A-Za-z0-9.]*)"),
                      SM(r"\s*%[Nn][Pp]roc=(?P<x_gaussian_number_of_processors>[A-Za-z0-9.]*)")
                      ]
             ),
               SM (name = 'SectionMethod',
               sections = ['section_method'],
                   startReStr = r"\s*#",
                   forwardMatch = True,
                   subMatchers = [
                       SM(r"\s*(?P<x_gaussian_settings>([a-zA-Z0-9-/=(),#*+:]*\s*)+)"),
                       SM(r"\s*(?P<x_gaussian_settings>([a-zA-Z0-9-/=(),#*+:]*\s*)+)")
                       ]
             ),
               SM(name = 'charge_multiplicity_cell_natoms',
               sections = ['section_system'],
		  startReStr = r"\s*Charge =",
                  endReStr = r"\s*Leave Link  101\s*",
                  subFlags = SM.SubFlags.Unordered,
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*Charge =\s*(?P<x_gaussian_total_charge>[-+0-9]*) Multiplicity =\s*(?P<x_gaussian_spin_target_multiplicity>[0-9]*)"),
                      SM(r"\s*(Tv|Tv\s*[0]|TV|TV\s*[0])\s*(?P<x_gaussian_geometry_lattice_vector_x>[0-9.]*)\s+(?P<x_gaussian_geometry_lattice_vector_y>[0-9.]*)\s+(?P<x_gaussian_geometry_lattice_vector_z>[0-9.]*)", repeats = True),
                      SM(r"\s*AtmWgt=\s+(?P<x_gaussian_atomic_masses>[0-9.]+(\s+[0-9.]+)(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?)", repeats = True)
                      ]
             ),
            SM (name = 'SingleConfigurationCalculationWithSystemDescription',
                startReStr = "\s*Standard orientation:",
                repeats = False,
                forwardMatch = True,
                subMatchers = [
                SM (name = 'SingleConfigurationCalculation',
                  startReStr = "\s*Standard orientation:",
                  repeats = True,
                  forwardMatch = True,
                  sections = ['section_single_configuration_calculation'],
                  subMatchers = [
                  SM(name = 'geometry',
                   sections  = ['x_gaussian_section_geometry'],
                   startReStr = r"\s*Standard orientation:",
                   endReStr = r"\s*Rotational constants",
                      subMatchers = [
                      SM(r"\s+[0-9]+\s+(?P<x_gaussian_atomic_number>[0-9]+)\s+[0-9]+\s+(?P<x_gaussian_atom_x_coord__angstrom>[-+0-9EeDd.]+)\s+(?P<x_gaussian_atom_y_coord__angstrom>[-+0-9EeDd.]+)\s+(?P<x_gaussian_atom_z_coord__angstrom>[-+0-9EeDd.]+)",repeats = True),
                      SM(r"\s*Rotational constants")
                    ]
                ),
                    SM(name = 'TotalEnergyScfGaussian',
                    sections  = ['section_scf_iteration'],
                    startReStr = r"\s*Cycle\s+[0-9]+",
                    endReStr = r"\s*Leave Link  502\s*",
                    forwardMatch = True,
                    repeats = True, 
                    subMatchers = [
                    SM(r"\s*Cycle\s+[0-9]+"),
                    SM(r"\s*E=\s*(?P<energy_total_scf_iteration__hartree>[-+0-9.]+)\s*Delta-E=\s*(?P<x_gaussian_delta_energy_total_scf_iteration__hartree>[-+0-9.]+)"),
                    SM(r"\s*(?P<x_gaussian_single_configuration_calculation_converged>SCF Done):\s*[(),A-Za-z0-9-]+\s*=\s*(?P<x_gaussian_energy_total__hartree>[-+0-9.]+)"),
                     SM(r"\s*NFock=\s*[0-9]+\s*Conv=(?P<x_gaussian_energy_error__hartree>[-+0-9EeDd.]+)\s*"),
                     SM(r"\s*KE=\s*(?P<x_gaussian_electronic_kinetic_energy__hartree>[-+0-9EeDd.]+)\s*"),
                    ]
                ), 
                    SM(name = 'RealSpinValue',
                    sections  = ['x_gaussian_section_real_spin_squared'],
                    startReStr = r"\s*Annihilation of the first spin contaminant",
                    forwardMatch = True,
                    repeats = True,
                    subMatchers = [
                     SM(r"\s*Annihilation of the first spin contaminant"),
                     SM(r"\s*[A-Z][*][*][0-9]\s*before annihilation\s*(?P<spin_S2>[0-9.]+),\s*after\s*(?P<x_gaussian_after_annihilation_spin_S2>[0-9.]+)")
                     ]
                ),
                    SM(name = 'PerturbationEnergies',
                    sections = ['x_gaussian_section_moller_plesset'],
                    startReStr = r"\s*E2 =\s*",
                    forwardMatch = True,
                    subMatchers = [
                     SM(r"\s*E2 =\s*(?P<x_gaussian_mp2_correction_energy__hartree>[-+0-9EeDd.]+)\s*EUMP2 =\s*(?P<x_gaussian_mp2_energy__hartree>[-+0-9EeDd.]+)"),
                     SM(r"\s*E3=\s*(?P<x_gaussian_mp3_correction_energy__hartree>[-+0-9EeDd.]+)\s*EUMP3=\s*(?P<x_gaussian_mp3_energy__hartree>[-+0-9EeDd.]+)\s*"),
                     SM(r"\s*E4\(DQ\)=\s*(?P<x_gaussian_mp4dq_correction_energy__hartree>[-+0-9EeDd.]+)\s*UMP4\(DQ\)=\s*(?P<x_gaussian_mp4dq_energy__hartree>[-+0-9EeDd.]+)\s*"),
                     SM(r"\s*E4\(SDQ\)=\s*(?P<x_gaussian_mp4sdq_correction_energy__hartree>[-+0-9EeDd.]+)\s*UMP4\(SDQ\)=\s*(?P<x_gaussian_mp4sdq_energy__hartree>[-+0-9EeDd.]+)"),
                     SM(r"\s*E4\(SDTQ\)=\s*(?P<x_gaussian_mp4sdtq_correction_energy__hartree>[-+0-9EeDd.]+)\s*UMP4\(SDTQ\)=\s*(?P<x_gaussian_mp4sdtq_energy__hartree>[-+0-9EeDd.]+)"),
                     SM(r"\s*DEMP5 =\s*(?P<x_gaussian_mp5_correction_energy__hartree>[-+0-9EeDd.]+)\s*MP5 =\s*(?P<x_gaussian_mp5_energy__hartree>[-+0-9EeDd.]+)"),
                     ]
                ),
                    SM(name = 'CoupledClusterEnergies',
                    sections = ['x_gaussian_section_coupled_cluster'],
                    startReStr = r"\s*CCSD\(T\)\s*",
                    endReStr = r"\s*Population analysis using the SCF density",
                    forwardMatch = True,
                    subMatchers = [
                     SM(r"\s*DE\(Corr\)=\s*(?P<x_gaussian_ccsd_correction_energy__hartree>[-+0-9EeDd.]+)\s*E\(CORR\)=\s*(?P<x_gaussian_ccsd_energy__hartree>[-+0-9EeDd.]+)", repeats = True),
                     SM(r"\s*CCSD\(T\)=\s*(?P<x_gaussian_ccsdt_energy__hartree>[-+0-9EeDd.]+)"),
                     ]
                ),
                    SM(name = 'QuadraticCIEnergies',
                    sections = ['x_gaussian_section_quadratic_ci'],
                    startReStr = r"\s*Quadratic Configuration Interaction\s*",
                    endReStr = r"\s*Population analysis using the SCF density",
                    forwardMatch = True,
                    subMatchers = [
                     SM(r"\s*DE\(Z\)=\s*(?P<x_gaussian_qcisd_correction_energy__hartree>[-+0-9EeDd.]+)\s*E\(Z\)=\s*(?P<x_gaussian_qcisd_energy__hartree>[-+0-9EeDd.]+)", repeats = True),
                     SM(r"\s*DE\(Corr\)=\s*(?P<x_gaussian_qcisd_correction_energy__hartree>[-+0-9EeDd.]+)\s*E\(CORR\)=\s*(?P<x_gaussian_qcisd_energy__hartree>[-+0-9EeDd.]+)", repeats = True),
                     SM(r"\s*QCISD\(T\)=\s*(?P<x_gaussian_qcisdt_energy__hartree>[-+0-9EeDd.]+)"),
                     SM(r"\s*DE5\s*=\s*(?P<x_gaussian_qcisdtq_correction_energy__hartree>[-+0-9EeDd.]+)\s*QCISD\(TQ\)\s*=\s*(?P<x_gaussian_qcisdtq_energy__hartree>[-+0-9EeDd.]+)", repeats = True),
                     ]
                ),
                    SM(name = 'CIEnergies',
                    sections = ['x_gaussian_section_ci'],
                    startReStr = r"\s*Configuration Interaction\s*",
                    endReStr = r"\s*Population analysis using the SCF density",
                    forwardMatch = True,
                    subMatchers = [
                     SM(r"\s*DE\(CI\)=\s*(?P<x_gaussian_ci_correction_energy__hartree>[-+0-9EeDd.]+)\s*E\(CI\)=\s*(?P<x_gaussian_ci_energy__hartree>[-+0-9EeDd.]+)", repeats = True),
                     ]
                ),
                    SM(name = 'SemiempiricalEnergies',
                    sections = ['x_gaussian_section_semiempirical'],
                    startReStr = r"\s*[-A-Z0-9]+\s*calculation of energy[a-zA-Z,. ]+\s*",
                    endReStr = r"\s*Population analysis using the SCF density",
                    forwardMatch = True,
                    subMatchers = [
                     SM(r"\s*(?P<x_gaussian_semiempirical_method>[-A-Z0-9]+\s*calculation of energy[a-zA-Z,. ]+)"),
                     SM(r"\s*It=\s*[0-9]+\s*PL=\s*[-+0-9EeDd.]+\s*DiagD=[A-Z]\s*ESCF=\s*(?P<x_gaussian_semiempirical_energy>[-+0-9.]+)\s*", repeats = True),
                     SM(r"\s*Energy=\s*(?P<x_gaussian_semiempirical_energy_converged>[-+0-9EeDd.]+)"),
                     ]
                ),
                    SM(name = 'MolecularMechanicsEnergies',
                    sections = ['x_gaussian_section_molmech'],
                    startReStr = r"\s*[-A-Z0-9]+\s*calculation of energy[a-zA-Z,. ]+\s*",
                    forwardMatch = False,
                    repeats = True,
                    subMatchers = [
                     SM(r"\s*(?P<x_gaussian_molmech_method>[a-zA-Z0-9]+\s*calculation of energy[a-z,. ]+)"),
                     SM(r"\s*Energy=\s*(?P<x_gaussian_molmech_energy>[-+0-9EeDd.]+)\s*NIter=\s*[0-9.]"),
                     ]
                ),
                  SM(name = 'ExcitedStates',
                   sections  = ['x_gaussian_section_excited_initial'],
                   startReStr = r"\s*Excitation energies and oscillator strengths",
                   forwardMatch = False,
                   repeats = True,
                   subMatchers = [
                    SM(name = 'ExcitedStates',
                    sections = ['x_gaussian_section_excited'],
                    startReStr = r"\s*Excited State\s*",
                    forwardMatch = True,
                    repeats = True,
                    subMatchers = [
                     SM(r"\s*Excited State\s*(?P<x_gaussian_excited_state_number>[0-9]+):\s*[-+0-9A-Za-z.\?]+\s*(?P<x_gaussian_excited_energy__eV>[0-9.]+)\s*eV\s*[0-9.]+\s*nm\s*f=(?P<x_gaussian_excited_oscstrength>[0-9.]+)\s*<[A-Z][*][*][0-9]>=(?P<x_gaussian_excited_spin_squared>[0-9.]+)"),
                     SM(r"\s*(?P<x_gaussian_excited_transition>[0-9A-Z]+\s*->\s*[0-9A-Z]+\s*[-+0-9.]+)", repeats = True),
                     SM(r"\s*This state for optimization|\r?\n"),
                     ]
                    )
                   ]
               ),  
                  SM(name = 'CASSCFStates',
                   sections = ['x_gaussian_section_casscf'],
                   startReStr = r"\s*EIGENVALUES AND\s*",
                   forwardMatch = True,
                   repeats = False,
                   subMatchers = [
                    SM(r"\s*EIGENVALUES AND\s*"),
                    SM(r"\s*\(\s*[0-9]+\)\s*EIGENVALUE\s*(?P<x_gaussian_casscf_energy__hartree>[-+0-9.]+)", repeats = True),
                   ]
               ),
                  SM(name = 'Geometry_optimization',
                  sections  = ['x_gaussian_section_geometry_optimization_info'],
                  startReStr = r"\s*Optimization completed.",
                  forwardMatch = True,
                  subMatchers = [
                  SM(r"\s*(?P<x_gaussian_geometry_optimization_converged>Optimization completed)"),
                  SM(r"\s*(?P<x_gaussian_geometry_optimization_converged>Optimization stopped)"),
                  SM(r"\s+[0-9]+\s+[0-9]+\s+[0-9]+\s+[-+0-9EeDd.]+\s+[-+0-9EeDd.]+\s+[-+0-9EeDd.]+",repeats = True),
                  SM(r"\s*Distance matrix|\s*Rotational constants|\s*Stoichiometry")
                    ]
               ),
                SM(name = 'Orbital symmetries',
                sections = ['x_gaussian_section_orbital_symmetries'],
                startReStr = r"\s+Population analysis",
                subFlags = SM.SubFlags.Sequenced,
                subMatchers = [
                      SM(r"\s*Orbital symmetries"), 
                      SM(r"\s*Alpha Orbitals"),
                      SM(r"\s*Occupied\s+(?P<x_gaussian_alpha_occ_symmetry_values>\((.+)\))?"),
                      SM(r"\s+(?P<x_gaussian_alpha_occ_symmetry_values>\((.+)\)?)", repeats = True),
                      SM(r"\s*Virtual\s+(?P<x_gaussian_alpha_vir_symmetry_values>\((.+)\))?"),
                      SM(r"\s+(?P<x_gaussian_alpha_vir_symmetry_values>\((.+)\)?)", repeats = True),
                      SM(r"\s*Beta Orbitals"),
                      SM(r"\s*Occupied\s+(?P<x_gaussian_beta_occ_symmetry_values>\((.+)\))?"),
                      SM(r"\s+(?P<x_gaussian_beta_occ_symmetry_values>\((.+)\)?)", repeats = True),
                      SM(r"\s*Virtual\s+(?P<x_gaussian_beta_vir_symmetry_values>\((.+)\))?"),
                      SM(r"\s+(?P<x_gaussian_beta_vir_symmetry_values>\((.+)\)?)", repeats = True),
                      ]
             ),
                SM(name = 'Electronicstatesymmetry',
                sections = ['x_gaussian_section_symmetry'],
                startReStr = r"\s*The electronic state is",
                forwardMatch = True,
                subMatchers = [
                      SM(r"\s*The electronic state is\s*(?P<x_gaussian_elstate_symmetry>[A-Z0-9-']+)[.]")
                      ]
             ),
                SM(name = 'Eigenvalues',
                sections = ['x_gaussian_section_eigenvalues'],
                startReStr = r"\s*Alpha  occ. eigenvalues --",
                forwardMatch = True,
                subFlags = SM.SubFlags.Sequenced,
                subMatchers = [
                      SM(r"\s*Alpha  occ. eigenvalues --\s+(?P<x_gaussian_alpha_occ_eigenvalues_values>(.+)?)", repeats = True), 
                      SM(r"\s*Alpha virt. eigenvalues --\s+(?P<x_gaussian_alpha_vir_eigenvalues_values>(.+)?)", repeats = True),
                      SM(r"\s*Beta  occ. eigenvalues --\s+(?P<x_gaussian_beta_occ_eigenvalues_values>(.+)?)", repeats = True),
                      SM(r"\s*Beta virt. eigenvalues --\s+(?P<x_gaussian_beta_vir_eigenvalues_values>(.+)?)", repeats = True),
                      SM(r"\s*- Condensed to atoms (all electrons)"),
                      ]
             ),
                SM(name = 'Multipoles',
                  sections = ['x_gaussian_section_molecular_multipoles'],
                  startReStr = r"\s*Electronic spatial extent",
                  forwardMatch = False,
                  subMatchers = [
                      SM(r"\s*Charge=(?P<charge>\s*[-0-9.]+)"),
                      SM(r"\s*Dipole moment "), 
                      SM(r"\s+\w+=\s+(?P<dipole_moment_x>[-+0-9EeDd.]+)\s+\w+=\s+(?P<dipole_moment_y>[-+0-9EeDd.]+)\s+\w+=\s+(?P<dipole_moment_z>[-+0-9EeDd.]+)"),
                      SM(r"\s*Quadrupole moment"), 
                      SM(r"\s+\w+=\s+(?P<quadrupole_moment_xx>[0-9-.]+)\s+\w+=\s+(?P<quadrupole_moment_yy>[0-9-.]+)\s+\w+=\s+(?P<quadrupole_moment_zz>[0-9-.]+)"), 
                      SM(r"\s+\w+=\s+(?P<quadrupole_moment_xy>[0-9-.]+)\s+\w+=\s+(?P<quadrupole_moment_xz>[0-9-.]+)\s+\w+=\s+(?P<quadrupole_moment_yz>[0-9-.]+)"),
                      SM(r"\s*Traceless Quadrupole moment"),
                      SM(r"\s+\w+=\s+[0-9-.]+\s+\w+=\s+[0-9-.]+\s+\w+=\s+[0-9-.]+"),
                      SM(r"\s+\w+=\s+[0-9-.]+\s+\w+=\s+[0-9-.]+\s+\w+=\s+[0-9-.]+"),
                      SM(r"\s*Octapole moment"),
                      SM(r"\s+\w+=\s+(?P<octapole_moment_xxx>[-+0-9EeDd.]+)\s+\w+=\s+(?P<octapole_moment_yyy>[-+0-9EeDd.]+)\s+\w+=\s+(?P<octapole_moment_zzz>[-+0-9EeDd.]+)\s+\w+=\s+(?P<octapole_moment_xyy>[-+0-9EeDd.]+)"),
                      SM(r"\s+\w+=\s+(?P<octapole_moment_xxy>[-+0-9EeDd.]+)\s+\w+=\s+(?P<octapole_moment_xxz>[-+0-9EeDd.]+)\s+\w+=\s+(?P<octapole_moment_xzz>[-+0-9EeDd.]+)\s+\w+=\s+(?P<octapole_moment_yzz>[-+0-9EeDd.]+)"),
                      SM(r"\s+\w+=\s+(?P<octapole_moment_yyz>[-+0-9EeDd.]+)\s+\w+=\s+(?P<octapole_moment_xyz>[-+0-9EeDd.]+)"),
                      SM(r"\s*Hexadecapole moment"),
                      SM(r"\s+\w+=\s+(?P<hexadecapole_moment_xxxx>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_yyyy>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_zzzz>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_xxxy>[-+0-9EeDd.]+)"),
                      SM(r"\s+\w+=\s+(?P<hexadecapole_moment_xxxz>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_yyyx>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_yyyz>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_zzzx>[-+0-9EeDd.]+)"),
                      SM(r"\s+\w+=\s+(?P<hexadecapole_moment_zzzy>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_xxyy>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_xxzz>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_yyzz>[-+0-9EeDd.]+)"),
                      SM(r"\s+\w+=\s+(?P<hexadecapole_moment_xxyz>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_yyxz>[-+0-9EeDd.]+)\s+\w+=\s+(?P<hexadecapole_moment_zzxy>[-+0-9EeDd.]+)")
                      ]
             ),    
                   SM(name = 'ForcesGaussian',
                   sections  = ['x_gaussian_section_atom_forces'],
                   startReStr = "\s*Center\s+Atomic\s+Forces ",
                   forwardMatch = True,
                   subMatchers = [
                    SM(r"\s*Center\s+Atomic\s+Forces "),
                    SM(r"\s+[0-9]+\s+[0-9]+\s+(?P<x_gaussian_atom_x_force__hartree_bohr_1>[-+0-9EeDd.]+)\s+(?P<x_gaussian_atom_y_force__hartree_bohr_1>[-+0-9EeDd.]+)\s+(?P<x_gaussian_atom_z_force__hartree_bohr_1>[-+0-9EeDd.]+)",repeats = True),
                    SM(r"\s*Cartesian Forces:\s+")
                    ]
                ),
                     SM (name = 'Frequencies',
                     sections = ['x_gaussian_section_frequencies'],
                     startReStr = r"\s*Frequencies --",
                     endReStr = r"\s*- Thermochemistry -",
                     forwardMatch = True,
                     repeats = True,
                     subFlags = SM.SubFlags.Unordered,
                     subMatchers = [
                          SM(r"\s*Frequencies --\s+(?P<x_gaussian_frequency_values>([-]?[0-9]+\.\d*)\s*([-]?[-0-9]+\.\d*)?\s*([-]?[-0-9]+\.\d*)?)", repeats = True),
                          SM(r"\s*Red. masses --\s+(?P<x_gaussian_reduced_masses>(.+))", repeats = True),
                          SM(r"\s*[0-9]+\s*[0-9]+\s*(?P<x_gaussian_normal_modes>([-0-9.]+)\s*([-0-9.]+)\s*([-0-9.]+)\s*([-0-9.]+)\s*([-0-9.]+)\s*([-0-9.]+)\s*([-0-9.]+)\s*([-0-9.]+)\s*([-0-9.]+))", repeats = True),
                          SM(r"\s*[0-9]+\s*([0-9]+)?\s*([0-9]+)?"),
                      ]
             ),
                SM(name = 'Thermochemistry',
                sections = ['x_gaussian_section_thermochem'],
                startReStr = r"\s*Temperature",
                forwardMatch = True,
                subMatchers = [
                      SM(r"\s*Temperature\s*(?P<x_gaussian_temperature>[0-9.]+)\s*Kelvin.\s*Pressure\s*(?P<x_gaussian_pressure__atmosphere>[0-9.]+)\s*Atm."),
                      SM(r"\s*Principal axes and moments of inertia in atomic units:"),
                      SM(r"\s*Eigenvalues --\s*(?P<x_gaussian_moment_of_inertia_X__amu_angstrom_angstrom>[0-9.]+)\s*(?P<x_gaussian_moment_of_inertia_Y__amu_angstrom_angstrom>[0-9.]+)\s*(?P<x_gaussian_moment_of_inertia_Z__amu_angstrom_angstrom>[0-9.]+)"),
                      SM(r"\s*Zero-point correction=\s*(?P<x_gaussian_zero_point_energy__hartree>[0-9.]+)"),
                      SM(r"\s*Thermal correction to Energy=\s*(?P<x_gaussian_thermal_correction_energy__hartree>[0-9.]+)"),
                      SM(r"\s*Thermal correction to Enthalpy=\s*(?P<x_gaussian_thermal_correction_enthalpy__hartree>[0-9.]+)"),
                      SM(r"\s*Thermal correction to Gibbs Free Energy=\s*(?P<x_gaussian_thermal_correction_free_energy__hartree>[0-9.]+)"), 
                      ]
             ),       
                SM(name = 'Forceconstantmatrix',
                sections = ['x_gaussian_section_force_constant_matrix'],
                startReStr = r"\s*Force constants in Cartesian coordinates",
                forwardMatch = True,
                subMatchers = [
                      SM(r"\s*Force constants in Cartesian coordinates"),
                      SM(r"\s*[0-9]+\s*(?P<x_gaussian_force_constants>(\-?\d+\.\d*[-+D0-9]+)\s*(\-?\d+\.\d*[-+D0-9]+)?\s*(\-?\d+\.\d*[-+D0-9]+)?\s*(\-?\d+\.\d*[-+D0-9]+)?\s*(\-?\d+\.\d*[-+D0-9]+)?)", repeats = True),
                      SM(r"\s*Force constants in internal coordinates")
                      ]
             ),
                SM(name = 'CompositeModelEnergies',
                sections = ['x_gaussian_section_models'],
                startReStr = r"\s*Diagonal vibrational polarizability\s*",
                forwardMatch = False,
                repeats = True,
                subMatchers = [
                 SM(r"\s*G1 Energy=\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*G2 Energy=\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*G2MP2 Energy=\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*G3 Energy=\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*G3MP2 Energy=\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*CBS-4 Energy=\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*CBS-q Energy=\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*CBS-Q Energy=\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*CBS-QB3 Energy=\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*W1U   Electronic Energy\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*W1RO  Electronic Energy\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                 SM(r"\s*W1BD  Electronic Energy\s*(?P<x_gaussian_model_energy__hartree>[-+0-9.]+)"),
                       ]
             ),
                SM(name = 'run times',
                  sections = ['x_gaussian_section_times'],
                  startReStr = r"\s*Job cpu time:",
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*Job cpu time:\s*(?P<x_gaussian_program_cpu_time>\s*[0-9]+\s*[a-z]+\s*[0-9]+\s*[a-z]+\s*[0-9]+\s*[a-z]+\s*[0-9.]+\s*[a-z]+)"),
                      SM(r"\s*Normal termination of Gaussian\s*[0-9]+\s* at \s*(?P<x_gaussian_program_termination_date>[A-Za-z]+\s*[A-Za-z]+\s*[0-9]+\s*[0-9:]+\s*[0-9]+)"),
                      ]
             )
          ])
        ])
      ])
    ])

# loading metadata from nomad-meta-info/meta_info/nomad_meta_info/gaussian.nomadmetainfo.json
metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/gaussian.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath = metaInfoPath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)

parserInfo = {
  "name": "parser_gaussian",
  "version": "1.0"
}

class GaussianParserContext(object):
      """Context for parsing Gaussian output file.

        This class keeps tracks of several Gaussian settings to adjust the parsing to them.
        The onClose_ functions allow processing and writing of cached values after a section is closed.
        They take the following arguments:
        backend: Class that takes care of writing and caching of metadata.
        gIndex: Index of the section that is closed.
        section: The cached values and sections that were found in the section that is closed.
      """
      def __init__(self):
        # dictionary of energy values, which are tracked between SCF iterations and written after convergence
        self.totalEnergyList = {
                                'x_gaussian_electronic_kinetic_energy': None,
                                'x_gaussian_energy_electrostatic': None,
                                'x_gaussian_energy_error': None,
                               }

      def initialize_values(self):
        """Initializes the values of certain variables.

        This allows a consistent setting and resetting of the variables,
        when the parsing starts and when a section_run closes.
        """
        self.secMethodIndex = None
        self.secSystemDescriptionIndex = None
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        self.singleConfCalcs = []
        self.scfConvergence = False
        self.geoConvergence = False
        self.scfenergyconverged = 0.0
        self.scfkineticenergyconverged = 0.0
        self.scfelectrostaticenergy = 0.0
        self.periodicCalc = False

      def startedParsing(self, path, parser):
        self.parser = parser
        # save metadata
        self.metaInfoEnv = self.parser.parserBuilder.metaInfoEnv
        # allows to reset values if the same superContext is used to parse different files
        self.initialize_values()

      def onClose_section_run(self, backend, gIndex, section):
          """Trigger called when section_run is closed.

          Write convergence of geometry optimization.
          Variables are reset to ensure clean start for new run.
          """
          # write geometry optimization convergence
          sampling_method = None
          gIndexTmp = backend.openSection('section_frame_sequence')
          backend.addValue('geometry_optimization_converged', self.geoConvergence)
          backend.closeSection('section_frame_sequence', gIndexTmp)
          # frame sequence
          if self.geoConvergence:
              sampling_method = "geometry_optimization"
          elif len(self.singleConfCalcs) > 1:
              pass # to do
          else:
              return
          samplingGIndex = backend.openSection("section_sampling_method")
          backend.addValue("sampling_method", sampling_method)
          backend.closeSection("section_sampling_method", samplingGIndex)
          frameSequenceGIndex = backend.openSection("section_frame_sequence")
          backend.addValue("frame_sequence_to_sampling_ref", samplingGIndex)
          backend.addArrayValues("frame_sequence_local_frames_ref", np.asarray(self.singleConfCalcs))
          backend.closeSection("section_frame_sequence", frameSequenceGIndex)
          # reset all variables
          self.initialize_values()

      def onClose_x_gaussian_section_geometry(self, backend, gIndex, section):
        xCoord = section["x_gaussian_atom_x_coord"]
        yCoord = section["x_gaussian_atom_y_coord"]
        zCoord = section["x_gaussian_atom_z_coord"]
        numbers = section["x_gaussian_atomic_number"]
        atom_coords = np.zeros((len(xCoord),3), dtype=float)
        atom_numbers = np.zeros(len(xCoord), dtype=int)
        atomic_symbols = np.empty((len(xCoord)), dtype=object)
        for i in range(len(xCoord)):
           atom_coords[i,0] = xCoord[i]
           atom_coords[i,1] = yCoord[i]
           atom_coords[i,2] = zCoord[i]
        for i in range(len(xCoord)):
          atom_numbers[i] = numbers[i]
          atomic_symbols[i] = ase.data.chemical_symbols[atom_numbers[i]]
        gIndexTmp = backend.openSection("section_system")
        backend.addArrayValues("atom_labels", atomic_symbols)
        backend.addArrayValues("atom_positions", atom_coords)
        backend.closeSection("section_system", gIndexTmp)

      def onClose_x_gaussian_section_atom_forces(self, backend, gIndex, section):
        xForce = section["x_gaussian_atom_x_force"]
        yForce = section["x_gaussian_atom_y_force"]
        zForce = section["x_gaussian_atom_z_force"]
        atom_forces = np.zeros((len(xForce),3), dtype=float)
        for i in range(len(xForce)):
           atom_forces[i,0] = xForce[i]
           atom_forces[i,1] = yForce[i]
           atom_forces[i,2] = zForce[i]
        backend.addArrayValues("atom_forces_raw", atom_forces)

      def onOpen_section_single_configuration_calculation(self, backend, gIndex, section):
          self.singleConfCalcs.append(gIndex)

      def onClose_section_single_configuration_calculation(self, backend, gIndex, section):
        """Trigger called when section_single_configuration_calculation is closed.
         Write number of SCF iterations and convergence.
         Check for convergence of geometry optimization.
        """
        # write SCF convergence and reset
<<<<<<< HEAD
        backend.addValue('x_gaussian_single_configuration_calculation_converged', self.scfConvergence)
        backend.addValue('energy_total', self.scfenergyconverged[-1])
=======
        backend.addValue('single_configuration_calculation_converged', self.scfConvergence)
>>>>>>> Gaussian parser with ground and excited state energies
        self.scfConvergence = False
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        # write the references to section_method and section_system
        backend.addValue('single_configuration_to_calculation_method_ref', self.secMethodIndex)
        backend.addValue('single_configuration_calculation_to_system_ref', self.secSystemDescriptionIndex)

      def onClose_x_gaussian_section_geometry_optimization_info(self, backend, gIndex, section):
        # check for geometry optimization convergence
        if section['x_gaussian_geometry_optimization_converged'] is not None:
           if section['x_gaussian_geometry_optimization_converged'] == ['Optimization completed']:
              self.geoConvergence = True
           elif section['x_gaussian_geometry_optimization_converged'] == ['Optimization stopped']:
              self.geoConvergence = False

      def onClose_section_scf_iteration(self, backend, gIndex, section):
        # count number of SCF iterations
        self.scfIterNr += 1
        # check for SCF convergence
        if section['x_gaussian_single_configuration_calculation_converged'] is not None:
           self.scfConvergence = True
           if section['x_gaussian_energy_total']:
              if section['x_gaussian_electronic_kinetic_energy']:
                 self.scfkineticenergyconverged = float(str(section['x_gaussian_electronic_kinetic_energy']).replace("[","").replace("]","").replace("D","E"))
                 self.scfelectrostaticenergy = self.scfenergyconverged - self.scfkineticenergyconverged
                 backend.addValue('x_gaussian_energy_electrostatic', self.scfelectrostaticenergy)

      def onClose_x_gaussian_section_eigenvalues(self, backend, gIndex, section):
          eigenenergies = str(section["x_gaussian_alpha_occ_eigenvalues_values"])
          eigenen1 = []
          energy = [float(f) for f in eigenenergies[1:].replace("'","").replace(",","").replace("]","").replace("one","").replace(" ."," 0.").replace(" -."," -0.").split()]
          eigenen1 = np.append(eigenen1, energy)
          if(section["x_gaussian_beta_occ_eigenvalues_values"]):
             occoccupationsalp = np.ones(len(eigenen1), dtype=float)
          else:
             occoccupationsalp = 2.0 * np.ones(len(eigenen1), dtype=float)

          eigenenergies = str(section["x_gaussian_alpha_vir_eigenvalues_values"])
          eigenen2 = []
          energy = [float(f) for f in eigenenergies[1:].replace("'","").replace(",","").replace("]","").replace("one","").replace(" ."," 0.").replace(" -."," -0.").split()]
          eigenen2 = np.append(eigenen2, energy)
          viroccupationsalp = np.zeros(len(eigenen2), dtype=float)
          eigenencon = np.zeros(len(eigenen1) + len(eigenen2))
          eigenencon = np.concatenate((eigenen1,eigenen2), axis=0)
          eigenencon = convert_unit(eigenencon, "hartree", "J")
          occupcon = np.concatenate((occoccupationsalp, viroccupationsalp), axis=0)
          backend.addArrayValues("x_gaussian_alpha_eigenvalues", eigenencon)
          backend.addArrayValues("x_gaussian_alpha_occupations", occupcon)

          if(section["x_gaussian_beta_occ_eigenvalues_values"]):
             eigenenergies = str(section["x_gaussian_beta_occ_eigenvalues_values"])
             eigenen1 = []
             energy = [float(f) for f in eigenenergies[1:].replace("'","").replace(",","").replace("]","").replace("one","").replace(" ."," 0.").replace(" -."," -0.").split()]
             eigenen1 = np.append(eigenen1, energy)
             occoccupationsbet = np.ones(len(eigenen1), dtype=float)
             eigenenergies = str(section["x_gaussian_beta_vir_eigenvalues_values"])
             eigenen2 = []
             energy = [float(f) for f in eigenenergies[1:].replace("'","").replace(",","").replace("]","").replace("one","").replace(" ."," 0.").replace(" -."," -0.").split()]
             eigenen2 = np.append(eigenen2, energy)
             viroccupationsbet = np.zeros(len(eigenen2), dtype=float)
             eigenencon = np.zeros(len(eigenen1) + len(eigenen2))
             eigenencon = np.concatenate((eigenen1,eigenen2), axis=0)
             eigenencon = convert_unit(eigenencon, "hartree", "J")
             occupcon = np.concatenate((occoccupationsbet, viroccupationsbet), axis=0)
             backend.addArrayValues("x_gaussian_beta_eigenvalues", eigenencon)
             backend.addArrayValues("x_gaussian_beta_occupations", occupcon)

      def onClose_x_gaussian_section_orbital_symmetries(self, backend, gIndex, section):
          symoccalpha = str(section["x_gaussian_alpha_occ_symmetry_values"])
          symviralpha = str(section["x_gaussian_alpha_vir_symmetry_values"])
          if(section["x_gaussian_beta_occ_symmetry_values"]):
             symoccbeta = str(section["x_gaussian_beta_occ_symmetry_values"])
             symvirbeta = str(section["x_gaussian_beta_vir_symmetry_values"])

          symmetry = [str(f) for f in symoccalpha[1:].replace(",","").replace("(","").replace(")","").replace("]","").replace("'A","A").replace("\\'","'").replace("A''","A'").replace("'E","E").replace("G'","G").replace("\"A'\"","A'").split()]
          sym1 = []
          sym1 = np.append(sym1, symmetry)  
          symmetry = [str(f) for f in symviralpha[1:].replace(",","").replace("(","").replace(")","").replace("]","").replace("'A","A").replace("\\'","'").replace("A''","A'").replace("\"A'\"","A'").replace("'E","E").replace("G'","G").split()]
          sym2 = []
          sym2 = np.append(sym2, symmetry)
          symmetrycon = np.concatenate((sym1, sym2), axis=0)
          backend.addArrayValues("x_gaussian_alpha_symmetries", symmetrycon) 

          if(section["x_gaussian_beta_occ_symmetry_values"]):
             symmetry = [str(f) for f in symoccbeta[1:].replace(",","").replace("(","").replace(")","").replace("]","").replace("'A","A").replace("\\'","'").replace("A''","A'").replace("\"A'\"","A'").replace("'E","E").replace("G'","G").split()]
             sym1 = []
             sym1 = np.append(sym1, symmetry)
             symmetry = [str(f) for f in symvirbeta[1:].replace(",","").replace("(","").replace(")","").replace("]","").replace("'A","A").replace("\\'","'").replace("A''","A'").replace("\"A'\"","A'").replace("'E","E").replace("G'","G").split()]
             sym2 = []
             sym2 = np.append(sym2, symmetry)
             symmetrycon = np.concatenate((sym1, sym2), axis=0)
             backend.addArrayValues("x_gaussian_beta_symmetries", symmetrycon)

      def onClose_x_gaussian_section_molecular_multipoles(self, backend, gIndex, section):
          if(section["quadrupole_moment_xx"]):
             x_gaussian_number_of_lm_molecular_multipoles = 35
          else:
             x_gaussian_number_of_lm_molecular_multipoles = 4

          x_gaussian_molecular_multipole_m_kind = 'polynomial'

          char = str(section["charge"])
          cha = str([char])
          charge = [float(f) for f in cha[1:].replace("-."," -0.").replace("'."," 0.").replace("'","").replace("[","").replace("]","").replace(",","").replace('"','').split()]

          dipx = section["dipole_moment_x"]
          dipy = section["dipole_moment_y"]
          dipz = section["dipole_moment_z"]
          dip = str([dipx, dipy, dipz])
          dipoles = [float(f) for f in dip[1:].replace("-."," -0.").replace("'."," 0.").replace("'","").replace("[","").replace("]","").replace(",","").split()]
          dipoles = convert_unit(dipoles, "debye", "coulomb * meter")

          quadxx = section["quadrupole_moment_xx"]
          quadxy = section["quadrupole_moment_xy"]
          quadyy = section["quadrupole_moment_yy"]
          quadxz = section["quadrupole_moment_xz"]
          quadyz = section["quadrupole_moment_yz"]
          quadzz = section["quadrupole_moment_zz"]
          quad = str([quadxx, quadxy, quadyy, quadxz, quadyz, quadzz])
          quadrupoles = [float(f) for f in quad[1:].replace("-."," -0.").replace("'."," 0.").replace("'","").replace("[","").replace("]","").replace(",","").split()] 
          if(section["quadrupole_moment_xx"]): 
             quadrupoles = convert_unit(quadrupoles, "debye * angstrom", "coulomb * meter**2")

          octaxxx = section["octapole_moment_xxx"]
          octayyy = section["octapole_moment_yyy"]
          octazzz = section["octapole_moment_zzz"]
          octaxyy = section["octapole_moment_xyy"]
          octaxxy = section["octapole_moment_xxy"]
          octaxxz = section["octapole_moment_xxz"]
          octaxzz = section["octapole_moment_xzz"]
          octayzz = section["octapole_moment_yzz"]
          octayyz = section["octapole_moment_yyz"]
          octaxyz = section["octapole_moment_xyz"]
          octa = str([octaxxx, octayyy, octazzz, octaxyy, octaxxy, octaxxz, octaxzz, octayzz, octayyz, octaxyz])
          octapoles = [float(f) for f in octa[1:].replace("-."," -0.").replace("'."," 0.").replace("'","").replace("[","").replace("]","").replace(",","").split()]
          if(section["octapole_moment_xxx"]):
             octapoles = convert_unit(octapoles, "debye * angstrom**2", "coulomb * meter**3")

          hexadecaxxxx = section["hexadecapole_moment_xxxx"]
          hexadecayyyy = section["hexadecapole_moment_yyyy"]
          hexadecazzzz = section["hexadecapole_moment_zzzz"]
          hexadecaxxxy = section["hexadecapole_moment_xxxy"]
          hexadecaxxxz = section["hexadecapole_moment_xxxz"]
          hexadecayyyx = section["hexadecapole_moment_yyyx"]
          hexadecayyyz = section["hexadecapole_moment_yyyz"]
          hexadecazzzx = section["hexadecapole_moment_zzzx"]
          hexadecazzzy = section["hexadecapole_moment_zzzy"]
          hexadecaxxyy = section["hexadecapole_moment_xxyy"]
          hexadecaxxzz = section["hexadecapole_moment_xxzz"]
          hexadecayyzz = section["hexadecapole_moment_yyzz"]
          hexadecaxxyz = section["hexadecapole_moment_xxyz"]
          hexadecayyxz = section["hexadecapole_moment_yyxz"]
          hexadecazzxy = section["hexadecapole_moment_zzxy"]
          hexa = str([hexadecaxxxx, hexadecayyyy, hexadecazzzz, hexadecaxxxy, hexadecaxxxz, hexadecayyyx, hexadecayyyz,
          hexadecazzzx, hexadecazzzy, hexadecaxxyy, hexadecaxxzz, hexadecayyzz, hexadecaxxyz, hexadecayyxz, hexadecazzxy])
          hexadecapoles = [float(f) for f in hexa[1:].replace("-."," -0.").replace("'."," 0.").replace("'","").replace("[","").replace("]","").replace(",","").split()]
          if(section["hexadecapole_moment_xxxx"]):
             hexadecapoles = convert_unit(hexadecapoles, "debye * angstrom**3", "coulomb * meter**4")

          if(section["quadrupole_moment_xx"]):
             multipoles = np.hstack((charge, dipoles, quadrupoles, octapoles, hexadecapoles))
          else:
             multipoles = np.hstack((charge, dipoles)) 

          x_gaussian_molecular_multipole_values = np.resize(multipoles, (x_gaussian_number_of_lm_molecular_multipoles))

          backend.addArrayValues("x_gaussian_molecular_multipole_values", x_gaussian_molecular_multipole_values)
          backend.addValue("x_gaussian_molecular_multipole_m_kind", x_gaussian_molecular_multipole_m_kind)

      def onClose_x_gaussian_section_frequencies(self, backend, gIndex, section):
          frequencies = str(section["x_gaussian_frequency_values"])
          vibfreqs = []
          freqs = [float(f) for f in frequencies[1:].replace("'","").replace(",","").replace("]","").replace("one","").replace("\\n","").replace(" ."," 0.").replace(" -."," -0.").split()]
          vibfreqs = np.append(vibfreqs, freqs)
          vibfreqs = convert_unit(vibfreqs, "inversecm", "J")
          backend.addArrayValues("x_gaussian_frequencies", vibfreqs)

          masses = str(section["x_gaussian_reduced_masses"])
          vibreducedmasses = []
          reduced = [float(f) for f in masses[1:].replace("'","").replace(",","").replace("]","").replace("one","").replace(" ."," 0.").split()]
          vibreducedmasses = np.append(vibreducedmasses, reduced)
          vibreducedmasses = convert_unit(vibreducedmasses, "amu", "kilogram")
          backend.addArrayValues("x_gaussian_red_masses", vibreducedmasses)

          vibnormalmodes = []
          vibdisps = str(section["x_gaussian_normal_modes"])
          disps = [float(s) for s in vibdisps[1:].replace("'","").replace(",","").replace("]","").replace("one","").replace("\\n","").replace(" ."," 0.").replace(" -."," -0.").split()]
          dispsnew = np.zeros(len(disps), dtype = float)

#  Reorder disps 

          if len(vibfreqs) % 3 == 0:
             k = 0
             for p in range(0,len(vibfreqs) // 3):
                M = int(len(disps)/len(vibfreqs)) * (p+1) 
                for m in range(3):
                  for n in range(M - int(len(disps) / len(vibfreqs)),M,3):
                    for l in range(3):
                      dispsnew[k] = disps[3*(n + m) + l]
                      k = k + 1
          elif len(vibfreqs) % 3 != 0:
             k = 0
             for p in range(len(vibfreqs)-1,0,-3):
                M = (len(disps) - int(len(disps) / len(vibfreqs))) // p
                for m in range(3):
                  for n in range(M - int(len(disps) / len(vibfreqs)),M,3):
                    for l in range(3):
                      dispsnew[k] = disps[3*(n + m) + l]
                      k = k + 1
             for m in range(int(len(disps) / len(vibfreqs))):
                   dispsnew[k] = disps[k]
                   k = k + 1

          vibnormalmodes = np.append(vibnormalmodes, dispsnew)
          natoms = int(len(disps) / len(vibfreqs) / 3)
          vibnormalmodes = np.reshape(vibnormalmodes,(len(vibfreqs),natoms,3))
          backend.addArrayValues("x_gaussian_normal_mode_values", vibnormalmodes)

      def onClose_x_gaussian_section_force_constant_matrix(self, backend, gIndex, section):

          forcecnstvalues = []
          forceconst = str(section["x_gaussian_force_constants"])
          numbers = [float(s) for s in forceconst[1:].replace("'","").replace(",","").replace("]","").replace("\\n","").replace("D","E").replace(" ."," 0.").replace(" -."," -0.").split()]
          length = len(numbers)
          dim = int(((1 + 8 * length)**0.5 - 1) / 2) 
          cartforceconst = np.zeros([dim, dim])
          forcecnstvalues = np.append(forcecnstvalues, numbers) 
          if dim > 6:
             l = 0
             for i in range(0,5):
                for k in range(0,i+1):
                   l = l + 1
                   cartforceconst[i,k] = forcecnstvalues[l-1]
             for i in range(5,dim):
                for k in range(0,5):
                   l = l + 1
                   cartforceconst[i,k] = forcecnstvalues[l-1]
             for i in range(5,dim-2): 
                for k in range(5,i+1):
                   l = l + 1
                   cartforceconst[i,k] = forcecnstvalues[l-1]
             for i in range(dim-2,dim):
                for k in range(5,dim-2):
                   l = l + 1
                   cartforceconst[i,k] = forcecnstvalues[l-1]
             for i in range(dim-2,dim):
                for k in range(i,dim):
                   l = l + 1
                   cartforceconst[i,k] = forcecnstvalues[l-1]
          elif dim == 6:
             l = 0
             for i in range(0,5):
                for k in range(0,i+1):
                   l = l + 1
                   cartforceconst[i,k] = forcecnstvalues[l-1]
             for i in range(5,dim):
                for k in range(0,5):
                   l = l + 1
                   cartforceconst[i,k] = forcecnstvalues[l-1]
             for i in range(dim,dim):
                for k in range(i,dim):
                   l = l + 1
                   cartforceconst[i,k] = forcecnstvalues[l-1]
 
          for i in range(0,dim):
             for k in range(i+1,dim):
                 cartforceconst[i,k] = cartforceconst[k,i]

          cartforceconst = convert_unit(cartforceconst, "forceAu / bohr", "J / (meter**2)")

          backend.addArrayValues("x_gaussian_force_constant_values", cartforceconst) 

      def onOpen_section_method(self, backend, gIndex, section):
        # keep track of the latest method section
        self.secMethodIndex = gIndex

      def onClose_section_method(self, backend, gIndex, section):
        # handling of xc functional
        # two functions to convert hybrid_xc_coeff to the correct weight
        def GGA_weight(x):
          return 1.0 - x
        def HF_weight(x):
          return x
       # TODO vdW functionals and double-hybrid functionals
       # Dictionary for conversion of xc functional name in aims to metadata format.
       # The individual x and c components of the functional are given as dictionaries.
       # Possible keys of such a dictionary are 'name', 'weight', and 'convert'.
       # If 'weight' is not given it is not written.
       # With 'convert', a funtion is specified how hybrid_xc_coeff is converted to the correct weight for this xc component.
        xcDict = {
              'S':           [{'name': 'LDA_X'}],
              'XA':	     [{'name': 'X_ALPHA'}],
              'VWN5':        [{'name': 'LDA_C_VWN'}],
              'VWN':         [{'name': 'LDA_C_VWN_3'}],
              'LSDA':        [{'name': 'LDA_X'}, {'name': 'LDA_C_VWN'}], 
              'B':           [{'name': 'GGA_X_B88'}],
              'BLYP':        [{'name': 'GGA_C_LYP'}, {'name': 'GGA_X_B88'}],
              'PBE':         [{'name': 'GGA_C_PBE'}],
              'PBEPBE':      [{'name': 'GGA_C_PBE'}, {'name': 'GGA_X_PBE'}],
              'PBEH':        [{'name': 'GGA_X_PBEH'}],
              'WPBEH':       [{'name': 'GGA_X_WPBEH'}],
              'PW91PW91':    [{'name': 'GGA_C_PW91'}, {'name': 'GGA_X_PW91'}],
              'M06L':        [{'name': 'MGGA_C_M06_L'}, {'name': 'MGGA_X_M06_L'}],
              'M11L':        [{'name': 'MGGA_C_M11_L'}, {'name': 'MGGA_X_M11_L'}],
              'SOGGA11':     [{'name': 'GGA_XC_SOGGA11'}],
              'MN12L':       [{'name': 'GGA_XC_MN12L'}],
              'N12':         [{'name': 'GGA_C_N12'}, {'name': 'GGA_X_N12'}],
              'VSXC':        [{'name': 'GGA_XC_VSXC'}],
              'HCTH93':      [{'name': 'GGA_XC_HCTH_93'}],
              'HCTH147':     [{'name': 'GGA_XC_HCTH_147'}],
              'HCTH407':     [{'name': 'GGA_XC_HCTH_407'}],
              'HCTH':        [{'name': 'GGA_XC_HCTH_407'}],
              'B97D':        [{'name': 'GGA_XC_B97D'}],
              'B97D3':       [{'name': 'GGA_XC_B97D3'}],
              'MPW':         [{'name': 'GGA_X_MPW'}],
              'G96':         [{'name': 'GGA_X_G96'}],
              'O':           [{'name': 'GGA_X_O'}],
              'BRX':         [{'name': 'GGA_X_BRX'}],
              'PKZB':        [{'name': 'GGA_C_PKZB'}, {'name': 'GGA_X_PKZB'}],
              'PL':          [{'name': 'GGA_C_PL'}],
              'P86':         [{'name': 'GGA_C_P86'}],
              'B95':         [{'name': 'GGA_C_B95'}],
              'KCIS':        [{'name': 'GGA_C_KCIS'}],
              'BRC':         [{'name': 'GGA_C_BRC'}],
              'VP86':        [{'name': 'GGA_C_VP86'}],
              'V5LYP':       [{'name': 'GGA_C_V5LYP'}],
              'tHCTH':       [{'name': 'MGGA_XC_TAU_HCTH'}],
              'TPSSTPSS':    [{'name': 'MGGA_C_TPSS'}, {'name': 'MGGA_X_TPSS'}],
              'B3LYP':       [{'name': 'HYB_GGA_XC_B3LYP'}], 
              'B3PW91':      [{'name': 'HYB_GGA_XC_B3PW91'}],
              'B3P86':       [{'name': 'HYB_GGA_XC_B3P86'}], 
              'B1B95':       [{'name': 'HYB_GGA_XC_B1B95'}],
              'MPW1PW91':    [{'name': 'HYB_GGA_XC_MPW1PW91'}],
              'MPW1LYP':     [{'name': 'HYB_GGA_XC_MPW1LYP'}],
              'MPW1PBE':     [{'name': 'HYB_GGA_XC_MPW1PBE'}],
              'MPW3PBE':     [{'name': 'HYB_GGA_XC_MPW3PBE'}],
              'B98':         [{'name': 'HYB_GGA_XC_B98'}],
              'B971':        [{'name': 'HYB_GGA_XC_B971'}],
              'B972':        [{'name': 'HYB_GGA_XC_B972'}],
              'O3LYP':       [{'name': 'HYB_GGA_XC_O3LYP'}], 
              'TPSSh':       [{'name': 'HYB_GGA_XC_TPSSh'}],
              'BMK':         [{'name': 'HYB_GGA_XC_BMK'}],
              'X3LYP':       [{'name': 'HYB_GGA_XC_X3LYP'}],
              'tHCTHhyb':    [{'name': 'HYB_GGA_XC_tHCTHHYB'}],
              'BHANDH':      [{'name': 'HYB_GGA_XC_BHANDH'}],
              'BHANDHLYP':   [{'name': 'HYB_GGA_XC_BHANDHLYP'}],
              'APF':         [{'name': 'HYB_GGA_XC_APF'}],
              'APFD':        [{'name': 'HYB_GGA_XC_APFD'}],
              'B97D':        [{'name': 'HYB_GGA_XC_B97D'}],
              'RHF':         [{'name': 'RHF_X'}],
              'UHF':         [{'name': 'UHF_X'}],
              'ROHF':        [{'name': 'ROHF_X'}],
              'OHSE2PBE':    [{'name': 'HYB_GGA_XC_HSE03'}],
              'HSEH1PBE':    [{'name': 'HYB_GGA_XC_HSE06'}],
              'OHSE1PBE':    [{'name': 'HYB_GGA_XC_HSEOLD'}],
              'PBEH1PBE':    [{'name': 'HYB_GGA_XC_PBEh1PBE'}],
              'PBE1PBE':     [{'name': 'GGA_C_PBE'}, {'name': 'GGA_X_PBE', 'weight': 0.75, 'convert': GGA_weight}, {'name': 'HF_X', 'weight': 0.25, 'convert': HF_weight}],
              'M05':         [{'name': 'HYB_MGGA_XC_M05'}],
              'M052X':       [{'name': 'HYB_MGGA_XC_M05_2X'}],
              'M06':         [{'name': 'HYB_MGGA_XC_M06'}],
              'M062X':       [{'name': 'HYB_MGGA_XC_M06_2X'}],
              'M06HF':       [{'name': 'HYB_MGGA_XC_M06_HF'}],
              'M11':         [{'name': 'HYB_MGGA_XC_M11'}],
              'SOGGAX11':    [{'name': 'HYB_MGGA_XC_SOGGA11_X'}],
              'MN12SX':      [{'name': 'HYB_MGGA_XC_MN12_SX'}],
              'N12SX':       [{'name': 'HYB_MGGA_XC_N12_SX'}],
              'LC-WPBE':     [{'name': 'LC-WPBE'}],
              'CAM-B3LYP':   [{'name': 'CAM-B3LYP'}],
              'WB97':        [{'name': 'WB97'}],
              'WB97X':       [{'name': 'WB97X'}],
              'WB97XD':      [{'name': 'WB97XD'}],
              'HISSBPBE':    [{'name': 'HISSBPBE'}],
              'B2PLYP':      [{'name': 'B2PLYP'}],
              'MPW2PLYP':    [{'name': 'MPW2PLYP'}],
              'B2PLYPD':     [{'name': 'B2PLYPD'}],
              'MPW2PLYPD':   [{'name': 'MPW2PLYPD'}],
              'B97D3':       [{'name': 'B97D3'}], 
              'B2PLYPD3':    [{'name': 'B2PLYPD3'}],
              'MPW2PLYPD3':  [{'name': 'MPW2PLYPD3'}],
              'LC-':         [{'name': 'Long-range corrected'}],
             }

        methodDict = {
              'AMBER':     [{'name': 'Amber'}],
              'DREIDING':  [{'name': 'Dreiding'}],
              'UFF':       [{'name': 'UFF'}],
              'AM1':       [{'name': 'AM1'}],
              'PM3':       [{'name': 'PM3'}],
              'PM3MM':     [{'name': 'PM3MM'}],
              'PM3D3':     [{'name': 'PM3D3'}],
              'PM6':       [{'name': 'PM6'}],
              'PDDG':      [{'name': 'PDDG'}],
              'CNDO':      [{'name': 'CNDO'}],
              'INDO':      [{'name': 'INDO'}],
              'MINDO':     [{'name': 'MINDO'}],
              'MINDO3':    [{'name': 'MINDO3'}],
              'ZINDO':     [{'name': 'ZINDO'}],
              'HUCKEL':    [{'name': 'HUCKEL'}],
              'EXTENDEDHUCKEL':    [{'name': 'HUCKEL'}],
              'ONIOM':     [{'name': 'ONIOM'}],
              'RHF':       [{'name': 'RHF'}],
              'UHF':       [{'name': 'UHF'}],
              'ROHF':      [{'name': 'ROHF'}],
              'GVB':       [{'name': 'GVB'}],
              'DFT':       [{'name': 'DFT'}],
              'CID':       [{'name': 'CID'}],
              'CISD':      [{'name': 'CISD'}],
              'CIS':       [{'name': 'CIS'}],
              'BD':        [{'name': 'BD'}],
              'BD(T)':     [{'name': 'BD(T)'}],
              'CCD':       [{'name': 'CCD'}],
              'CCSD':      [{'name': 'CCSD'}],
              'EOMCCSD':   [{'name': 'EOMCCSD'}],
              'QCISD':     [{'name': 'QCISD'}],
              'CCSD(T)':   [{'name': 'CCSD(T)'}],
              'QCISD(T)':  [{'name': 'QCISD(T)'}],
              'QCISD(TQ)': [{'name': 'QCISD(TQ)'}],
              'MP2':       [{'name': 'MP2'}],
              'MP3':       [{'name': 'MP3'}],
              'MP4':       [{'name': 'MP4'}],
              'MP4DQ':     [{'name': 'MP4DQ'}],
              'MP4(DQ)':   [{'name': 'MP4DQ'}],
              'MP4SDQ':    [{'name': 'MP4SDQ'}],
              'MP4(SDQ)':  [{'name': 'MP4SDQ'}],
              'MP4SDTQ':   [{'name': 'MP4SDTQ'}],
              'MP4(SDTQ)': [{'name': 'MP4SDTQ'}],
              'MP5':       [{'name': 'MP5'}],
              'CAS':       [{'name': 'CASSCF'}],
              'CASSCF':    [{'name': 'CASSCF'}],
              'G1':        [{'name': 'G1'}],
              'G2':        [{'name': 'G2'}],
              'G2MP2':     [{'name': 'G2MP2'}],
              'G3':        [{'name': 'G3'}],
              'G3MP2':     [{'name': 'G3MP2'}],
              'G3B3':      [{'name': 'G3B3'}],
              'G3MP2B3':   [{'name': 'G3MP2B3'}],
              'G4':        [{'name': 'G4'}],
              'G4MP2':     [{'name': 'G4MP2'}],
              'CBSEXTRAP':   [{'name': 'CBSExtrapolate'}],
              'CBSEXTRAPOLATE':   [{'name': 'CBSExtrapolate'}],
              'CBS-4M':    [{'name': 'CBS-4M'}],
              'CBS-4O':    [{'name': 'CBS-4O'}],
              'CBS-QB3':   [{'name': 'CBS-QB3'}],
              'CBS-QB3O':  [{'name': 'CBS-QB3O'}],
              'CBS-APNO':  [{'name': 'CBS-APNO'}],
              'W1U':       [{'name': 'W1U'}],
              'W1BD':      [{'name': 'W1BD'}],
              'W1RO':      [{'name': 'W1RO'}],
             }

        basissetDict = {
              'STO-3G':      [{'name': 'STO-3G'}],
              '3-21G':       [{'name': '3-21G'}],
              '6-21G':       [{'name': '6-21G'}],
              '4-31G':       [{'name': '4-31G'}],
              '6-31G':       [{'name': '6-31G'}],
              '6-311G':      [{'name': '6-311G'}],
              'D95V':        [{'name': 'D95V'}],
              'D95':         [{'name': 'D95'}],
              'CC-PVDZ':     [{'name': 'cc-pVDZ'}],
              'CC-PVTZ':     [{'name': 'cc-pVTZ'}],
              'CC-PVQZ':     [{'name': 'cc-pVQZ'}],
              'CC-PV5Z':     [{'name': 'cc-pV5Z'}],
              'CC-PV6Z':     [{'name': 'cc-pV6Z'}],
              'SV':          [{'name': 'SV'}],
              'SVP':         [{'name': 'SVP'}],
              'TZV':         [{'name': 'TZV'}],
              'TZVP':        [{'name': 'TZVP'}],              
              'DEF2SV':      [{'name': 'Def2SV'}],
              'DEF2SVP':     [{'name': 'Def2SVP'}],
              'DEF2SVPP':    [{'name': 'Def2SVPP'}],
              'DEF2TZV':     [{'name': 'Def2TZV'}],
              'DEF2TZVP':    [{'name': 'Def2TZVP'}],
              'DEF2TZVPP':   [{'name': 'Def2TZVPP'}],
              'DEF2QZV':     [{'name': 'Def2QZV'}],
              'DEF2QZVP':    [{'name': 'Def2QZVP'}],
              'DEF2QZVPP':   [{'name': 'Def2QZVPP'}],
              'QZVP':        [{'name': 'QZVP'}],
              'MIDIX':       [{'name': 'MidiX'}],
              'EPR-II':      [{'name': 'EPR-II'}],
              'EPR-III':     [{'name': 'EPR-III'}],
              'UGBS':        [{'name': 'UGBS'}],     
              'MTSMALL':     [{'name': 'MTSmall'}],
              'DGDZVP':      [{'name': 'DGDZVP'}],
              'DGDZVP2':     [{'name': 'DGDZVP2'}],
              'DGTZVP':      [{'name': 'DGTZVP'}],
              'CBSB3':       [{'name': 'CBSB3'}],  
              'CBSB7':       [{'name': 'CBSB7'}],
              'SHC':         [{'name': 'SHC'}],
              'SEC':         [{'name': 'SHC'}],
              'CEP-4G':      [{'name': 'CEP-4G'}],
              'CEP-31G':     [{'name': 'CEP-31G'}],
              'CEP-121G':    [{'name': 'CEP-121G'}],
              'LANL1':       [{'name': 'LANL1'}],
              'LANL2':       [{'name': 'LANL2'}],
              'SDD':         [{'name': 'SDD'}],
              'OLDSDD':      [{'name': 'OldSDD'}], 
              'SDDALL':      [{'name': 'SDDAll'}],
              'GEN':         [{'name': 'General'}],
              'CHKBAS':      [{'name': 'CHKBAS'}],
              'EXTRABASIS':  [{'name': 'ExtraBasis'}],
              'DGA1':        [{'name': 'DGA1'}],
              'DGA2':        [{'name': 'DGA2'}],
              'SVPFIT':      [{'name': 'SVPFit'}],
              'TZVPFIT':     [{'name': 'TZVPFit'}],
              'W06':         [{'name': 'W06'}],
              'CHF':         [{'name': 'CHF'}],
              'FIT':         [{'name': 'FIT'}],
              'AUTO':        [{'name': 'AUTO'}],
             }

        global xc, method, basisset, xcWrite, methodWrite, basissetWrite, methodreal, basissetreal, exc, corr, exccorr, methodprefix
        xc = None
        method = None
        basisset = None
        xcWrite = False
        methodWrite = False
        basissetWrite = False
        methodreal = None
        basissetreal = None 
        methodprefix = None
        exc = None
        corr = None
        exccorr = None

        settings = section["x_gaussian_settings"]
        settings = map(str.strip, settings)  
        settings = [''.join(map(str,settings))]
        settings = str(settings)
        settings = re.sub('[-]{2,}', '', settings)

        method1 = settings.replace("['#p ","").replace("['#P ","")
        method1 = method1.upper()

        if 'ONIOM' not in method1: 
          if settings.find("/") >= 0:
               method1 = settings.split('/')[0].replace("['#p ","").replace("['#P ","")
               method1 = method1.upper()
               for x in method1.split():
                  method2 = str(x)
                  if method2 != 'RHF' and method2 != 'UHF' and method2 != 'ROHF' and method2 != 'UFF':
                     if (method2[0] == 'R' and method2[0:2] != 'RO') or method2[0] == 'U':
                        methodprefix = method2[0] 
                        method2 = method2[1:]
                     elif method2[0:2] == 'RO':
                        methodprefix = method2[0:2]
                        method2 = method2[2:]
                  if method2[0] == 'S' or method2[0] == 'B' or method2[0] == 'O':
                     if method2[1] != '2' and method2[1] != '3':
                        exc = method2[0]
                        corr = method2[1:]
                        if exc in xcDict.keys() and corr in xcDict.keys():
                           xc = xcDict.get([exc][-1]) + xcDict.get([corr][-1])
                  if method2[0:3] == 'BRX' or method2[0:3] == 'G96':
                     exc = method2[0:3]
                     corr = method2[3:]
                  if exc in xcDict.keys() and corr in xcDict.keys():
                      xc = xcDict.get([exc][-1]) + xcDict.get([corr][-1])
                  if method2[0:5] == 'WPBEH':
                     exc = method2[0:5]
                     corr = method2[6:]
                     if exc in xcDict.keys() and corr in xcDict.keys():
                        xc = xcDict.get([exc][-1]) + xcDict.get([corr][-1])
                  if method2[0:3] == 'LC-':
                     exccorr = method2[3:]
                     if exccorr in xcDict.keys():
                        xc = 'LC-' + xcDict.get([exccorr][-1])
                  if method2 in xcDict.keys():
                     xc = method2
                     xcWrite= True
                     methodWrite = True
                     method = 'DFT'
                  if method2 in methodDict.keys():
                     method = method2
                     methodWrite = True
                     methodreal = method2
                  else:
                     for n in range(2,9):
                        if method2[0:n] in methodDict.keys():
                          if method2[0:n] in xcDict.keys():
                            method = 'DFT'
                            methodWrite = True
                          else:
                            method = method2[0:n]
                            methodWrite = True
                            methodreal = method2
                          if method2[0:9] == 'CBSEXTRAP':
                            method = method2[0:9]
                            methodWrite = True
                            methodreal = method2
               rest = settings.split('/')[1].replace("'","").replace("]","")
               rest = rest.upper() 
               for x in rest.split():
                  if x in basissetDict.keys():
                     basisset = x
                     basissetWrite = True
                     basissetreal = x
                  if 'D95' in x:
                     method2 = x
                     basisset = method2[0:3]
                     basissetWrite = True
                     basissetreal = method2
                  if 'AUG-' in x:
                     method2 = x
                     basisset = method2[4:]
                     basissetWrite = True
                     basissetreal = method2
                  if 'UGBS' in x:
                     method2 = x
                     basisset = method2[0:4]
                     basissetWrite = True
                     basissetreal = method2
                  if 'CBSB7' in x:
                     method2 = x
                     basisset = method2[0:5]
                     basissetWrite = True
                     basissetreal = method2
                  if 'LANL1' in x:
                     method2 = x
                     basisset = method2[0:5]
                     basissetWrite = True
                     basissetreal = method2
                  if 'LANL2' in x:
                     method2 = x
                     basisset = method2[0:5]
                     basissetWrite = True
                     basissetreal = method2
                  if '6-31' in x:
                     method2 = x
                     if '6-311' in x:
                        basisset = '6-311G'
                        basissetWrite = True
                        basissetreal = '6-311' + method2[5:]
                     else:
                        basisset = '6-31G'
                        basissetWrite = True
                        basissetreal = '6-31' + method2[4:]
                  slashes = settings.count('/')
                  if slashes > 1:
                    rest2 = settings.split()[1]
                    rest2 = rest2.upper()
                    for z in rest2.split('/'):
                       if z in basissetDict.keys():
                         basisset = z
                    basissetWrite = True
                    basissetreal = rest2.split('/')[1] + '/' + basisset
          else:
               method1 = settings.split()
               for x in method1: 
                  method2 = str(x)
                  method2 = method2.upper() 
                  if method2 != 'RHF' and method2 != 'UHF' and method2 != 'ROHF' and method2 != 'UFF':
                    if (method2[0] == 'R' and method2[0:2] != 'RO') or method2[0] == 'U':
                      methodprefix = method2[0] 
                      method2 = method2[1:]
                    elif method2[0:2] == 'RO':
                      methodprefix = method2[0:2] 
                      method2 = method2[2:]
                  if method2[0] == 'S' or method2[0] == 'B' or method2[0] == 'O':
                   if method2[0] in xcDict.keys() and method2[1:] in xcDict.keys():
                      exc = method2[0]
                      corr = method2[1:]
                      xc = xcDict.get([exc][-1]) + xcDict.get([corr][-1])
                  if method2[0:3] == 'BRX' or method2[0:3] == 'G96':
                   exc = method2[0:3]
                   corr = method2[3:]
                   if exc in xcDict.keys() and corr in xcDict.keys():
                      xc = xcDict.get([exc][-1]) + xcDict.get([corr][-1])
                  if method2[0:5] == 'WPBEH':
                   exc = method2[0:5]
                   corr = method2[6:]
                   if exc in xcDict.keys() and corr in xcDict.keys():
                      xc = xcDict.get([exc][-1]) + xcDict.get([corr][-1])
                  if method2[0:3] == 'LC-':
                   exccorr = method2[3:]
                   if exccorr in xcDict.keys():
                      xc = 'LC-' + xcDict.get([exccorr][-1])
                  if method2 in xcDict.keys(): 
                   xc = method2
                   xcWrite= True
                   method = 'DFT'
                  if method2 in methodDict.keys():
                   method = method2 
                   methodWrite = True
                   methodreal = method2
                  else:
                   for n in range(2,9):
                      if method2[0:n] in methodDict.keys():
                         method = method2[0:n]
                         methodWrite = True
                         methodreal = method2
                      if method2[0:9] == 'CBSEXTRAP':
                         method = method2[0:9]
                         methodWrite = True
                         methodreal = method2
                  if method2 in basissetDict.keys():
                   basisset = method2
                   basissetWrite = True
                   basissetreal = method2
                  if 'D95' in method2:
                   basisset = method2[0:3]
                   basissetWrite = True
                   basissetreal = method2
                  if 'AUG-' in method2:
                   basisset = method2[4:]
                   basissetWrite = True
                   basissetreal = method2
                  if 'UGBS' in method2:
                   basisset = method2[0:4]
                   basissetWrite = True
                   basissetreal = method2
                  if 'CBSB7' in method2:
                   basisset = method2[0:5]
                   basissetWrite = True
                   basissetreal = method2
                  if '6-31' in method2:
                   if '6-311' in method2:
                      basisset = '6-311G'
                      basissetWrite = True
                      basissetreal = '6-311' + method2[5:]
                   else:
                      basisset = '6-31G'
                      basissetWrite = True
                      basissetreal = '6-31' + method2[4:]

# special options for ONIOM calculations
        else:
          method = 'ONIOM'       
          methodWrite = True
          method1 = settings.split()
          for x in method1:
             method2 = str(x)
             method2 = method2.upper()
             if 'ONIOM' in method2:
                methodreal = method2

# description for hybrid coefficient
        xcHybridCoeffDescr = 'hybrid coefficient $\\alpha$'
        hseFunc = 'HSEH1PBE'
# functionals where hybrid_xc_coeff is written
        writeHybridCoeff = ['B3LYP', 'OHSE2PBE', 'HSEh1PBE', 'PBE1PBE' ]
        if xc is not None:
          # check if only one xc keyword was found in output
          if len([xc]) > 1:
              logger.error("Found %d settings for the xc functional: %s. This leads to an undefined behavior of the calculation and no metadata can be written for xc." % (len(xc), xc))
          else:
              backend.superBackend.addValue('xc', [xc][-1])
            # check for hybrid_xc_coeff
#                hybridCoeff = valuesDict.get('hybrid_xc_coeff')
            # write hybrid_xc_coeff for certain functionals
#                if hybridCoeff is not None and xc[-1] in writeHybridCoeff:
#                    backend.superBackend.addValue('hybrid_xc_coeff', hybridCoeff[-1])
            # convert xc functional for metadata
              if xcWrite:
              # get list of xc components according to parsed value
                  xcList = xcDict.get([xc][-1])
                  if xcList is not None:
                  # loop over the xc components
                      for xcItem in xcList:
                          xcName = xcItem.get('name')
                          if xcName is not None:
                          # write section and XC_functional_name
                              gIndexTmp = backend.openSection('section_XC_functionals')
                              backend.addValue('XC_functional_name', xcName)
                            # write hybrid_xc_coeff for B3LYP and HSE03 into XC_functional_parameters
#                                if hybridCoeff is not None and xc[-1] in ['B3LYP', 'OHSE2PBE']:
#                                    backend.addValue('XC_functional_parameters', {xcHybridCoeffDescr: hybridCoeff[-1]})
                            # write hybrid_xc_coeff for HSE06
#                                elif xc[-1] == hseFunc:
                                # add hybrid_xc_coeff
#                                    if hybridCoeff is not None:
#                                        hybrid = hybridCoeff[-1]
#                                    else:
#                                        hybrid = 0.25
#                                    parameters[xcHybridCoeffDescr] = hybrid
#                                    backend.addValue('XC_functional_parameters', parameters)
                            # adjust weight of functionals that are affected by hybrid_xc_coeff
#                                elif hybridCoeff is not None and 'convert' in xcItem:
#                                    backend.addValue('XC_functional_weight', xcItem['convert'](hybridCoeff[-1]))
                            # write weight if present for current xcItem
#                                else:
#                                    xcWeight = xcItem.get('weight')
#                                    if xcWeight is not None:
#                                        backend.addValue('XC_functional_weight', xcWeight)
#                                backend.closeSection('section_XC_functionals', gIndexTmp)
                          else:
                              logger.error("The dictionary for xc functional '%s' does not have the key 'name'. Please correct the dictionary xcDict in %s." % (xc[-1], os.path.basename(__file__)))
                  else:
                      logger.error("The xc functional '%s' could not be converted for the metadata. Please add it to the dictionary xcDict in %s." % (xc[-1], os.path.basename(__file__)))

# Write electronic structure method to metadata

        if method is not None:
          # check if only one method keyword was found in output
          if len([method]) > 1:
              logger.error("Found %d settings for the method: %s. This leads to an undefined behavior of the calculation and no metadata can be written for the method." % (len(method), method))
          else:
              backend.superBackend.addValue('method', [method][-1])
          methodList = methodDict.get([method][-1])
          if methodWrite:
               if methodList is not None:
        # loop over the method components
                  for methodItem in methodList:
                        methodName = methodItem.get('name')
                        if methodName is not None:
                 # write section and method name
                           gIndexTmp = backend.openSection('x_gaussian_section_elstruc_method')
                           if methodprefix != None and methodreal != None:
                              backend.addValue('x_gaussian_elstruc_method_name', str(methodprefix) + methodreal)
                           elif methodreal != None:
                              backend.addValue('x_gaussian_elstruc_method_name', methodreal)
                        else:
                              logger.error("The dictionary for method '%s' does not have the key 'name'. Please correct the dictionary methodDict in %s." % (method[-1], os.path.basename(__file__)))
               else:
                      logger.error("The method '%s' could not be converted for the metadata. Please add it to the dictionary methodDict in %s." % (method[-1], os.path.basename(__file__)))

#Write basis sets to metadata

        if basisset is not None:
          # check if only one method keyword was found in output
          if len([basisset]) > 1:
              logger.error("Found %d settings for the basis set: %s. This leads to an undefined behavior of the calculation and no metadata can be written for the basis set." % (len(method), method))
          else:
              backend.superBackend.addValue('basis_set', basisset)
          basissetList = basissetDict.get([basisset][-1])
          if basissetWrite:
               if basissetList is not None:
        # loop over the basis set components
                  for basissetItem in basissetList:
                        basissetName = basissetItem.get('name')
                        if basissetName is not None:
                 # write section and basis set name(s)
                           gIndexTmp = backend.openSection('section_basis_set_atom_centered')
                           backend.addValue('basis_set_atom_centered_short_name', basissetreal)
                        else:
                              logger.error("The dictionary for basis set '%s' does not have the key 'name'. Please correct the dictionary basissetDict in %s." % (basisset[-1], os.path.basename(__file__)))
               else:
                      logger.error("The basis set '%s' could not be converted for the metadata. Please add it to the dictionary basissetDict in %s." % (basisset[-1], os.path.basename(__file__)))

      def onOpen_section_system(self, backend, gIndex, section):
          # keep track of the latest system description section
          self.secSystemDescriptionIndex = gIndex

      def onClose_section_system(self, backend, gIndex, section):
            # write/store unit cell if present and set flag self.periodicCalc
            if(section['x_gaussian_geometry_lattice_vector_x']):
              unit_cell = []
              for i in ['x', 'y', 'z']:
                  uci = str(section['x_gaussian_geometry_lattice_vector_' + i])
                  uci = uci.split() 
                  for i in range(len(uci)):
                    uci[i] = str(uci[i]).replace("[","").replace("'","").replace("]","").replace("\"","").replace(",","")
                    uci[i] = float(uci[i])
                  if uci is not None:
                     uci = convert_unit(uci, "angstrom", "m")
                     unit_cell.append(uci)
              if unit_cell:
                  # from metadata: "The first index is x,y,z and the second index the lattice vector."
                  # => unit_cell has already the right format
                  backend.addArrayValues('simulation_cell', np.asarray(unit_cell))
                  if np.shape(unit_cell) == (3, 1):
                    backend.addArrayValues('configuration_periodic_dimensions', np.asarray([True, False, False]))
                  if np.shape(unit_cell) == (3, 2):
                    backend.addArrayValues('configuration_periodic_dimensions', np.asarray([True, True, False]))
                  if np.shape(unit_cell) == (3, 3):
                    backend.addArrayValues('configuration_periodic_dimensions', np.asarray([True, True, True]))
                  self.periodicCalc = True
            if(section["x_gaussian_atomic_masses"]):
               atomicmasses = str(section["x_gaussian_atomic_masses"])
               atmass = []
               mass = [float(f) for f in atomicmasses[1:].replace("'","").replace(",","").replace("]","").replace(" ."," 0.").replace(" -."," -0.").split()]
               atmass = np.append(atmass, mass)
               numberofatoms = len(atmass)
               backend.addArrayValues("x_gaussian_masses", atmass)
               backend.addValue("x_gaussian_number_of_atoms",numberofatoms)

# which values to cache or forward (mapping meta name -> CachingLevel)

cachingLevelForMetaName = {
        "x_gaussian_atom_x_coord": CachingLevel.Cache,
        "x_gaussian_atom_y_coord": CachingLevel.Cache,
        "x_gaussian_atom_z_coord": CachingLevel.Cache,
        "x_gaussian_atomic_number": CachingLevel.Cache,
        "x_gaussian_section_geometry": CachingLevel.Ignore,
        "x_gaussian_atom_x_force": CachingLevel.Cache,
        "x_gaussian_atom_y_force": CachingLevel.Cache,
        "x_gaussian_atom_z_force": CachingLevel.Cache,
        "x_gaussian_section_frequencies": CachingLevel.Forward,
        "x_gaussian_atomic_masses": CachingLevel.Cache, 
        "x_gaussian_section_eigenvalues": CachingLevel.Cache,
        "x_gaussian_section_orbital_symmetries": CachingLevel.Cache,
        "x_gaussian_section_molecular_multipoles": CachingLevel.Cache,
        "x_gaussian_single_configuration_calculation_converged": CachingLevel.Cache,
        "x_gaussian_geometry_optimization_converged": CachingLevel.Cache,
}

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = GaussianParserContext())
