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
import csv

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
               SM(name = 'charge_multiplicity_natoms',
               sections  = ['x_gaussian_section_system'],
		  startReStr = r"\s*Charge =",
                  subFlags = SM.SubFlags.Sequenced,
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*Charge =\s*(?P<x_gaussian_total_charge>[-+0-9]+) Multiplicity =\s*(?P<x_gaussian_spin_target_multiplicity>[0-9]+)"),
                      SM(r"\s*NAtoms=(?P<x_gaussian_natoms>[0-9]+)"),
                      ]
             ),
               SM(name = 'atomic masses',
                  sections = ['x_gaussian_section_atomic_masses'],
                  startReStr = r"\s*AtmWgt=",
                  endReStr = r"\s*Leave Link  101",
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*AtmWgt=\s+(?P<x_gaussian_atomic_masses>[0-9.]+(\s+[0-9.]+)(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?(\s+[0-9.]+)?)", repeats = True)
                      ]
             ),
            # this SimpleMatcher groups a single configuration calculation together with output after SCF convergence from relaxation
            SM (name = 'SingleConfigurationCalculationWithSystemDescription',
                startReStr = "\s*Standard orientation:",
#                endReStr = "\s*Link1:  Proceeding to internal job step number",
                repeats = False,
                forwardMatch = True,
                subMatchers = [
                # the actual section for a single configuration calculation starts here
                SM (name = 'SingleConfigurationCalculation',
                  startReStr = "\s*Standard orientation:",
                  repeats = True,
                  forwardMatch = True,
                  sections = ['x_gaussian_section_single_configuration_calculation'],
                  subMatchers = [
                  SM(name = 'geometry',
                   sections  = ['x_gaussian_section_geometry'],
                   startReStr = r"\s*Standard orientation",
                   endReStr = r"\s*Rotational constants",
                      subMatchers = [
                      SM(r"\s+[0-9]+\s+(?P<x_gaussian_atomic_number>[0-9]+)\s+[0-9]+\s+(?P<x_gaussian_atom_x_coord__angstrom>[-+0-9EeDd.]+)\s+(?P<x_gaussian_atom_y_coord__angstrom>[-+0-9EeDd.]+)\s+(?P<x_gaussian_atom_z_coord__angstrom>[-+0-9EeDd.]+)",repeats = True),
                      SM(r"\s*Rotational constants")
                    ]
                ),
                    SM(name = 'TotalEnergyScfGaussian',
                    sections  = ['x_gaussian_section_scf_iteration'],
                    startReStr = r"\s*Cycle\s+[0-9]+",
                    forwardMatch = True,
                    repeats = True, 
                    subMatchers = [
                    SM(r"\s*Cycle\s+[0-9]+"),
                    SM(r"\s*E=\s*(?P<x_gaussian_energy_total_scf_iteration__hartree>[-+0-9.]+)\s*Delta-E=\s*(?P<x_gaussian_delta_energy_total_scf_iteration__hartree>[-+0-9.]+)")
                    ]
                ), 
                    SM(name = 'TotalEnergyScfConverged',
                    sections  = ['x_gaussian_section_total_scf_one_geometry'],
                    startReStr = r"\s*SCF Done",
                    forwardMatch = True,
                    subMatchers = [
                    SM(r"\s*(?P<x_gaussian_single_configuration_calculation_converged>SCF Done):\s*[()A-Za-z0-9-]+\s*=\s*(?P<x_gaussian_energy_total_scf_converged__hartree>[-+0-9.]+)")
                    ]  
                ),
                    SM(name = 'RealSpinValue',
                    sections  = ['x_gaussian_section_real_spin_squared'],
                    startReStr = r"\s*Convg\s*=",
                    forwardMatch = True,
                    subMatchers = [
                     SM(r"\s*[A-Z][*][*][0-9]\s*=\s*(?P<x_gaussian_spin_S2>[0-9.]+)"),
                     SM(r"\s*Annihilation of the first spin contaminant"),
                     SM(r"\s*[A-Z][*][*][0-9]\s*before annihilation\s*[0-9.,]+\s*after\s*(?P<x_gaussian_after_annihilation_spin_S2>[0-9.]+)")
                     ]
                ),
                   SM(name = 'ForcesScfGaussian',
                   sections  = ['x_gaussian_section_atom_forces'],
                   startReStr = "\s*Center\s+Atomic\s+Forces ",
                   forwardMatch = True,
                   subMatchers = [
                    SM(r"\s*Center\s+Atomic\s+Forces "),
                    SM(r"\s+[0-9]+\s+[0-9]+\s+(?P<x_gaussian_atom_x_force__hartree_bohr_1>[-+0-9EeDd.]+)\s+(?P<x_gaussian_atom_y_force__hartree_bohr_1>[-+0-9EeDd.]+)\s+(?P<x_gaussian_atom_z_force__hartree_bohr_1>[-+0-9EeDd.]+)",repeats = True),
                    SM(r"\s*Cartesian Forces:\s+")
                    ]
                ),
                  SM(name = 'Geometry_optimization',
                  startReStr = r"\s*Optimization completed.",
                  subMatchers = [
                  SM(r"\s*(?P<x_gaussian_geometry_optimization_converged>Optimized Parameters)"),
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
                      SM(r"\s*The electronic state is\s*(?P<x_gaussian_elstate_symmetry>[A-Z0-9-]+)[.]")
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
                  startReStr = r"\s*Charge=",
                  forwardMatch = True,
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
                                'energy_XC_potential': None
                               }

      def initialize_values(self):
        """Initializes the values of certain variables.

        This allows a consistent setting and resetting of the variables,
        when the parsing starts and when a section_run closes.
        """
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        self.scfConvergence = False
        self.geoConvergence = None
        self.geoCrashed = None
        self.scfenergyconverged = 0.0

      def startedParsing(self, path, parser):
        self.parser = parser
        # allows to reset values if the same superContext is used to parse different files
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
        backend.addArrayValues("x_gaussian_atom_labels", atomic_symbols)
        backend.addArrayValues("x_gaussian_atom_positions", atom_coords)

      def onClose_x_gaussian_section_atom_forces(self, backend, gIndex, section):
        xForce = section["x_gaussian_atom_x_force"]
        yForce = section["x_gaussian_atom_y_force"]
        zForce = section["x_gaussian_atom_z_force"]
        atom_forces = np.zeros((len(xForce),3), dtype=float)
        for i in range(len(xForce)):
           atom_forces[i,0] = xForce[i]
           atom_forces[i,1] = yForce[i]
           atom_forces[i,2] = zForce[i]
        backend.addArrayValues("x_gaussian_atom_forces", atom_forces)

      def onClose_section_run(self, backend, gIndex, section):
        """Trigger called when section_run is closed.
        """
                # write geometry optimization convergence
        gIndexTmp = backend.openSection('x_gaussian_section_single_configuration_calculation')
        if self.geoConvergence is not None:
            backend.addValue('x_gaussian_geometry_optimization_converged', self.geoConvergence)
        # use values of control.in which was parsed in section_method
        backend.closeSection('x_gaussian_section_single_configuration_calculation', gIndexTmp)

      def onClose_x_gaussian_section_single_configuration_calculation(self, backend, gIndex, section):
        """Trigger called when section_single_configuration_calculation is closed.
         Write number of SCF iterations and convergence.
         Check for convergence of geometry optimization.
        """
        # write number of SCF iterations
        self.scfenergyconverged = section['x_gaussian_energy_total_scf_converged']
        backend.addValue('x_gaussian_number_of_scf_iterations', self.scfIterNr)
        # write SCF convergence and reset
        backend.addValue('x_gaussian_single_configuration_calculation_converged', self.scfConvergence)
        backend.addValue('x_gaussian_energy_total_scf_converged', self.scfenergyconverged)
        self.scfConvergence = False
        # check for geometry optimization convergence
        if section['x_gaussian_geometry_optimization_converged'] is not None:
            if section['x_gaussian_geometry_optimization_converged'][-1] == 'Optimization completed':
               self.geoConvergence = True
            else:
               if section['x_gaussian_geometry_optimization_converged'][-1] == 'Optimization stopped':
                  self.geoConvergence = False
        if section['x_gaussian_geometry_optimization_converged'] is not None:
             if section['x_gaussian_geometry_optimization_converged'][-1] == 'Optimization stopped':
               self.geoCrashed = True
             else:
               self.geoCrashed = False
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1

      def onClose_x_gaussian_section_scf_iteration(self, backend, gIndex, section):
        # count number of SCF iterations
        self.scfIterNr += 1
        # check for SCF convergence
        if section['x_gaussian_single_configuration_calculation_converged'] is not None:
            self.scfConvergence = True

      def onClose_x_gaussian_section_atomic_masses(self, backend, gIndex, section):
          if(section["x_gaussian_atomic_masses"]):
             atomicmasses = str(section["x_gaussian_atomic_masses"])
             atmass = []
             mass = [float(f) for f in atomicmasses[1:].replace("'","").replace(",","").replace("]","").replace(" ."," 0.").replace(" -."," -0.").split()]
             atmass = np.append(atmass, mass)
             backend.addArrayValues("x_gaussian_masses", atmass)   
       
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

          symmetry = [str(f) for f in symoccalpha[1:].replace("'","").replace(",","").replace("(","").replace(")","").replace("]","").split()]
          sym1 = []
          sym1 = np.append(sym1, symmetry)  
          symmetry = [str(f) for f in symviralpha[1:].replace("'","").replace(",","").replace("(","").replace(")","").replace("]","").split()]
          sym2 = []
          sym2 = np.append(sym2, symmetry)
          symmetrycon = np.concatenate((sym1, sym2), axis=0)
          backend.addArrayValues("x_gaussian_alpha_symmetries", symmetrycon) 

          if(section["x_gaussian_beta_occ_symmetry_values"]):
             symmetry = [str(f) for f in symoccbeta[1:].replace("'","").replace(",","").replace("(","").replace(")","").replace("]","").split()]
             sym1 = []
             sym1 = np.append(sym1, symmetry)
             symmetry = [str(f) for f in symvirbeta[1:].replace("'","").replace(",","").replace("(","").replace(")","").replace("]","").split()]
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
#         charge = convert_unit

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

#          x_gaussian_molecular_multipole_lm[0] = (0,0) 
#          x_gaussian_molecular_multipole_lm[1] = (1,0)
#          x_gaussian_molecular_multipole_lm[2] = (1,1)
#          x_gaussian_molecular_multipole_lm[3] = (1,2)
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
                M = len(disps)/len(vibfreqs) * (p+1) 
                for m in range(3):
                  for n in range(M - len(disps) // len(vibfreqs),M,3):
                    for l in range(3):
                      dispsnew[k] = disps[3*(n + m) + l]
                      k = k + 1
          elif len(vibfreqs) % 3 != 0:
             k = 0
             for p in range(len(vibfreqs)-1,0,-3):
                M = (len(disps) - len(disps) // len(vibfreqs)) // p
                for m in range(3):
                  for n in range(M - len(disps) // len(vibfreqs),M,3):
                    for l in range(3):
                      dispsnew[k] = disps[3*(n + m) + l]
                      k = k + 1
             for m in range(len(disps) // len(vibfreqs)):
                   dispsnew[k] = disps[k]
                   k = k + 1

          vibnormalmodes = np.append(vibnormalmodes, dispsnew)
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

# which values to cache or forward (mapping meta name -> CachingLevel)
cachingLevelForMetaName = {
        "x_gaussian_atom_x_coord": CachingLevel.Cache,
        "x_gaussian_atom_y_coord": CachingLevel.Cache,
        "x_gaussian_atom_z_coord": CachingLevel.Cache,
        "x_gaussian_atomic_number": CachingLevel.Cache,
        "x_gaussian_section_geometry": CachingLevel.Ignore,
        "x_gaussian_natoms": CachingLevel.Cache,
        "x_gaussian_section_total_scf_one_geometry": CachingLevel.Cache,
        "x_gaussian_geometry_optimization_converged": CachingLevel.Cache,
        "x_gaussian_section_scf_iteration": CachingLevel.Cache,
        "x_gaussian_single_configuration_calculation_converged": CachingLevel.Ignore,
        "x_gaussian_atom_x_force": CachingLevel.Cache,
        "x_gaussian_atom_y_force": CachingLevel.Cache,
        "x_gaussian_atom_z_force": CachingLevel.Cache,
        "x_gaussian_section_atom_forces": CachingLevel.Ignore, 
        "x_gaussian_section_frequencies": CachingLevel.Forward,
        "x_gaussian_atomic_masses": CachingLevel.Cache, 
        "x_gaussian_section_eigenvalues": CachingLevel.Cache,
        "x_gaussian_section_orbital_symmetries": CachingLevel.Cache,
        "x_gaussian_section_molecular_multipoles": CachingLevel.Cache,
}

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = GaussianParserContext())
