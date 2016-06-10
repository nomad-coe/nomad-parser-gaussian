import setup_paths
from nomadcore.simple_parser import mainFunction, SimpleMatcher as SM
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.caching_backend import CachingLevel
import os, sys, json, logging
import numpy as np
import ase

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
                      SM(r"\s*(?P<program_name>Gaussian)\s*(?P<program_version>[0-9]+):\s*(?P<x_gaussian_program_implementation>[A-Za-z0-9-.]+)\s*(?P<x_gaussian_program_release_date>[0-9][0-9]?\-[A-Z][a-z][a-z]\-[0-9]+)"),
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
                SM(name = 'Frequencies',
                sections = ['x_gaussian_section_frequencies'],
                startReStr = r"\s*Frequencies --",
                forwardMatch = True,
                subFlags = SM.SubFlags.Sequenced,
                subMatchers = [
                      SM(r"\s*Frequencies --\s*(?P<x_gaussian_frequency_values>(.+)?)", repeats = True),
                      SM(r"\s*- Thermochemistry -"),
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

      def onClose_x_gaussian_section_frequencies(self, backend, gIndex, section):
          frequencies = str(section["x_gaussian_frequency_values"])
          vibfreqs = []
          freqs = [float(f) for f in frequencies[1:].replace("'","").replace(",","").replace("]","").split()]
          vibfreqs = np.append(vibfreqs, freqs)
          backend.addArrayValues("x_gaussian_frequencies", vibfreqs)

      def onClose_x_gaussian_section_atomic_masses(self, backend, gIndex, section):
          atomicmasses = str(section["x_gaussian_atomic_masses"])
          atmass = []
          mass = [float(f) for f in atomicmasses[1:].replace("'","").replace(",","").replace("]","").split()]
          atmass = np.append(atmass, mass)
          backend.addArrayValues("x_gaussian_masses", atmass)   
       
      def onClose_x_gaussian_section_eigenvalues(self, backend, gIndex, section):
          eigenenergies = str(section["x_gaussian_alpha_occ_eigenvalues_values"])
          eigenen1 = []
          energy = [float(f) for f in eigenenergies[1:].replace("'","").replace(",","").replace("]","").split()]
          eigenen1 = np.append(eigenen1, energy)
          occoccupationsalp = np.ones(len(eigenen1), dtype=float)
          eigenenergies = str(section["x_gaussian_alpha_vir_eigenvalues_values"])
          eigenen2 = []
          energy = [float(f) for f in eigenenergies[1:].replace("'","").replace(",","").replace("]","").split()]
          eigenen2 = np.append(eigenen2, energy)
          viroccupationsalp = np.zeros(len(eigenen2), dtype=float)
          eigenencon = np.zeros(len(eigenen1) + len(eigenen2))
          eigenencon = np.concatenate((eigenen1,eigenen2), axis=0)
          occupcon = np.concatenate((occoccupationsalp, viroccupationsalp), axis=0)
          backend.addArrayValues("x_gaussian_alpha_eigenvalues", eigenencon)
          backend.addArrayValues("x_gaussian_alpha_occupations", occupcon)

          eigenenergies = str(section["x_gaussian_beta_occ_eigenvalues_values"])
          eigenen1 = []
          energy = [float(f) for f in eigenenergies[1:].replace("'","").replace(",","").replace("]","").split()]
          eigenen1 = np.append(eigenen1, energy)
          occoccupationsbet = np.ones(len(eigenen1), dtype=float)
          eigenenergies = str(section["x_gaussian_beta_vir_eigenvalues_values"])
          eigenen2 = []
          energy = [float(f) for f in eigenenergies[1:].replace("'","").replace(",","").replace("]","").split()]
          eigenen2 = np.append(eigenen2, energy)
          viroccupationsbet = np.zeros(len(eigenen2), dtype=float)
          eigenencon = np.zeros(len(eigenen1) + len(eigenen2))
          eigenencon = np.concatenate((eigenen1,eigenen2), axis=0)
          occupcon = np.concatenate((occoccupationsbet, viroccupationsbet), axis=0)
          backend.addArrayValues("x_gaussian_beta_eigenvalues", eigenencon)
          backend.addArrayValues("x_gaussian_beta_occupations", occupcon)

      def onClose_x_gaussian_section_orbital_symmetries(self, backend, gIndex, section):
          symoccalpha = str(section["x_gaussian_alpha_occ_symmetry_values"])
          symviralpha = str(section["x_gaussian_alpha_vir_symmetry_values"])
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
          symmetry = [str(f) for f in symoccbeta[1:].replace("'","").replace(",","").replace("(","").replace(")","").replace("]","").split()]
          sym1 = []
          sym1 = np.append(sym1, symmetry)
          symmetry = [str(f) for f in symvirbeta[1:].replace("'","").replace(",","").replace("(","").replace(")","").replace("]","").split()]
          sym2 = []
          sym2 = np.append(sym2, symmetry)
          symmetrycon = np.concatenate((sym1, sym2), axis=0)
          backend.addArrayValues("x_gaussian_beta_symmetries", symmetrycon)

# which values to cache or forward (mapping meta name -> CachingLevel)
cachingLevelForMetaName = {
        "x_gaussian_atom_x_coord": CachingLevel.Cache,
        "x_gaussian_atom_y_coord": CachingLevel.Cache,
        "x_gaussian_atom_z_coord": CachingLevel.Cache,
        "x_gaussian_atomic_number": CachingLevel.Cache,
        "x_gaussian_section_geometry": CachingLevel.Ignore,
        "x_gaussian_section_total_scf_one_geometry": CachingLevel.Cache,
        "x_gaussian_geometry_optimization_converged": CachingLevel.Cache,
        "x_gaussian_section_scf_iteration": CachingLevel.Cache,
        "x_gaussian_single_configuration_calculation_converged": CachingLevel.Ignore,
        "x_gaussian_atom_x_force": CachingLevel.Cache,
        "x_gaussian_atom_y_force": CachingLevel.Cache,
        "x_gaussian_atom_z_force": CachingLevel.Cache,
        "x_gaussian_section_atom_forces": CachingLevel.Ignore, 
        "x_gaussian_section_frequencies": CachingLevel.Cache,
        "x_gaussian_atomic_masses": CachingLevel.Cache, 
        "x_gaussian_section_eigenvalues": CachingLevel.Cache,
        "x_gaussian_section_orbital_symmetries": CachingLevel.Cache,
}

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = GaussianParserContext())
