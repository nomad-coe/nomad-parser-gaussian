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
    startReStr = "",
    subMatchers = [
        SM(name = 'newRun',
           startReStr = r"\s*Entering Link 1 ",
           repeats = True,
           required = True,
           forwardMatch = True,
           fixedStartValues={ 'program_basis_set_type': 'gaussians' },
           sections   = ['section_run','section_method'],
           subMatchers = [
               SM(name = 'header',
                  startReStr = r"\s*Entering Link 1 ",
                  subMatchers = [
                      SM(r"\s*Cite this work as:"),
                      SM(r"\s*Gaussian [0-9]+, Revision [A-Za-z0-9.]*,"),
                      SM(r"\s\*\*\*\*\*\*\*\*\*\*\*\**"),
                      SM(r"\s*(?P<program_name>Gaussian)\s*(?P<program_version>[0-9]*:\s.*)")
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
               SM(name = 'charge_multiplicity',
	          sections  = ['section_system_description','x_gaussian_section_chargemult'],
		  startReStr = r"\s*Charge =",
                  subFlags = SM.SubFlags.Sequenced,
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*Charge =\s*(?P<total_charge>[-+0-9]+) Multiplicity =\s*(?P<target_multiplicity>[0-9]+)"),
                      ]
              ),
               SM(name = 'geometry',
                  sections  = ['section_system_description','x_gaussian_section_geometry'],
                  startReStr = r"\s*Input orientation:|\s*Z-Matrix orientation:|\s*Standard orientation:",
                  subFlags = SM.SubFlags.Sequenced,
                      subMatchers = [
                      SM(r"\s+[0-9]+\s+(?P<x_gaussian_atomic_number>[0-9]+)\s+[0-9]+\s+(?P<x_gaussian_atom_x_coord__angstrom>[-+0-9EeDd.]+)\s+(?P<x_gaussian_atom_y_coord__angstrom>[-+0-9EeDd.]+)\s+(?P<x_gaussian_atom_z_coord__angstrom>[-+0-9EeDd.]+)",repeats = True),
                      SM(r"\s*Distance matrix|\s*Rotational constants|\s*Stoichiometry")
                      ]
              ), 
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
    """main place to keep the parser status, open ancillary files,..."""
    def __init__(self):
        pass

    def startedParsing(self, path, parser):
        self.parser = parser

    def onClose_x_gaussian_section_geometry(self, backend, gIndex, section):
	xCoord = section["x_gaussian_atom_x_coord"]
	yCoord = section["x_gaussian_atom_y_coord"]
        zCoord = section["x_gaussian_atom_z_coord"]
        numbers = section["x_gaussian_atomic_number"]
        atom_positions = np.zeros((len(xCoord),3), dtype=float)
        atom_numbers = np.zeros(len(xCoord), dtype=int)
        atomic_symbols = np.empty((len(xCoord)), dtype=object) 
	for i in range(len(xCoord)):
	    atom_positions[i,0] = xCoord[i]
            atom_positions[i,1] = yCoord[i]
            atom_positions[i,2] = zCoord[i]
        for i in range(len(xCoord)):
            atom_numbers[i] = numbers[i]
            atomic_symbols[i] = ase.data.chemical_symbols[atom_numbers[i]] 
	backend.addArrayValues("atom_position", atom_positions)
        backend.addArrayValues("atom_label", atomic_symbols)

# which values to cache or forward (mapping meta name -> CachingLevel)
cachingLevelForMetaName = {
	"x_gaussian_atom_x_coord": CachingLevel.Cache,
        "x_gaussian_atom_y_coord": CachingLevel.Cache,
        "x_gaussian_atom_z_coord": CachingLevel.Cache,
        "x_gaussian_atomic_number": CachingLevel.Cache,
	"x_gaussian_section_geometry": CachingLevel.Ignore,
}

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = GaussianParserContext())
