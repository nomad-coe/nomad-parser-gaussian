import setup_paths
from nomadcore.simple_parser import mainFunction, SimpleMatcher as SM
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.caching_backend import CachingLevel
import os, sys, json, logging
import numpy as np

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
                      SM(r"\s*%[Cc]hk=(?P<gaussian_chk_file>[A-Za-z0-9.]*)"),
                      SM(r"\s*%[Mm]em=(?P<gaussian_memory>[A-Za-z0-9.]*)"),
                      SM(r"\s*%[Nn][Pp]roc=(?P<gaussian_number_of_processors>[A-Za-z0-9.]*)")
                      ]
              ),
               SM(name = 'charge_multiplicity',
	          sections  = ['section_system_description','gaussian_section_labels'],
		  startReStr = r"\s*Charge =",
                  subFlags = SM.SubFlags.Sequenced,
                  forwardMatch = True,
                  subMatchers = [
		      SM(r"\s*Charge =\s*(?P<total_charge>[-+0-9]+) Multiplicity =\s*(?P<target_multiplicity>[0-9]+)"),
                      SM(r"\sModel"),
                      SM(r"\sShort"), 
                      SM(r"\s*Atom"),
                      SM(r"\s*\d\d?\d?\s{7,8}?[A-Za-z-]", repeats = True),
                      SM(r"\s*Generated"), 
                      SM(r"\sNo Z-Matrix found in file|\sZ-Matrix found in file"),
                      SM(r"\sRedundant internal coordinates found in file"),
                      SM(r"\s*(?P<gaussian_atom_label>([A-Za-z][A-Za-z]|[A-WYZa-wyz]|\d\d?\d?))[^A-Za-z]", repeats=True),
                      SM(r"\sRecover connectivity data from disk."),
                      SM(r"\s*Variables:|\s*------|\s*\r?\n")
                  ]
              ),
               SM(name = 'geometry',
                  sections  = ['section_system_description','gaussian_section_geometry'],
                  startReStr = r"\s*Z-Matrix orientation:|\s*Input orientation:|\s*Standard orientation:",
                      subMatchers = [
                      SM(r"\s+[0-9]+\s+[0-9]+\s+[0-9]+\s+(?P<gaussian_atom_x_coord__angstrom>[-+0-9EeDd.]+)\s+(?P<gaussian_atom_y_coord__angstrom>[-+0-9EeDd.]+)\s+(?P<gaussian_atom_z_coord__angstrom>[-+0-9EeDd.]+)",repeats = True)
                      ]
              ), 
                   SM(r"\s*Symmetry|\s*Stoichiometry")
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

    def onClose_gaussian_section_labels(self, backend, gIndex, section):
        labels = section["gaussian_atom_label"]
        backend.addValue("atom_label", labels)

    def onClose_gaussian_section_geometry(self, backend, gIndex, section):
	xCoord = section["gaussian_atom_x_coord"]
	yCoord = section["gaussian_atom_y_coord"]
        zCoord = section["gaussian_atom_z_coord"]
        atom_positions = np.zeros((len(xCoord),3), dtype=float)
	for i in range(len(xCoord)):
	    atom_positions[i,0] = xCoord[i]
            atom_positions[i,1] = yCoord[i]
            atom_positions[i,2] = zCoord[i]
	backend.addArrayValues("atom_position", atom_positions)

# which values to cache or forward (mapping meta name -> CachingLevel)
cachingLevelForMetaName = {
	"gaussian_atom_x_coord": CachingLevel.Cache,
        "gaussian_atom_y_coord": CachingLevel.Cache,
        "gaussian_atom_z_coord": CachingLevel.Cache,
	"gaussian_atom_label": CachingLevel.Cache,
	"gaussian_section_geometry": CachingLevel.Ignore,
        "gaussian_section_labels": CachingLevel.Ignore,
}

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = GaussianParserContext())
