import setup_paths
from nomadcore.simple_parser import SimpleMatcher, mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
import os, sys, json

# description of the input
mainFileDescription = SimpleMatcher(name = 'root',
              weak = True,
              startReStr = "",
              subMatchers = [
  SimpleMatcher(name = 'newRun',
                startReStr = r"\s*# SampleParser #\s*",
                repeats = True,
                required = True,
                forwardMatch = True,
                sections   = ['section_run'],
                subMatchers = [
    SimpleMatcher(name = 'header',
                  startReStr = r"\s*# SampleParser #\s*")
                ])
              ])

# loading metadata from nomad-meta-info/meta_info/nomad_meta_info/gaussian.nomadmetainfo.json
metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/gaussian.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath = metaInfoPath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)

parserInfo = {
  "name": "parser_gaussian",
  "version": "1.0"
}

# default unit conversions (actually it might be better to use the sourceUnits argument of the SimpleMatcher)
defaultSourceUnits = {}

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo, defaultSourceUnits = defaultSourceUnits)
