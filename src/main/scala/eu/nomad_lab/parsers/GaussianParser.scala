package eu.nomad_lab.parsers
import eu.nomad_lab.DefaultPythonInterpreter
import org.{json4s => jn}

object GaussianParser extends SimpleExternalParserGenerator(
      name = "GaussianParser",
      parserInfo = jn.JObject(
        ("name" -> jn.JString("GaussianParser")) ::
          ("version" -> jn.JString("1.0")) :: Nil),
      mainFileTypes = Seq("text/.*"),
      mainFileRe = """\s*Gaussian, Inc\.  All Rights Reserved\.\s*
\s*
\s*This is part of the Gaussian\(R\) [0-9]* program.""".r,
      cmd = Seq(DefaultPythonInterpreter.python2Exe(), "${envDir}/parsers/gaussian/parser/parser-gaussian/parser_gaussian.py",
        "--uri", "${mainFileUri}", "${mainFilePath}"),
      resList = Seq(
        "parser-gaussian/parser_gaussian.py",
        "parser-gaussian/setup_paths.py",
        "nomad_meta_info/common.nomadmetainfo.json",
        "nomad_meta_info/meta_types.nomadmetainfo.json",
        "nomad_meta_info/gaussian.nomadmetainfo.json"
      ) ++ DefaultPythonInterpreter.commonFiles(),
      dirMap = Map(
        "parser-gaussian" -> "parsers/gaussian/parser/parser-gaussian",
        "nomad_meta_info" -> "nomad-meta-info/meta_info/nomad_meta_info",
        "python" -> "python-common/common/python/nomadcore") ++ DefaultPythonInterpreter.commonDirMapping()
)
