package eu.nomad_lab.parsers

import eu.nomad_lab.{parsers, DefaultPythonInterpreter}
import org.scalacheck.Properties
import org.specs2.mutable.Specification
import org.{json4s => jn}


object GaussianParserSpec extends Specification {
  "GaussianParserTest" >> {
    examplesBlock {
      ParserRun.parse(GaussianParser,"/home/kariryaa/NoMad/nomad-lab-base/parsers/gaussian/test/examples/Al.out","") must_== ParseResult.ParseSuccess
    }
  }
}
