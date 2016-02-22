package eu.nomad_lab.parsers

import eu.nomad_lab.{parsers, DefaultPythonInterpreter}
import org.scalacheck.Properties
import org.specs2.mutable.Specification
import org.{json4s => jn}


object GaussianParserSpec extends Specification {
  "GaussianParserTest" >> {
    "test with Al.out">> {
      "test with json-events" >> {
        ParserRun.parse(GaussianParser,"parsers/gaussian/test/examples/Al.out","json-events") must_== ParseResult.ParseSuccess
      }
      "test with json" >> {
        ParserRun.parse(GaussianParser,"parsers/gaussian/test/examples/Al.out","json") must_== ParseResult.ParseSuccess
      }
    }
  }
}
