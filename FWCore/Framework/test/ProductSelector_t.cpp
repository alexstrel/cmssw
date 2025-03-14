#include "catch.hpp"
#include <iostream>
#include <string>
#include <vector>

#include <memory>

#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"

typedef std::vector<edm::ProductDescription const*> VCBDP;

void apply_gs(edm::ProductSelector const& gs, VCBDP const& allbranches, std::vector<bool>& results) {
  VCBDP::const_iterator it = allbranches.begin();
  VCBDP::const_iterator end = allbranches.end();
  for (; it != end; ++it)
    results.push_back(gs.selected(**it));
}

void doTest(edm::ParameterSet const& params,
            char const* testname,
            VCBDP const& allbranches,
            std::vector<bool>& expected) {
  edm::ProductSelectorRules gsr(params, "outputCommands", testname);
  edm::ProductSelector gs;
  gs.initialize(gsr, allbranches);

  std::vector<bool> results;
  apply_gs(gs, allbranches, results);

  CHECK(expected == results);
}

TEST_CASE("test ProductSelector", "[ProductSelector]") {
  edm::ParameterSet dummyProcessPset;
  dummyProcessPset.registerIt();
  auto processConfiguration = std::make_shared<edm::ProcessConfiguration>();
  processConfiguration->setParameterSetID(dummyProcessPset.id());

  edm::TypeWithDict dummyTypeWithDict;
  // We pretend to have one module, with two products. The products
  // are of the same and, type differ in instance name.
  std::set<edm::ParameterSetID> psetsA;
  edm::ParameterSet modAparams;
  modAparams.addParameter<int>("i", 2112);
  modAparams.addParameter<std::string>("s", "hi");
  modAparams.registerIt();
  psetsA.insert(modAparams.id());

  edm::ProductDescription b1(edm::InEvent, "modA", "PROD", "UglyProdTypeA", "ProdTypeA", "i1", dummyTypeWithDict);
  edm::ProductDescription b2(edm::InEvent, "modA", "PROD", "UglyProdTypeA", "ProdTypeA", "i2", dummyTypeWithDict);

  // Our second pretend module has only one product, and gives it no
  // instance name.
  std::set<edm::ParameterSetID> psetsB;
  edm::ParameterSet modBparams;
  modBparams.addParameter<double>("d", 2.5);
  modBparams.registerIt();
  psetsB.insert(modBparams.id());

  edm::ProductDescription b3(edm::InEvent, "modB", "HLT", "UglyProdTypeB", "ProdTypeB", "", dummyTypeWithDict);

  // Our third pretend is like modA, except it hass processName_ of
  // "USER"

  edm::ProductDescription b4(edm::InEvent, "modA", "USER", "UglyProdTypeA", "ProdTypeA", "i1", dummyTypeWithDict);
  edm::ProductDescription b5(edm::InEvent, "modA", "USER", "UglyProdTypeA", "ProdTypeA", "i2", dummyTypeWithDict);

  // These are pointers to all the branches that are available. In a
  // framework program, these would come from the ProductRegistry
  // which is used to initialze the OutputModule being configured.
  VCBDP allbranches;
  allbranches.push_back(&b1);  // ProdTypeA_modA_i1. (PROD)
  allbranches.push_back(&b2);  // ProdTypeA_modA_i2. (PROD)
  allbranches.push_back(&b3);  // ProdTypeB_modB_HLT. (no instance name)
  allbranches.push_back(&b4);  // ProdTypeA_modA_i1_USER.
  allbranches.push_back(&b5);  // ProdTypeA_modA_i2_USER.

  // Test default parameters
  SECTION("default parameters") {
    bool wanted[] = {true, true, true, true, true};
    std::vector<bool> expected(wanted, wanted + sizeof(wanted) / sizeof(bool));
    edm::ParameterSet noparams;

    doTest(noparams, "default parameters", allbranches, expected);
  }

  // Keep all branches with instance name i2.
  SECTION("keep_i2 parameters") {
    bool wanted[] = {false, true, false, false, true};
    std::vector<bool> expected(wanted, wanted + sizeof(wanted) / sizeof(bool));

    edm::ParameterSet keep_i2;
    std::string const keep_i2_rule = "keep *_*_i2_*";
    std::vector<std::string> cmds;
    cmds.push_back(keep_i2_rule);
    keep_i2.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    keep_i2.registerIt();

    doTest(keep_i2, "keep_i2 parameters", allbranches, expected);
  }

  // Drop all branches with instance name i2.
  SECTION("drop_i2 parameters") {
    bool wanted[] = {true, false, true, true, false};
    std::vector<bool> expected(wanted, wanted + sizeof(wanted) / sizeof(bool));

    edm::ParameterSet drop_i2;
    std::string const drop_i2_rule1 = "keep *";
    std::string const drop_i2_rule2 = "drop *_*_i2_*";
    std::vector<std::string> cmds;
    cmds.push_back(drop_i2_rule1);
    cmds.push_back(drop_i2_rule2);
    drop_i2.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    drop_i2.registerIt();

    doTest(drop_i2, "drop_i2 parameters", allbranches, expected);
  }

  // Now try dropping all branches with product type "foo". There are
  // none, so all branches should be written.
  SECTION("drop_foo parameters") {
    bool wanted[] = {true, true, true, true, true};
    std::vector<bool> expected(wanted, wanted + sizeof(wanted) / sizeof(bool));

    edm::ParameterSet drop_foo;
    std::string const drop_foo_rule1 = "keep *_*_*_*";  // same as "keep *"
    std::string const drop_foo_rule2 = "drop foo_*_*_*";
    std::vector<std::string> cmds;
    cmds.push_back(drop_foo_rule1);
    cmds.push_back(drop_foo_rule2);
    drop_foo.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    drop_foo.registerIt();

    doTest(drop_foo, "drop_foo parameters", allbranches, expected);
  }

  // Now try dropping all branches with product type "ProdTypeA".
  SECTION("drop_ProdTypeA") {
    bool wanted[] = {false, false, true, false, false};
    std::vector<bool> expected(wanted, wanted + sizeof(wanted) / sizeof(bool));

    edm::ParameterSet drop_ProdTypeA;
    std::string const drop_ProdTypeA_rule1 = "keep *";
    std::string const drop_ProdTypeA_rule2 = "drop ProdTypeA_*_*_*";
    std::vector<std::string> cmds;
    cmds.push_back(drop_ProdTypeA_rule1);
    cmds.push_back(drop_ProdTypeA_rule2);
    drop_ProdTypeA.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    drop_ProdTypeA.registerIt();

    doTest(drop_ProdTypeA, "drop_ProdTypeA", allbranches, expected);
  }

  // Keep only branches with instance name 'i1', from Production.
  SECTION("keep_i1prod") {
    bool wanted[] = {true, false, false, false, false};
    std::vector<bool> expected(wanted, wanted + sizeof(wanted) / sizeof(bool));

    edm::ParameterSet keep_i1prod;
    std::string const keep_i1prod_rule = "keep *_*_i1_PROD";
    std::vector<std::string> cmds;
    cmds.push_back(keep_i1prod_rule);
    keep_i1prod.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    keep_i1prod.registerIt();

    doTest(keep_i1prod, "keep_i1prod", allbranches, expected);
  }

  // First say to keep everything,  then  to drop everything, then  to
  // keep it again. The end result should be to keep everything.
  SECTION("indecisive") {
    bool wanted[] = {true, true, true, true, true};
    std::vector<bool> expected(wanted, wanted + sizeof(wanted) / sizeof(bool));

    edm::ParameterSet indecisive;
    std::string const indecisive_rule1 = "keep *";
    std::string const indecisive_rule2 = "drop *";
    std::string const indecisive_rule3 = "keep *";
    std::vector<std::string> cmds;
    cmds.push_back(indecisive_rule1);
    cmds.push_back(indecisive_rule2);
    cmds.push_back(indecisive_rule3);
    indecisive.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    indecisive.registerIt();

    doTest(indecisive, "indecisive", allbranches, expected);
  }

  // Keep all things, bu drop all things from modA, but later keep all
  // things from USER.
  SECTION("drop_modA_keep_user") {
    bool wanted[] = {false, false, true, true, true};
    std::vector<bool> expected(wanted, wanted + sizeof(wanted) / sizeof(bool));

    edm::ParameterSet params;
    std::string const rule1 = "keep *";
    std::string const rule2 = "drop *_modA_*_*";
    std::string const rule3 = "keep *_*_*_USER";
    std::vector<std::string> cmds;
    cmds.push_back(rule1);
    cmds.push_back(rule2);
    cmds.push_back(rule3);
    params.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    params.registerIt();

    doTest(params, "drop_modA_keep_user", allbranches, expected);
  }

  // Exercise the wildcards * and ?
  SECTION("excercise wildcards1") {
    bool wanted[] = {true, true, true, false, false};
    std::vector<bool> expected(wanted, wanted + sizeof(wanted) / sizeof(bool));

    edm::ParameterSet params;
    std::string const rule1 = "drop *";
    std::string const rule2 = "keep Pr*A_m?dA_??_P?O*";
    std::string const rule3 = "keep *?*?***??*????*?***_??***?__*?***T";
    std::vector<std::string> cmds;
    cmds.push_back(rule1);
    cmds.push_back(rule2);
    cmds.push_back(rule3);
    params.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    params.registerIt();

    doTest(params, "excercise wildcards1", allbranches, expected);
  }

  SECTION("illegal") {
    // Now try an illegal specification: not starting with 'keep' or 'drop'
    edm::ParameterSet bad;
    std::string const bad_rule = "beep *_*_i2_*";
    std::vector<std::string> cmds;
    cmds.push_back(bad_rule);
    bad.addUntrackedParameter<std::vector<std::string> >("outputCommands", cmds);
    bad.registerIt();
    REQUIRE_THROWS_MATCHES(
        edm::ProductSelectorRules(bad, "outputCommands", "ProductSelectorTest"),
        edm::Exception,
        Catch::Predicate<edm::Exception>([](auto const& x) { return x.categoryCode() == edm::errors::Configuration; }));
  }
}
