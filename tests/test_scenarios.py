from pathlib import Path

from bioreasoner import (
    ReasoningEngine,
    build_default_wnt_rules,
    load_scenario,
    vocab as V,
)

DEFAULT_CONTRADICTIONS = [
    (V.BETA_CAT_LEVEL_UP, V.BETA_CAT_LEVEL_BASELINE),
    (V.BETA_CAT_LEVEL_UP, V.BETA_CAT_LEVEL_DOWN),
    (V.BETA_CAT_LEVEL_BASELINE, V.BETA_CAT_LEVEL_DOWN),
    (V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH, V.DESTRUCTION_COMPLEX_ACTIVITY_LOW),
]

def make_engine():
    rules = build_default_wnt_rules()
    return ReasoningEngine(rules=rules, contradiction_pairs=DEFAULT_CONTRADICTIONS)


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"

def test_wnt_demo_scenario_behaves_as_expected():
    engine = make_engine()
    scenario = load_scenario(SCENARIOS / "wnt_demo.yaml")

    result = engine.run(scenario.initial_facts)

    # Wnt demo should form a signalosome, phosphorylate LRP6, and drive beta-catenin up.
    assert V.SIGNALOSOME_STATE_FORMED in result.final_facts
    assert V.LRP6_SER_SITES_P in result.final_facts
    assert V.LRP6_SIGNALING_ACTIVE in result.final_facts
    assert V.BETA_CAT_LEVEL_UP in result.final_facts

    # Contradiction between baseline and up should be detected.
    assert (V.BETA_CAT_LEVEL_BASELINE, V.BETA_CAT_LEVEL_UP) in {
        tuple(sorted(p)) for p in result.contradictions
    }

    # Queries from YAML should resolve accordingly.
    assert "BETA_CAT__LEVEL__UP" in scenario.queries
    assert "BETA_CAT__LEVEL__DOWN" in scenario.queries

    assert V.BETA_CAT_LEVEL_UP in result.final_facts
    assert V.BETA_CAT_LEVEL_DOWN not in result.final_facts


def test_no_wnt_scenario_defaults_to_destruction_complex_high_and_beta_cat_down():
    engine = make_engine()
    scenario = load_scenario(SCENARIOS / "no_wnt_baseline.yaml")

    result = engine.run(scenario.initial_facts)

    # Baseline Wnt-OFF should infer destruction complex HIGH
    assert V.WNT_STATE_OFF in result.final_facts
    assert V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH in result.final_facts

    # This should drive beta-catenin DOWN
    assert V.BETA_CAT_LEVEL_DOWN in result.final_facts

    # No activation
    assert V.SIGNALOSOME_STATE_FORMED not in result.final_facts
    assert V.LRP6_SIGNALING_ACTIVE not in result.final_facts

    # No contradictions
    assert result.contradictions == []


def test_mut_lrp6_scenario_prevents_activation():
    engine = make_engine()
    scenario = load_scenario(SCENARIOS / "mut_lrp6_no_activation.yaml")

    result = engine.run(scenario.initial_facts)

    # Wnt present, LRP6 and Frizzled present -> signalosome can form.
    assert V.WNT_LIGAND_PRESENT in result.final_facts
    assert V.LRP6_PROTEIN_PRESENT in result.final_facts
    assert V.FRIZZLED_PROTEIN_PRESENT in result.final_facts
    assert V.SIGNALOSOME_STATE_FORMED in result.final_facts

    # But serine sites are mutated, so no phosphorylation or activation.
    assert V.LRP6_SER_SITES_P not in result.final_facts
    assert V.LRP6_SIGNALING_ACTIVE not in result.final_facts

    # Beta-catenin remains baseline.
    assert V.BETA_CAT_LEVEL_BASELINE in result.final_facts
    assert V.BETA_CAT_LEVEL_UP not in result.final_facts
    assert result.contradictions == []

def test_high_destruction_complex_scenario_drives_beta_cat_down():
    engine = make_engine()
    scenario = load_scenario(SCENARIOS / "destruction_complex_high.yaml")

    result = engine.run(scenario.initial_facts)

    # Destruction complex high -> beta-catenin down.
    assert V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH in result.final_facts
    assert V.BETA_CAT_LEVEL_DOWN in result.final_facts

    # No baseline/up states inferred.
    assert V.BETA_CAT_LEVEL_BASELINE not in result.final_facts
    assert V.BETA_CAT_LEVEL_UP not in result.final_facts

    # No contradictions: only one beta-catenin level is present.
    assert result.contradictions == []

def test_wnt_with_high_destruction_complex_scenario_flags_dc_conflict():
    engine = make_engine()
    scenario = load_scenario(SCENARIOS / "wnt_with_high_destruction_complex.yaml")

    result = engine.run(scenario.initial_facts)

    # Wnt should still drive signalosome, LRP6 activation, and beta-catenin up.
    assert V.SIGNALOSOME_STATE_FORMED in result.final_facts
    assert V.LRP6_SER_SITES_P in result.final_facts
    assert V.LRP6_SIGNALING_ACTIVE in result.final_facts
    assert V.BETA_CAT_LEVEL_UP in result.final_facts

    # Both HIGH and LOW destruction complex states present.
    assert V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH in result.final_facts
    assert V.DESTRUCTION_COMPLEX_ACTIVITY_LOW in result.final_facts

    # Contradiction between HIGH and LOW should be detected.
    assert (V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH, V.DESTRUCTION_COMPLEX_ACTIVITY_LOW) in {
        tuple(sorted(p)) for p in result.contradictions
    }
