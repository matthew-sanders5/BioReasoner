from bioreasoner import (
    FactSet,
    ReasoningEngine,
    build_default_wnt_rules,
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


def test_wnt_present_drives_beta_cat_up_with_signalosome_and_contradiction_to_baseline():
    engine = make_engine()
    initial = FactSet(
        [
            V.WNT_LIGAND_PRESENT,
            V.LRP6_PROTEIN_PRESENT,
            V.LRP6_SER_SITES_INTACT,
            V.FRIZZLED_PROTEIN_PRESENT,
            V.BETA_CAT_LEVEL_BASELINE,
        ]
    )

    result = engine.run(initial)

    # Signalosome should form
    assert V.SIGNALOSOME_STATE_FORMED in result.final_facts

    # LRP6 should be phosphorylated and active
    assert V.LRP6_SER_SITES_P in result.final_facts
    assert V.LRP6_SIGNALING_ACTIVE in result.final_facts

    # β-catenin should go up
    assert V.BETA_CAT_LEVEL_UP in result.final_facts

    # Contradiction between baseline and up should be detected
    assert (V.BETA_CAT_LEVEL_BASELINE, V.BETA_CAT_LEVEL_UP) in {
        tuple(sorted(pair)) for pair in result.contradictions
    }

def test_no_wnt_defaults_to_high_destruction_complex_and_beta_cat_down():
    engine = make_engine()
    initial = FactSet(
        [
            V.LRP6_PROTEIN_PRESENT,
            V.LRP6_SER_SITES_INTACT,
            V.FRIZZLED_PROTEIN_PRESENT,
            V.WNT_STATE_OFF,
            # Note: WNT_LIGAND_PRESENT is intentionally absent
        ]
    )

    result = engine.run(initial)

    # No Wnt -> DC should be HIGH due to baseline rule
    assert V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH in result.final_facts

    # DC_HIGH -> beta-catenin DOWN
    assert V.BETA_CAT_LEVEL_DOWN in result.final_facts

    # No signalosome, no LRP6 activation
    assert V.SIGNALOSOME_STATE_FORMED not in result.final_facts
    assert V.LRP6_SER_SITES_P not in result.final_facts
    assert V.LRP6_SIGNALING_ACTIVE not in result.final_facts

    # No contradictions: only DOWN state present
    assert result.contradictions == []

def test_destruction_complex_high_drives_beta_cat_down():
    engine = make_engine()
    initial = FactSet(
        [
            V.LRP6_PROTEIN_PRESENT,
            V.LRP6_SER_SITES_INTACT,
            V.FRIZZLED_PROTEIN_PRESENT,
            V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH,
            # Note: no Wnt, no initial beta-catenin state
        ]
    )

    result = engine.run(initial)

    # Destruction complex high should drive beta-catenin down.
    assert V.BETA_CAT_LEVEL_DOWN in result.final_facts

    # No other beta-catenin states inferred.
    assert V.BETA_CAT_LEVEL_BASELINE not in result.final_facts
    assert V.BETA_CAT_LEVEL_UP not in result.final_facts

    # No contradictions because we only have one beta-catenin state.
    assert result.contradictions == []

def test_wnt_signalosome_lowers_destruction_complex_and_flags_conflict_when_high():
    engine = make_engine()
    initial = FactSet(
        [
            V.WNT_LIGAND_PRESENT,
            V.LRP6_PROTEIN_PRESENT,
            V.LRP6_SER_SITES_INTACT,
            V.FRIZZLED_PROTEIN_PRESENT,
            V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH,
        ]
    )

    result = engine.run(initial)

    # Wnt pathway logic should form a signalosome and activate LRP6.
    assert V.SIGNALOSOME_STATE_FORMED in result.final_facts
    assert V.LRP6_SER_SITES_P in result.final_facts
    assert V.LRP6_SIGNALING_ACTIVE in result.final_facts

    # Signalosome should infer DESTRUCTION_COMPLEX_ACTIVITY_LOW.
    assert V.DESTRUCTION_COMPLEX_ACTIVITY_LOW in result.final_facts
    # Initial fact still present.
    assert V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH in result.final_facts

    # A contradiction between HIGH and LOW should be detected.
    assert (V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH, V.DESTRUCTION_COMPLEX_ACTIVITY_LOW) in {
        tuple(sorted(p)) for p in result.contradictions
    }
