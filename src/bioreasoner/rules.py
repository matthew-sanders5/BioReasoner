from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, List, Set

from .facts import FactSet
from . import vocab as V


@dataclass(frozen=True)
class Rule:
    """
    IF all `conditions` are present in the FactSet THEN add all `conclusions`.

    Monotonic: rules only add facts, never remove.
    """
    name: str
    conditions: FrozenSet[str]
    conclusions: FrozenSet[str]
    priority: int = 0
    tags: FrozenSet[str] = field(default_factory=frozenset)
    description: str | None = None
    citation: str | None = None

    @classmethod
    def from_sets(
        cls,
        name: str,
        conditions: Iterable[str],
        conclusions: Iterable[str],
        priority: int = 0,
        tags: Iterable[str] | None = None,
        description: str | None = None,
        citation: str | None = None,
    ) -> "Rule":
        return cls(
            name=name,
            conditions=frozenset(conditions),
            conclusions=frozenset(conclusions),
            priority=priority,
            tags=frozenset(tags or ()),
            description=description,
            citation=citation,
        )

    def is_applicable(self, facts: FactSet) -> bool:
        return all(cond in facts for cond in self.conditions)

    def new_facts_if_applied(self, facts: FactSet) -> Set[str]:
        """
        Return the subset of conclusions that are not yet in `facts`.
        """
        return {c for c in self.conclusions if c not in facts}


def sort_rules(rules: Iterable[Rule]) -> List[Rule]:
    """
    Deterministic ordering: higher priority first, then lexicographic by name.
    """
    return sorted(rules, key=lambda r: (-r.priority, r.name))


# ---- Domain-specific Wnt/LRP6/β-catenin logic (generic, literature-consistent) ----

def build_default_wnt_rules() -> list[Rule]:
    """
    Build the core rule set for the simple Wnt/LRP6/β-catenin universe.
    """
    rules: list[Rule] = []

    # 1) Wnt + Frizzled + LRP6 -> signalosome formed
    rules.append(
        Rule.from_sets(
            name="wnt_frizzled_lrp6_form_signalosome",
            conditions={
                V.WNT_LIGAND_PRESENT,
                V.LRP6_PROTEIN_PRESENT,
                V.FRIZZLED_PROTEIN_PRESENT,
            },
            conclusions={V.SIGNALOSOME_STATE_FORMED},
            priority=15,
            tags={"WNT", "LRP6", "FRIZZLED"},
            description="When Wnt ligand, LRP6, and Frizzled are present, a Wnt signalosome forms.",
        )
    )

    # 2) Signalosome + intact LRP6 serine sites -> LRP6 serine phosphorylation
    rules.append(
        Rule.from_sets(
            name="signalosome_drives_lrp6_ser_phosphorylation",
            conditions={
                V.SIGNALOSOME_STATE_FORMED,
                V.LRP6_SER_SITES_INTACT,
            },
            conclusions={V.LRP6_SER_SITES_P},
            priority=10,
            tags={"WNT", "LRP6"},
            description="A formed Wnt signalosome phosphorylates intact LRP6 serine sites.",
        )
    )

    # 3) Signalosome lowers destruction complex activity (Wnt ON)
    rules.append(
        Rule.from_sets(
            name="signalosome_lowers_destruction_complex_activity",
            conditions={V.SIGNALOSOME_STATE_FORMED},
            conclusions={V.DESTRUCTION_COMPLEX_ACTIVITY_LOW},
            priority=9,
            tags={"WNT", "DESTRUCTION_COMPLEX"},
            description="Wnt signalosome formation decreases destruction complex activity.",
        )
    )

    # 4) Serine-phosphorylated LRP6 -> LRP6 signaling active
    rules.append(
        Rule.from_sets(
            name="lrp6_active_when_p_s",
            conditions={V.LRP6_SER_SITES_P},
            conclusions={V.LRP6_SIGNALING_ACTIVE},
            priority=5,
            tags={"LRP6"},
            description="Serine-phosphorylated LRP6 is treated as an active Wnt co-receptor.",
        )
    )

    # 5) Active LRP6 signaling -> β-catenin up
    rules.append(
        Rule.from_sets(
            name="lrp6_active_drives_beta_cat_up",
            conditions={V.LRP6_SIGNALING_ACTIVE},
            conclusions={V.BETA_CAT_LEVEL_UP},
            priority=1,
            tags={"BETA_CAT"},
            description="Active LRP6 signaling increases beta-catenin levels.",
        )
    )

    # 6) High destruction complex activity -> β-catenin down
    rules.append(
        Rule.from_sets(
            name="destruction_complex_high_drives_beta_cat_down",
            conditions={V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH},
            conclusions={V.BETA_CAT_LEVEL_DOWN},
            priority=2,
            tags={"BETA_CAT", "DESTRUCTION_COMPLEX"},
            description="High destruction complex activity lowers beta-catenin levels.",
        )
    )

    # 7) Baseline: Wnt OFF + receptors present -> destruction complex HIGH
    rules.append(
        Rule.from_sets(
            name="baseline_destruction_complex_high_when_no_wnt",
            conditions={
                V.LRP6_PROTEIN_PRESENT,
                V.FRIZZLED_PROTEIN_PRESENT,
                V.WNT_STATE_OFF,
            },
            conclusions={V.DESTRUCTION_COMPLEX_ACTIVITY_HIGH},
            priority=-1,
            tags={"DESTRUCTION_COMPLEX"},
            description="In the absence of Wnt, with receptors present, the destruction complex is highly active.",
        )
    )


    # === PI3K/AKT PATHWAY RULES ===

    # 8) Growth factor + RTK present -> PI3K active
    rules.append(
        Rule.from_sets(
            name="gf_rtk_activate_pi3k",
            conditions={
                V.GROWTH_FACTOR_LIGAND_PRESENT,
                V.RTK_RECEPTOR_PRESENT,
            },
            conclusions={V.PI3K_STATE_ACTIVE},
            priority=15,
            tags={"PI3K", "RTK", "GROWTH_FACTOR"},
            description="When a growth factor ligand and its RTK receptor are present, PI3K becomes active.",
        )
    )

    # 9) PI3K active -> AKT phosphorylation at Thr308 and Ser473
    rules.append(
        Rule.from_sets(
            name="pi3k_active_phosphorylates_akt",
            conditions={V.PI3K_STATE_ACTIVE},
            conclusions={
                V.AKT_PHOSPHO_THR308_P,
                V.AKT_PHOSPHO_SER473_P,
            },
            priority=10,
            tags={"PI3K", "AKT"},
            description="Active PI3K signaling leads to AKT phosphorylation at Thr308 and Ser473.",
        )
    )

    # 10) Dual-phosphorylated AKT -> AKT active
    rules.append(
        Rule.from_sets(
            name="dual_phospho_akt_is_active",
            conditions={
                V.AKT_PHOSPHO_THR308_P,
                V.AKT_PHOSPHO_SER473_P,
            },
            conclusions={V.AKT_STATE_ACTIVE},
            priority=5,
            tags={"AKT"},
            description="AKT is considered active when both Thr308 and Ser473 sites are phosphorylated.",
        )
    )

    # 11) AKT active -> GSK3 inactive
    rules.append(
        Rule.from_sets(
            name="akt_active_inhibits_gsk3",
            conditions={V.AKT_STATE_ACTIVE},
            conclusions={V.GSK3_STATE_INACTIVE},
            priority=2,
            tags={"AKT", "GSK3"},
            description="Active AKT inhibits GSK3, rendering it inactive.",
        )
    )

    # 12) AKT active -> apoptotic tendency low
    rules.append(
        Rule.from_sets(
            name="akt_active_reduces_apoptosis_tendency",
            conditions={V.AKT_STATE_ACTIVE},
            conclusions={V.APOPTOSIS_TENDENCY_LOW},
            priority=1,
            tags={"AKT", "APOPTOSIS"},
            description="Active AKT signaling reduces apoptotic tendency (pro-survival effect).",
        )
    )

    # 13) Baseline RTK state without growth factor -> AKT inactive, GSK3 active, apoptosis high
    rules.append(
        Rule.from_sets(
            name="baseline_no_gf_leads_to_high_apoptosis",
            conditions={
                V.RTK_RECEPTOR_PRESENT,
                V.GROWTH_FACTOR_STATE_OFF,
            },
            conclusions={
                V.AKT_STATE_INACTIVE,
                V.GSK3_STATE_ACTIVE,
                V.APOPTOSIS_TENDENCY_HIGH,
            },
            priority=-1,
            tags={"RTK", "AKT", "GSK3", "APOPTOSIS"},
            description="Without growth factor, RTK is present but PI3K/AKT remain inactive, GSK3 is active, and apoptotic tendency is high.",
        )
    )

    # 14) GSK3 inactive contributes to low destruction complex activity (cross-talk)
    rules.append(
        Rule.from_sets(
            name="gsk3_inactive_lowers_destruction_complex_activity",
            conditions={V.GSK3_STATE_INACTIVE},
            conclusions={V.DESTRUCTION_COMPLEX_ACTIVITY_LOW},
            priority=3,
            tags={"GSK3", "DESTRUCTION_COMPLEX", "CROSSTALK"},
            description="When GSK3 is inactive (e.g., via AKT), destruction complex activity is inferred to be low.",
        )
    )

    return sort_rules(rules)
