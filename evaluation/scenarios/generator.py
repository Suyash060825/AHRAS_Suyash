"""
Combinatorial (2-wise/3-wise) scenario generator and Monte Carlo sampling engine.
"""

import itertools
import random
from typing import List, Dict, Any
from .schema import (
    ScenarioSpecification,
    EnvironmentType,
    AttackFamily,
    NoiseLevel,
    TelemetryQuality,
    ScenarioStage,
    ScenarioAssetContext,
    ScenarioIdentityContext
)

class CombinatorialScenarioGenerator:
    """Generates pairwise and 3-way combinatorial test scenarios across operational dimensions."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

    def generate_pairwise_scenarios(self) -> List[ScenarioSpecification]:
        """Generates scenarios covering all pairs of (Environment, AttackFamily, NoiseLevel)."""
        environments = list(EnvironmentType)
        attack_families = [
            AttackFamily.RECONNAISSANCE,
            AttackFamily.BRUTE_FORCE,
            AttackFamily.EXPLOITATION,
            AttackFamily.DENIAL_OF_SERVICE,
            AttackFamily.LATERAL_MOVEMENT,
            AttackFamily.EXFILTRATION,
        ]
        noise_levels = [NoiseLevel.LOW, NoiseLevel.MEDIUM, NoiseLevel.HIGH]

        # Simple combinatorial IPO/AETG-equivalent pair covering generator
        all_combinations = list(itertools.product(environments, attack_families, noise_levels))
        # Subsample/stratify if necessary or keep compact set (5 * 6 * 3 = 90 test cases)
        scenarios = []
        for idx, (env, atk, noise) in enumerate(all_combinations):
            spec = self._build_scenario(
                scenario_id=f"COMB-2W-{idx+1:03d}",
                name=f"Pairwise: {env.value} | {atk.value} | {noise.value}",
                env=env,
                atk=atk,
                noise=noise,
                telemetry=TelemetryQuality.PRISTINE,
                seed=self.random_seed + idx
            )
            scenarios.append(spec)
        return scenarios

    def generate_3way_scenarios(self, max_cases: int = 30) -> List[ScenarioSpecification]:
        """Generates 3-way interactions including degraded telemetry."""
        environments = [EnvironmentType.ENTERPRISE_IT, EnvironmentType.CLOUD_HYBRID, EnvironmentType.OT_ICS]
        attack_families = [AttackFamily.EXPLOITATION, AttackFamily.LATERAL_MOVEMENT, AttackFamily.COMMAND_AND_CONTROL]
        noise_levels = [NoiseLevel.LOW, NoiseLevel.HIGH]
        telemetry_qualities = [TelemetryQuality.PRISTINE, TelemetryQuality.DEGRADED_20, TelemetryQuality.INTERMITTENT]

        combos = list(itertools.product(environments, attack_families, noise_levels, telemetry_qualities))
        self.rng.shuffle(combos)
        selected = combos[:max_cases]

        scenarios = []
        for idx, (env, atk, noise, tel) in enumerate(selected):
            spec = self._build_scenario(
                scenario_id=f"COMB-3W-{idx+1:03d}",
                name=f"3-Way: {env.value} | {atk.value} | {noise.value} | {tel.value}",
                env=env,
                atk=atk,
                noise=noise,
                telemetry=tel,
                seed=self.random_seed + 1000 + idx
            )
            scenarios.append(spec)
        return scenarios

    def _build_scenario(
        self,
        scenario_id: str,
        name: str,
        env: EnvironmentType,
        atk: AttackFamily,
        noise: NoiseLevel,
        telemetry: TelemetryQuality,
        seed: int
    ) -> ScenarioSpecification:
        assets = [
            ScenarioAssetContext(asset_id="src-01", ip_address="192.168.1.50", hostname="workstation-01", criticality=0.4),
            ScenarioAssetContext(asset_id="target-01", ip_address="10.0.0.15", hostname="srv-core-db", criticality=0.9),
        ]
        identities = [
            ScenarioIdentityContext(user_id="user-corp", role="analyst", credential_level="standard")
        ]
        stages = [
            ScenarioStage(
                stage_index=0,
                attack_family=None,
                description="Benign baseline activity",
                duration_seconds=60.0,
                event_rate_eps=10.0,
                source_asset_id="src-01",
                target_asset_id="target-01",
                is_malicious=False
            ),
            ScenarioStage(
                stage_index=1,
                attack_family=atk,
                description=f"Injected {atk.value} attack execution",
                duration_seconds=120.0,
                event_rate_eps=35.0,
                source_asset_id="src-01",
                target_asset_id="target-01",
                is_malicious=True
            ),
        ]
        return ScenarioSpecification(
            scenario_id=scenario_id,
            name=name,
            scenario_type="SYNTHETIC",
            environment_type=env,
            stages=stages,
            assets=assets,
            identities=identities,
            noise_level=noise,
            telemetry_quality=telemetry,
            duration_total_seconds=180.0,
            random_seed=seed,
            metadata={"generator": "CombinatorialScenarioGenerator", "version": "1.0"}
        )
