"""
Tests for AHRAS Knowledge Graph Subsystem (Phase 4 / Sections 21, 22, 23, 47, 48)
----------------------------------------------------------------------------------
Validates:
  1. SecurityKnowledgeGraph semantic queries:
     - What enables detection of technique
     - Sensor gap identification
     - Detector impact under unhealthy sensor
     - Multi-stage attack path mitigation actions
  2. AttackFlow modeling:
     - Metric computations (stage, edge, entity completeness, temporal validity)
     - Full JSON export and import round-trip
  3. CampaignSimilarityEngine:
     - Multi-attribute similarity matching (Jaccard + LCS sequence + structural)
     - Strict attribution safety guard ("Never assert actor without evidence")
  4. VulnerabilityIntelligenceEngine:
     - Attack-path relevance elevating priority over static CVSS
  5. CaseBasedSecurityMemory:
     - Strict temporal causality guard (no future case data leakage)
     - Playbook response recommendation synthesis
"""

import json
import pytest
from pathlib import Path

from knowledge_graph.security_kg import (
    SecurityNodeType,
    SecurityEdgeType,
    SecurityNode,
    SecurityEdge,
    SecurityKnowledgeGraph,
)
from knowledge_graph.attack_flow import (
    AttackAction,
    AttackAsset,
    AttackFlow,
    AttackFlowEdge,
    AttackFlowMetrics,
)
from knowledge_graph.campaign_similarity import (
    IncidentProfile,
    CampaignSimilarityEngine,
    SimilarityMatchResult,
)
from knowledge_graph.vulnerability_intelligence import (
    NetworkZone,
    VulnerabilityRecord,
    AssetExposure,
    VulnerabilityIntelligenceEngine,
)
from knowledge_graph.case_memory import (
    HistoricalSecurityCase,
    CaseBasedSecurityMemory,
)


class TestSecurityKnowledgeGraph:
    def test_what_enables_detection(self):
        kg = SecurityKnowledgeGraph(populate_defaults=True)
        res = kg.what_enables_detection("tech-t1059")
        assert res["is_detected"] is True
        assert len(res["detection_rules"]) >= 1
        assert res["detection_rules"][0]["node_id"] == "rule-proc-002"
        assert "Process Execution & CmdLine" in res["required_telemetry"]
        assert "Host eBPF Tetragon Agent" in res["active_sensors"]

    def test_missing_sensors_query(self):
        kg = SecurityKnowledgeGraph(populate_defaults=True)
        # Windows ETW collector is seeded as inactive (is_active=False)
        missing = kg.find_missing_sensors("tech-t1059")
        assert len(missing) >= 1
        assert missing[0]["sensor_id"] == "sensor-win-etw"
        assert "unhealthy or inactive" in missing[0]["reason"]

    def test_sensor_unhealthy_impact(self):
        kg = SecurityKnowledgeGraph(populate_defaults=True)
        # If Zeek sensor becomes unhealthy, find affected detectors
        affected = kg.detections_affected_by_sensor("sensor-zeek")
        assert len(affected) >= 2
        detector_ids = [a["detector_id"] for a in affected]
        assert "rule-net-005" in detector_ids
        assert "model-iso-forest" in detector_ids

    def test_mitigating_responses_for_path(self):
        kg = SecurityKnowledgeGraph(populate_defaults=True)
        path = ["tech-t1110", "tech-t1059", "tech-t1486"]
        responses = kg.mitigating_responses_for_path(path)
        assert len(responses) == 3
        # tech-t1486 (Ransomware) must have host isolation mitigation
        ransom_mits = [m["action_name"] for m in responses[2]["mitigations"]]
        assert "Network Host Isolation" in ransom_mits


class TestAttackFlow:
    def test_attack_flow_metrics_and_temporal_validity(self):
        flow = AttackFlow(name="Test APT Flow")

        # Assets
        host1 = AttackAsset(asset_id="ast-1", name="web-server-01", asset_type="HOST", criticality=1.5)
        flow.add_asset(host1)

        # Actions in chronological order
        t0 = 1720000000.0
        act1 = AttackAction(
            action_id="act-1",
            name="Exploit Log4j",
            technique_id="T1190",
            technique_name="Exploit Public-Facing Application",
            tactic="Initial Access",
            timestamp=t0,
            confidence=0.98,
            epistemic_uncertainty=0.05,
            asset_ref="ast-1",
        )
        act2 = AttackAction(
            action_id="act-2",
            name="Execute Bash Payload",
            technique_id="T1059.004",
            technique_name="Unix Shell",
            tactic="Execution",
            timestamp=t0 + 2.5,
            confidence=0.95,
            epistemic_uncertainty=0.08,
            asset_ref="ast-1",
        )
        flow.add_action(act1)
        flow.add_action(act2)

        # Causal Edge
        flow.add_edge(source_id="act-1", target_id="act-2", edge_type="causes")

        metrics = flow.compute_completeness_metrics()
        assert metrics.action_count == 2
        assert metrics.asset_count == 1
        assert metrics.edge_count == 1
        assert metrics.temporal_ordering_valid is True
        assert metrics.technique_coverage_count == 2
        assert metrics.entity_completeness_pct == 100.0

    def test_attack_flow_json_export_import(self, tmp_path):
        flow = AttackFlow(name="Serialization Test")
        flow.add_action(
            AttackAction(
                action_id="a1",
                name="Scan",
                technique_id="T1046",
                technique_name="Network Service Discovery",
                tactic="Discovery",
                timestamp=100.0,
                confidence=0.9,
                epistemic_uncertainty=0.1,
            )
        )
        p = tmp_path / "flow.json"
        flow.export_json(p)
        assert p.exists()

        imported = AttackFlow.import_json(p)
        assert imported.name == "Serialization Test"
        assert len(imported.actions) == 1
        assert "a1" in imported.actions


class TestCampaignSimilarity:
    def test_similarity_matching_and_attribution_safety(self):
        engine = CampaignSimilarityEngine()

        # Historical Campaign 1: Ransomware Campaign
        hist_ransom = IncidentProfile(
            incident_id="camp-2025-01",
            name="LockBit Variant Incident",
            techniques=["T1190", "T1059", "T1021", "T1486"],
            affected_entities=["srv-web", "srv-db"],
            duration_seconds=1800.0,
            evidence_hashes=["hash-payload-x", "hash-script-y"],
            structural_features={"nodes": 4, "edges": 5},
            known_actor_indicator="FIN7",
        )
        engine.register_campaign(hist_ransom)

        # Historical Campaign 2: Cryptomining
        hist_crypto = IncidentProfile(
            incident_id="camp-2025-02",
            name="XMRig Miner Outbreak",
            techniques=["T1110", "T1059", "T1496"],
            affected_entities=["wkst-10"],
            duration_seconds=3600.0,
            evidence_hashes=["hash-miner-z"],
            structural_features={"nodes": 2, "edges": 1},
            known_actor_indicator="Unknown",
        )
        engine.register_campaign(hist_crypto)

        # Query Incident: Similar to Ransomware
        query = IncidentProfile(
            incident_id="inc-active-99",
            name="Active Ransomware Alert Cluster",
            techniques=["T1190", "T1059", "T1486"],
            affected_entities=["srv-web", "srv-db"],
            duration_seconds=1200.0,
            evidence_hashes=["hash-payload-diff"],  # Different hash!
            structural_features={"nodes": 4, "edges": 4},
            known_actor_indicator=None,
        )

        matches = engine.find_similar_campaigns(query, top_k=2)
        assert len(matches) >= 1
        top = matches[0]
        assert top.historical_incident_id == "camp-2025-01"
        assert top.composite_similarity > 0.60
        assert "T1190" in top.matching_techniques
        assert "T1486" in top.matching_techniques

        # SAFETY INVARIANT CHECK:
        # Without matching cryptographic hashes or identical actor indicator,
        # attribution MUST remain UNATTRIBUTED
        assert top.attribution_verdict == "UNATTRIBUTED"
        assert "actor attribution unconfirmed per AHRAS safety invariants" in top.attribution_rationale


class TestVulnerabilityIntelligence:
    def test_attack_path_relevance_elevates_priority(self):
        engine = VulnerabilityIntelligenceEngine()

        # Vuln 1: High static CVSS 9.8 on isolated server
        vuln_high_cvss = VulnerabilityRecord(
            cve_id="CVE-HIGH-ISOLATED",
            cvss_v3_base=9.8,
            epss_score=0.10,
            is_cisa_kev=False,
            affected_service="oracle-db",
            mitre_techniques=["T1190"],
        )
        engine.register_vulnerability(vuln_high_cvss)

        # Vuln 2: Moderate static CVSS 6.5, but actively exploited on CISA KEV and web server
        vuln_on_path = VulnerabilityRecord(
            cve_id="CVE-PATH-ACTIVE",
            cvss_v3_base=6.5,
            epss_score=0.85,
            is_cisa_kev=True,
            affected_service="apache-httpd",
            mitre_techniques=["T1190", "T1059"],
        )
        engine.register_vulnerability(vuln_on_path)

        # Asset 1: Isolated database (Tier 2, isolated zone)
        asset_isolated = AssetExposure(
            asset_id="ast-db-01",
            hostname="db-secure",
            network_zone=NetworkZone.ISOLATED_SECURE,
            criticality_tier=2,
            installed_vulnerabilities=["CVE-HIGH-ISOLATED"],
        )
        engine.register_asset(asset_isolated)

        # Asset 2: DMZ Web Server (Tier 1 crown jewel, public ingress)
        asset_web = AssetExposure(
            asset_id="ast-web-01",
            hostname="web-dmz-01",
            network_zone=NetworkZone.DMZ,
            criticality_tier=1,
            installed_vulnerabilities=["CVE-PATH-ACTIVE"],
            has_public_ingress=True,
        )
        engine.register_asset(asset_web)

        # Prioritize with web-dmz-01 on active attack path and active techniques
        prioritized = engine.prioritize_vulnerabilities(
            active_attack_path_entities={"ast-web-01", "web-dmz-01"},
            observed_techniques={"T1190", "T1059"},
        )

        assert len(prioritized) == 2
        # The moderate CVSS vulnerability on the active path must be ranked #1
        top_vuln = prioritized[0]
        assert top_vuln.cve_id == "CVE-PATH-ACTIVE"
        assert top_vuln.is_on_active_attack_path is True
        assert top_vuln.dynamic_risk_score > prioritized[1].dynamic_risk_score
        assert "Actively traversed lateral movement path" in top_vuln.justification or "actively executing matching techniques" in top_vuln.justification


class TestCaseBasedSecurityMemory:
    def test_strict_temporal_causality_guard(self):
        mem = CaseBasedSecurityMemory()

        t_eval = 1720001000.0  # Time of incident evaluation

        # Case 1: Closed in the past (valid)
        case_past = HistoricalSecurityCase(
            case_id="case-past-01",
            incident_id="inc-01",
            title="Past Lateral Movement Attack",
            techniques=["T1110", "T1021"],
            affected_entities=["host-a", "host-b"],
            timeline_start=t_eval - 10000,
            timeline_end=t_eval - 5000,
            closed_at=t_eval - 4000,  # In the past
            evidence_ids=["ev-1"],
            applied_responses=["ISOLATE_HOST"],
            outcome_verdict="SUCCESSFUL_CONTAINMENT",
            mean_risk_score=0.88,
        )
        mem.store_case(case_past)

        # Case 2: Closed in the future relative to t_eval (e.g. from future investigation)
        case_future = HistoricalSecurityCase(
            case_id="case-future-02",
            incident_id="inc-02",
            title="Future Lateral Movement Attack",
            techniques=["T1110", "T1021"],
            affected_entities=["host-a", "host-b"],
            timeline_start=t_eval + 500,
            timeline_end=t_eval + 1500,
            closed_at=t_eval + 2000,  # IN THE FUTURE!
            evidence_ids=["ev-2"],
            applied_responses=["REVOKE_CREDENTIALS"],
            outcome_verdict="SUCCESSFUL_CONTAINMENT",
            mean_risk_score=0.92,
        )
        mem.store_case(case_future)

        # Query at evaluation time t_eval
        retrieved = mem.retrieve_similar_cases(
            current_techniques=["T1110", "T1021"],
            current_entities=["host-a", "host-b"],
            query_timestamp=t_eval,
            k=5,
        )

        # STRICT TEMPORAL INVARIANT: Future case MUST NOT be returned!
        retrieved_ids = [c.case_id for c in retrieved]
        assert "case-past-01" in retrieved_ids
        assert "case-future-02" not in retrieved_ids

    def test_response_recommendation_synthesis(self):
        mem = CaseBasedSecurityMemory()
        t_now = 100000.0

        case = HistoricalSecurityCase(
            case_id="case-100",
            incident_id="inc-100",
            title="Ransomware Outbreak",
            techniques=["T1486"],
            affected_entities=["fileserver"],
            timeline_start=t_now - 500,
            timeline_end=t_now - 200,
            closed_at=t_now - 100,
            evidence_ids=["ev-100"],
            applied_responses=["ISOLATE_HOST", "KILL_PROCESS"],
            outcome_verdict="SUCCESSFUL_CONTAINMENT",
            mean_risk_score=0.95,
        )
        mem.store_case(case)

        retrieved = mem.retrieve_similar_cases(
            current_techniques=["T1486"],
            current_entities=["fileserver"],
            query_timestamp=t_now,
        )
        rec = mem.synthesize_response_recommendation(retrieved)
        assert "ISOLATE_HOST" in rec["recommended_actions"]
        assert rec["confidence"] > 0.8
        assert rec["grounding_case_count"] == 1
