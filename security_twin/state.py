from __future__ import annotations
"""
AHRAS Module / Digital Twin — State Management & Enterprise Topology
---------------------------------------------------------------------
Maintains a stateful representation of enterprise infrastructure, assets,
identities, process lineages, network connectivity, and active controls.
Supports atomic state capture, rollback, and formal state diffing.
"""

import copy
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from security_twin.models import (
    Host, User, Process, File, Network, Asset, Service, Identity,
    SecurityControl, SecurityTwinSnapshot
)


class SecurityTwin:
    """
    Stateful digital twin of enterprise network, hosts, identities, and controls.
    Enables safe dry-run counterfactual simulation of attack progression and mitigations.
    """

    def __init__(self, twin_id: Optional[str] = None):
        self.twin_id: str = twin_id or f"twin-{uuid.uuid4().hex[:8]}"
        self.hosts: Dict[str, Host] = {}
        self.users: Dict[str, User] = {}
        self.processes: Dict[int, Process] = {}
        self.files: Dict[str, File] = {}
        self.networks: Dict[str, Network] = {}
        self.assets: Dict[str, Asset] = {}
        self.services: Dict[str, Service] = {}
        self.identities: Dict[str, Identity] = {}
        self.controls: Dict[str, SecurityControl] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.risk_state: Dict[str, float] = {}
        self.model_versions: Dict[str, str] = {
            "encoder": "v6.1.0",
            "risk_engine": "v6.1.0",
            "conformal_gate": "v6.1.0",
        }
        self.response_policies: Dict[str, Any] = {
            "mode": "DRY_RUN",
            "require_analyst_approval": True,
            "max_blast_radius_autonomous": 0.35,
        }

    # ── Entity Registration ──────────────────────────────────────────────────

    def add_host(self, host: Host) -> None:
        self.hosts[host.host_id] = host

    def add_user(self, user: User) -> None:
        self.users[user.user_id] = user

    def add_process(self, process: Process) -> None:
        self.processes[process.pid] = process

    def add_file(self, file_entity: File) -> None:
        self.files[file_entity.path] = file_entity

    def add_network(self, network: Network) -> None:
        self.networks[network.network_id] = network

    def add_asset(self, asset: Asset) -> None:
        self.assets[asset.asset_id] = asset

    def add_service(self, service: Service) -> None:
        self.services[service.service_id] = service

    def add_identity(self, identity: Identity) -> None:
        self.identities[identity.token_id] = identity

    def add_control(self, control: SecurityControl) -> None:
        self.controls[control.control_id] = control

    def add_relationship(self, rel_type: str, src: str, dst: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.relationships.append({
            "rel": rel_type,
            "src": src,
            "dst": dst,
            "metadata": metadata or {},
        })

    # ── Entity Query & State Helpers ─────────────────────────────────────────

    def get_host(self, host_id: str) -> Optional[Host]:
        return self.hosts.get(host_id)

    def get_host_by_ip(self, ip: str) -> Optional[Host]:
        for h in self.hosts.values():
            if h.ip_address == ip:
                return h
        return None

    def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    def get_user_by_name(self, username: str) -> Optional[User]:
        for u in self.users.values():
            if u.username == username:
                return u
        return None

    def get_process(self, pid: int) -> Optional[Process]:
        return self.processes.get(pid)

    def get_identity(self, token_id: str) -> Optional[Identity]:
        return self.identities.get(token_id)

    def is_ip_blocked(self, ip: str) -> bool:
        for net in self.networks.values():
            if ip in net.blocked_ips:
                return True
        for ctrl in self.controls.values():
            if ctrl.is_active and ctrl.control_type == "FIREWALL":
                if ip in ctrl.rules.get("blocked_ips", []):
                    return True
        return False

    def is_host_isolated(self, host_id_or_ip: str) -> bool:
        h = self.hosts.get(host_id_or_ip) or self.get_host_by_ip(host_id_or_ip)
        if h and h.is_isolated:
            return True
        return False

    def is_process_terminated(self, pid: int) -> bool:
        proc = self.processes.get(pid)
        return proc.is_terminated if proc else True

    def is_token_revoked(self, token_id: str) -> bool:
        ident = self.identities.get(token_id)
        if ident:
            return not ident.is_valid
        return True

    # ── Topology & Blast Radius Metrics ──────────────────────────────────────

    def get_services_on_host(self, host_id: str) -> List[Service]:
        return [s for s in self.services.values() if s.host_id == host_id]

    def get_processes_on_host(self, host_id: str) -> List[Process]:
        return [p for p in self.processes.values() if p.host_id == host_id and not p.is_terminated]

    def compute_host_blast_radius(self, host_id: str) -> float:
        """
        Computes the collateral blast radius of isolating a host:
          blast_radius = 0.5 * criticality + 0.3 * (services_count / 10) + 0.2 * asset_impact
        Normalized to [0.0, 1.0].
        """
        host = self.hosts.get(host_id)
        if not host:
            return 0.1
        crit = host.criticality
        services = self.get_services_on_host(host_id)
        srv_factor = min(1.0, len(services) / 5.0)

        # Assets linked to this host
        linked_assets = [a for a in self.assets.values() if a.host_id == host_id]
        asset_crit = max([a.business_criticality for a in linked_assets], default=0.0)

        blast = (0.5 * crit) + (0.25 * srv_factor) + (0.25 * asset_crit)
        return min(1.0, max(0.05, float(blast)))

    def compute_process_blast_radius(self, pid: int) -> float:
        """
        Computes blast radius of terminating a process based on child processes and service roles.
        """
        proc = self.processes.get(pid)
        if not proc:
            return 0.05
        # Check if process is associated with a critical service
        host = self.hosts.get(proc.host_id)
        host_crit = host.criticality if host else 0.3
        
        # Check child processes
        child_count = sum(1 for p in self.processes.values() if p.parent_pid == pid and not p.is_terminated)
        child_factor = min(1.0, child_count / 4.0)

        is_svc = any(s.name.lower() in proc.process_name.lower() for s in self.services.values())
        svc_penalty = 0.3 if is_svc else 0.0

        blast = (0.4 * host_crit) + (0.3 * child_factor) + svc_penalty
        return min(1.0, max(0.05, float(blast)))

    # ── Snapshot, Restore, Diff ──────────────────────────────────────────────

    def capture_state(self) -> SecurityTwinSnapshot:
        """Captures a deep-copied, immutable snapshot of the digital twin state."""
        entities_dump = {
            "hosts": {k: v.to_dict() for k, v in self.hosts.items()},
            "users": {k: v.to_dict() for k, v in self.users.items()},
            "processes": {k: v.to_dict() for k, v in self.processes.items()},
            "files": {k: v.to_dict() for k, v in self.files.items()},
            "networks": {k: v.to_dict() for k, v in self.networks.items()},
            "assets": {k: v.to_dict() for k, v in self.assets.items()},
            "services": {k: v.to_dict() for k, v in self.services.items()},
            "identities": {k: v.to_dict() for k, v in self.identities.items()},
        }
        return SecurityTwinSnapshot(
            snapshot_id=f"snap-{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            entities=entities_dump,
            relationships=copy.deepcopy(self.relationships),
            active_controls={k: v.to_dict() for k, v in self.controls.items()},
            risk_state=copy.deepcopy(self.risk_state),
            model_versions=copy.deepcopy(self.model_versions),
            response_policies=copy.deepcopy(self.response_policies),
        )

    def restore_state(self, snapshot: SecurityTwinSnapshot) -> None:
        """Restores the digital twin state precisely from a captured snapshot."""
        e = snapshot.entities
        self.hosts = {k: Host(**v) for k, v in e.get("hosts", {}).items()}
        self.users = {k: User(**v) for k, v in e.get("users", {}).items()}
        self.processes = {int(k): Process(**v) for k, v in e.get("processes", {}).items()}
        self.files = {k: File(**v) for k, v in e.get("files", {}).items()}
        self.networks = {k: Network(**v) for k, v in e.get("networks", {}).items()}
        self.assets = {k: Asset(**v) for k, v in e.get("assets", {}).items()}
        self.services = {k: Service(**v) for k, v in e.get("services", {}).items()}
        self.identities = {k: Identity(**v) for k, v in e.get("identities", {}).items()}

        self.controls = {k: SecurityControl(**v) for k, v in snapshot.active_controls.items()}
        self.relationships = copy.deepcopy(snapshot.relationships)
        self.risk_state = copy.deepcopy(snapshot.risk_state)
        self.model_versions = copy.deepcopy(snapshot.model_versions)
        self.response_policies = copy.deepcopy(snapshot.response_policies)

    @staticmethod
    def diff_state(snap1: SecurityTwinSnapshot, snap2: SecurityTwinSnapshot) -> Dict[str, Any]:
        """
        Computes formal differences between two snapshots:
          - modified entity fields (e.g. is_isolated, is_terminated, blocked_ips)
          - risk state deltas
          - control modifications
        """
        diff: Dict[str, Any] = {
            "entity_modifications": {},
            "risk_deltas": {},
            "control_modifications": {},
        }

        # Entities diff
        for category in ["hosts", "users", "processes", "networks", "identities"]:
            e1 = snap1.entities.get(category, {})
            e2 = snap2.entities.get(category, {})
            all_keys = set(e1.keys()).union(set(e2.keys()))
            for k in all_keys:
                if k not in e1:
                    diff["entity_modifications"][f"{category}:{k}"] = {"status": "ADDED", "new": e2[k]}
                elif k not in e2:
                    diff["entity_modifications"][f"{category}:{k}"] = {"status": "REMOVED", "old": e1[k]}
                elif e1[k] != e2[k]:
                    changed = {
                        field_name: (e1[k].get(field_name), e2[k].get(field_name))
                        for field_name in set(e1[k].keys()).union(e2[k].keys())
                        if e1[k].get(field_name) != e2[k].get(field_name)
                    }
                    diff["entity_modifications"][f"{category}:{k}"] = {"status": "MODIFIED", "changes": changed}

        # Risk deltas
        r1 = snap1.risk_state
        r2 = snap2.risk_state
        for k in set(r1.keys()).union(set(r2.keys())):
            v1 = r1.get(k, 0.0)
            v2 = r2.get(k, 0.0)
            if abs(v1 - v2) > 1e-4:
                diff["risk_deltas"][k] = {"before": v1, "after": v2, "delta": round(v2 - v1, 4)}

        # Controls diff
        c1 = snap1.active_controls
        c2 = snap2.active_controls
        for k in set(c1.keys()).union(set(c2.keys())):
            if c1.get(k) != c2.get(k):
                diff["control_modifications"][k] = {"before": c1.get(k), "after": c2.get(k)}

        return diff


def create_enterprise_test_twin() -> SecurityTwin:
    """Creates a standardized enterprise digital twin topology for validation and benchmarking."""
    twin = SecurityTwin(twin_id="twin-enterprise-eval")

    # Networks
    net_dmz = Network(network_id="net-dmz", cidr="10.0.1.0/24", zone="DMZ")
    net_internal = Network(network_id="net-internal", cidr="10.0.2.0/24", zone="INTERNAL")
    net_prod = Network(network_id="net-prod", cidr="10.0.3.0/24", zone="PROD")
    twin.add_network(net_dmz)
    twin.add_network(net_internal)
    twin.add_network(net_prod)

    # Hosts
    host_web = Host(
        host_id="host-web-01",
        hostname="web-prod-01",
        ip_address="10.0.1.10",
        os_type="linux",
        criticality=0.60,
        services=["nginx", "sshd"],
        tags=["web", "dmz"],
    )
    host_bastion = Host(
        host_id="host-bastion-01",
        hostname="bastion-01",
        ip_address="10.0.1.5",
        os_type="linux",
        criticality=0.70,
        services=["sshd"],
        tags=["bastion", "dmz"],
    )
    host_file = Host(
        host_id="host-file-01",
        hostname="fs-internal-01",
        ip_address="10.0.2.20",
        os_type="linux",
        criticality=0.75,
        services=["smbd", "sshd"],
        tags=["fileserver", "internal"],
    )
    host_db = Host(
        host_id="host-db-prod-01",
        hostname="db-primary-01",
        ip_address="10.0.3.30",
        os_type="linux",
        criticality=0.95,
        services=["postgres", "sshd"],
        tags=["database", "prod"],
    )
    host_cloud = Host(
        host_id="host-cloud-mgmt",
        hostname="cloud-controller",
        ip_address="10.0.3.50",
        os_type="linux",
        criticality=0.85,
        services=["aws-ssm", "k8s-agent"],
        tags=["cloud", "management"],
    )

    twin.add_host(host_web)
    twin.add_host(host_bastion)
    twin.add_host(host_file)
    twin.add_host(host_db)
    twin.add_host(host_cloud)

    # Services
    twin.add_service(Service(service_id="srv-web", name="nginx", port=80, host_id="host-web-01"))
    twin.add_service(Service(service_id="srv-ssh-web", name="sshd", port=22, host_id="host-web-01"))
    twin.add_service(Service(service_id="srv-ssh-bastion", name="sshd", port=22, host_id="host-bastion-01"))
    twin.add_service(Service(service_id="srv-smb", name="smbd", port=445, host_id="host-file-01"))
    twin.add_service(Service(service_id="srv-postgres", name="postgres", port=5432, host_id="host-db-prod-01"))

    # Processes
    twin.add_process(Process(pid=1024, process_name="nginx", host_id="host-web-01", user_id="www-data"))
    twin.add_process(Process(pid=1100, process_name="sshd", host_id="host-bastion-01", user_id="root"))
    twin.add_process(Process(pid=1200, process_name="smbd", host_id="host-file-01", user_id="root"))
    twin.add_process(Process(pid=1300, process_name="postgres", host_id="host-db-prod-01", user_id="postgres"))

    # Files
    twin.add_file(File(path="/var/data/finance.db", host_id="host-file-01", is_encrypted=False, entropy=4.8))
    twin.add_file(File(path="/var/lib/postgresql/data", host_id="host-db-prod-01", is_encrypted=False, entropy=5.1))

    # Assets
    twin.add_asset(Asset(asset_id="asset-fin-db", asset_type="DATABASE", business_criticality=0.85, host_id="host-file-01"))
    twin.add_asset(Asset(asset_id="asset-prod-db", asset_type="DATABASE", business_criticality=0.95, host_id="host-db-prod-01"))

    # Users & Identities
    twin.add_user(User(user_id="usr-root", username="root", role="admin", is_privileged=True))
    twin.add_user(User(user_id="usr-ops-99", username="ops_lead", role="operations", is_privileged=True))
    twin.add_user(User(user_id="usr-analyst", username="analyst_01", role="analyst", is_privileged=False))
    twin.add_identity(Identity(token_id="tok-jwt-ops-99", user_id="usr-ops-99", is_valid=True, permissions=["read", "write", "admin"]))

    # Security Controls
    twin.add_control(SecurityControl(
        control_id="ctrl-fw-dmz",
        control_type="FIREWALL",
        target_entity="net-dmz",
        is_active=True,
        rules={"blocked_ips": []},
    ))

    # Baseline risk state
    twin.risk_state = {
        "host-web-01": 0.15,
        "host-bastion-01": 0.12,
        "host-file-01": 0.10,
        "host-db-prod-01": 0.08,
    }

    return twin
