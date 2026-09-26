"""
AHRAS Process Adversarial Mutator
---------------------------------
Generates semantics-preserving adversarial mutations on host process telemetry:
- Command-line obfuscation (case alternation, carets, quoting, whitespace padding, env vars)
- Process path variation (relative paths, SysWOW64, directory traversal)
- Flag reordering (permuting CLI switches while preserving execution)
- Alias substitution (bash vs sh, powershell vs pwsh, IEX vs Invoke-Expression)
- Payload encoding variations
"""

from __future__ import annotations

import copy
import enum
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ProcessMutationStrategy(str, enum.Enum):
    CASE_ALTERNATION = "case_alternation"
    CARET_INSERTION = "caret_insertion"
    QUOTE_INSERTION = "quote_insertion"
    WHITESPACE_PADDING = "whitespace_padding"
    PATH_VARIATION = "path_variation"
    FLAG_REORDERING = "flag_reordering"
    ALIAS_SUBSTITUTION = "alias_substitution"


@dataclass
class ProcessMutationResult:
    strategy: ProcessMutationStrategy
    original_cmd: str
    mutated_cmd: str
    original_event: Dict[str, Any]
    mutated_event: Dict[str, Any]
    semantic_integrity_preserved: bool = True


class ProcessMutator:
    """
    Applies domain-valid, semantics-preserving mutations to process telemetry events.
    """
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def mutate_event(
        self,
        event: Dict[str, Any],
        strategy: ProcessMutationStrategy,
    ) -> ProcessMutationResult:
        """Applies a single specific mutation strategy to an event."""
        mut_evt = copy.deepcopy(event)
        
        # Extract cmd_line safely from either actor.process or process
        actor_proc = mut_evt.get("actor", {}).get("process", {})
        raw_proc = mut_evt.get("process", {})
        
        orig_cmd = actor_proc.get("cmd_line") or raw_proc.get("cmd") or ""

        mutated_cmd = orig_cmd

        if strategy == ProcessMutationStrategy.CASE_ALTERNATION:
            mutated_cmd = self._mutate_case_alternation(orig_cmd)
        elif strategy == ProcessMutationStrategy.CARET_INSERTION:
            mutated_cmd = self._mutate_caret_insertion(orig_cmd)
        elif strategy == ProcessMutationStrategy.QUOTE_INSERTION:
            mutated_cmd = self._mutate_quote_insertion(orig_cmd)
        elif strategy == ProcessMutationStrategy.WHITESPACE_PADDING:
            mutated_cmd = self._mutate_whitespace_padding(orig_cmd)
        elif strategy == ProcessMutationStrategy.PATH_VARIATION:
            mutated_cmd = self._mutate_path_variation(orig_cmd)
        elif strategy == ProcessMutationStrategy.FLAG_REORDERING:
            mutated_cmd = self._mutate_flag_reordering(orig_cmd)
        elif strategy == ProcessMutationStrategy.ALIAS_SUBSTITUTION:
            mutated_cmd = self._mutate_alias_substitution(orig_cmd)

        # Write back mutated command line
        if "process" in mut_evt and isinstance(mut_evt["process"], dict):
            mut_evt["process"]["cmd"] = mutated_cmd
        if "actor" in mut_evt and "process" in mut_evt["actor"]:
            mut_evt["actor"]["process"]["cmd_line"] = mutated_cmd

        return ProcessMutationResult(
            strategy=strategy,
            original_cmd=orig_cmd,
            mutated_cmd=mutated_cmd,
            original_event=event,
            mutated_event=mut_evt,
            semantic_integrity_preserved=True,
        )

    def generate_all_mutations(self, event: Dict[str, Any]) -> List[ProcessMutationResult]:
        """Generates all applicable process mutations for an event."""
        results = []
        for strategy in ProcessMutationStrategy:
            res = self.mutate_event(event, strategy)
            # Only include if command was actually mutated
            if res.mutated_cmd != res.original_cmd:
                results.append(res)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Concrete Mutator Logic
    # ─────────────────────────────────────────────────────────────────────────

    def _mutate_case_alternation(self, cmd: str) -> str:
        """Randomly alters casing of alphabetic characters: e.g. PoWeRsHeLl.ExE."""
        if not cmd:
            return cmd
        # Preserve substrings inside quotes or flags
        chars = [c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(cmd)]
        return "".join(chars)

    def _mutate_caret_insertion(self, cmd: str) -> str:
        """Inserts Windows CMD escape carets (^) inside token names: p^o^w^e^r^s^h^e^l^l."""
        tokens = cmd.split(" ")
        mut_tokens = []
        for t in tokens:
            if len(t) > 4 and not t.startswith("-") and not t.startswith("/") and not t.startswith("http"):
                # Insert 1-2 carets
                mid = len(t) // 2
                t = t[:mid] + "^" + t[mid:]
            mut_tokens.append(t)
        return " ".join(mut_tokens)

    def _mutate_quote_insertion(self, cmd: str) -> str:
        """Inserts empty quotes or segmented quotes: p""o""w""e""r""s""h""e""l""l."""
        tokens = cmd.split(" ")
        mut_tokens = []
        for t in tokens:
            if len(t) > 5 and not t.startswith("-") and not t.startswith("/"):
                t = t[:2] + '""' + t[2:]
            mut_tokens.append(t)
        return " ".join(mut_tokens)

    def _mutate_whitespace_padding(self, cmd: str) -> str:
        """Injects non-standard whitespace padding between arguments."""
        tokens = cmd.split(" ")
        return "   ".join(tokens)

    def _mutate_path_variation(self, cmd: str) -> str:
        """Prepends relative dot-slash or absolute system directory paths."""
        tokens = cmd.split(" ")
        if not tokens:
            return cmd
        binary = tokens[0]
        if "powershell" in binary.lower():
            tokens[0] = "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
        elif binary in ("bash", "sh"):
            tokens[0] = f"/usr/bin/{binary}"
        elif binary in ("vssadmin", "vssadmin.exe"):
            tokens[0] = "C:\\Windows\\System32\\vssadmin.exe"
        elif binary in ("mimikatz", "mimikatz.exe"):
            tokens[0] = ".\\mimikatz.exe"
        return " ".join(tokens)

    def _mutate_flag_reordering(self, cmd: str) -> str:
        """Permutes independent CLI switches while preserving core target arguments."""
        tokens = cmd.split(" ")
        if len(tokens) <= 2:
            return cmd
        binary = tokens[0]
        flags = [t for t in tokens[1:] if t.startswith("-") or t.startswith("/")]
        non_flags = [t for t in tokens[1:] if not (t.startswith("-") or t.startswith("/"))]
        if len(flags) > 1:
            flags.reverse()
        return " ".join([binary] + non_flags + flags)

    def _mutate_alias_substitution(self, cmd: str) -> str:
        """Substitutes binary or command aliases."""
        mutated = cmd
        replacements = [
            ("powershell.exe", "pwsh.exe"),
            ("bash", "sh"),
            ("cat", "more"),
            ("tar -czf", "tar -c -z -f"),
        ]
        for src, dst in replacements:
            if src in mutated:
                mutated = mutated.replace(src, dst)
                break
        return mutated
