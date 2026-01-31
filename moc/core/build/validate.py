from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from moc.core.models.network import Network


@dataclass(frozen=True)
class ValidationIssue:
    level: str              # "error" | "warning"
    message: str
    hint: Optional[str] = None


class NetworkValidationError(ValueError):
    """Raised when validation finds one or more errors."""
    def __init__(self, issues: List[ValidationIssue]):
        self.issues = issues
        lines = ["Network validation failed with errors:"]
        for it in issues:
            if it.level == "error":
                lines.append(f"- {it.message}" + (f" | hint: {it.hint}" if it.hint else ""))
        super().__init__("\n".join(lines))


def validate_network(network: Network, *, require_event: bool = False) -> List[ValidationIssue]:
    """
    Validate a Network for basic consistency.
    Returns a list of issues (errors and warnings). If errors exist, caller may raise.

    This is intentionally minimal (Step 2.1). Later steps may add deeper checks.
    """
    issues: List[ValidationIssue] = []

    # --- Nodes ---
    if not network.nodes:
        issues.append(ValidationIssue(
            level="error",
            message="Network has zero nodes.",
            hint="Add at least 2 nodes in Excel sheet 'nodo'."
        ))

    # basic node properties
    for uid, n in network.nodes.items():
        if not uid or not isinstance(uid, str):
            issues.append(ValidationIssue("error", f"Node has invalid uid: {uid!r}"))
        if n.name is None or str(n.name).strip() == "":
            issues.append(ValidationIssue("warning", f"Node(uid={uid}) has empty name.", "Set 'nombre' in Excel."))
        # z should be a number (dataclass already enforces, but protect against NaN)
        try:
            z = float(n.z)
            if z != z:  # NaN check
                issues.append(ValidationIssue("error", f"Node(uid={uid}) elevation z is NaN."))
        except Exception:
            issues.append(ValidationIssue("error", f"Node(uid={uid}) elevation z is not numeric: {n.z!r}."))

        if n.bc_type is None:
            issues.append(ValidationIssue("error", f"Node(uid={uid}) bc_type is None.", "Use 'none' instead."))
        if n.bc_type != "none" and n.bc_value is None:
            issues.append(ValidationIssue(
                "error",
                f"Node(uid={uid}) has bc_type='{n.bc_type}' but bc_value is missing.",
                "Fill 'valor_cond_borde' for this node in Excel."
            ))

    # --- Pipes ---
    if not network.pipes:
        issues.append(ValidationIssue(
            level="error",
            message="Network has zero pipes.",
            hint="Add at least 1 pipe in Excel sheet 'tuberia'."
        ))

    for uid, p in network.pipes.items():
        if p.node_from not in network.nodes:
            issues.append(ValidationIssue(
                "error",
                f"Pipe(uid={uid}, name={p.name}) references unknown node_from uid={p.node_from!r}.",
                "Ensure 'nodo_inicio' exists in sheet 'nodo' and adapter mapping is correct."
            ))
        if p.node_to not in network.nodes:
            issues.append(ValidationIssue(
                "error",
                f"Pipe(uid={uid}, name={p.name}) references unknown node_to uid={p.node_to!r}.",
                "Ensure 'nodo_fin' exists in sheet 'nodo' and adapter mapping is correct."
            ))

        # geometry checks
        try:
            L = float(p.length)
            if L <= 0:
                issues.append(ValidationIssue("error", f"Pipe(uid={uid}, name={p.name}) length <= 0: {L}"))
        except Exception:
            issues.append(ValidationIssue("error", f"Pipe(uid={uid}, name={p.name}) length is not numeric: {p.length!r}"))

        try:
            D = float(p.diameter)
            if D <= 0:
                issues.append(ValidationIssue("error", f"Pipe(uid={uid}, name={p.name}) diameter <= 0: {D}"))
        except Exception:
            issues.append(ValidationIssue("error", f"Pipe(uid={uid}, name={p.name}) diameter is not numeric: {p.diameter!r}"))

        # optional fields sanity
        if p.wave_speed is not None:
            try:
                a = float(p.wave_speed)
                if a <= 0:
                    issues.append(ValidationIssue("error", f"Pipe(uid={uid}, name={p.name}) wave_speed <= 0: {a}"))
            except Exception:
                issues.append(ValidationIssue("error", f"Pipe(uid={uid}, name={p.name}) wave_speed not numeric: {p.wave_speed!r}"))

        if p.friction is not None:
            try:
                f = float(p.friction)
                if f <= 0 or f > 1:
                    issues.append(ValidationIssue(
                        "warning",
                        f"Pipe(uid={uid}, name={p.name}) friction seems unusual: f={f}",
                        "Darcy f is typically ~0.01–0.05 (varies with material/roughness)."
                    ))
            except Exception:
                issues.append(ValidationIssue("error", f"Pipe(uid={uid}, name={p.name}) friction not numeric: {p.friction!r}"))

        if p.dx_target is not None:
            try:
                dx = float(p.dx_target)
                if dx <= 0:
                    issues.append(ValidationIssue("error", f"Pipe(uid={uid}, name={p.name}) dx_target <= 0: {dx}"))
            except Exception:
                issues.append(ValidationIssue("error", f"Pipe(uid={uid}, name={p.name}) dx_target not numeric: {p.dx_target!r}"))

    # --- Connectivity quick check (series vs general) ---
    # We won't enforce a topology yet, but we can warn if many disconnected components exist.
    if network.nodes and network.pipes:
        # Build undirected adjacency
        adj = {nid: set() for nid in network.nodes.keys()}
        for p in network.pipes.values():
            if p.node_from in adj and p.node_to in adj:
                adj[p.node_from].add(p.node_to)
                adj[p.node_to].add(p.node_from)

        # Count components (simple DFS)
        visited = set()
        comps = 0
        for nid in adj:
            if nid in visited:
                continue
            comps += 1
            stack = [nid]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                stack.extend(list(adj[cur] - visited))

        if comps > 1:
            issues.append(ValidationIssue(
                "warning",
                f"Network appears to have {comps} disconnected components.",
                "If this is unintended, check node ids and pipe connections in Excel."
            ))

    # --- Events ---
    if require_event and not network.events:
        issues.append(ValidationIssue(
            "error",
            "Network has zero events but require_event=True.",
            "Add at least 1 row in sheet 'eventos'."
        ))

    for e in network.events:
        if e.target_uid not in network.nodes and e.target_uid not in network.pipes:
            issues.append(ValidationIssue(
                "error",
                f"Event(uid={e.uid}, type={e.event_type}) target_uid does not match any node or pipe.",
                "Check 'objetivo_tipo' and 'objetivo_id' in sheet 'eventos'."
            ))
        try:
            t0 = float(e.t_start)
            t1 = float(e.t_end)
            if t1 < t0:
                issues.append(ValidationIssue(
                    "error",
                    f"Event(uid={e.uid}) has t_end < t_start ({t1} < {t0})."
                ))
        except Exception:
            issues.append(ValidationIssue("error", f"Event(uid={e.uid}) has non-numeric times."))

    return issues


def raise_on_errors(issues: List[ValidationIssue]) -> None:
    errors = [i for i in issues if i.level == "error"]
    if errors:
        raise NetworkValidationError(errors)
