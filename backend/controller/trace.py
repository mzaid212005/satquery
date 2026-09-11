import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    step_id: int
    step_name: str
    tool_or_model: str
    status: str = "success"  # "success", "warning", "failed"
    duration_ms: float = 0.0
    parameters_applied: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    evidence_produced: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    trace_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    query: str
    inferred_task: str
    workflow_type: str  # "single_model", "chained", "fallback"
    entities_extracted: List[str] = Field(default_factory=list)
    inputs_evaluated: List[Dict[str, Any]] = Field(default_factory=list)
    compatibility_status: str = "passed"
    validation_flags: List[str] = Field(default_factory=list)
    steps: List[TraceStep] = Field(default_factory=list)
    final_confidence: float = 0.0
    total_latency_ms: float = 0.0
    status: str = "completed"  # "completed", "rejected", "error"

    def add_step(
        self,
        step_name: str,
        tool_or_model: str,
        duration_ms: float,
        parameters: Dict[str, Any],
        summary: str,
        evidence: Optional[Dict[str, Any]] = None,
        status: str = "success",
    ) -> None:
        step_id = len(self.steps) + 1
        self.steps.append(
            TraceStep(
                step_id=step_id,
                step_name=step_name,
                tool_or_model=tool_or_model,
                status=status,
                duration_ms=round(duration_ms, 2),
                parameters_applied=parameters,
                summary=summary,
                evidence_produced=evidence or {},
            )
        )

    def to_audit_dict(self) -> Dict[str, Any]:
        """Returns the auditable dictionary without internal reasoning tokens."""
        return self.model_dump()


class TraceRecorder:
    """Context manager and utility to record clean observable execution traces."""

    def __init__(self, trace_id: str, query: str, inferred_task: str, workflow_type: str = "single_model"):
        self.trace = ExecutionTrace(
            trace_id=trace_id,
            query=query,
            inferred_task=inferred_task,
            workflow_type=workflow_type,
        )
        self._start_time = time.perf_counter()

    def record_step(
        self,
        step_name: str,
        tool_or_model: str,
        duration_ms: float,
        parameters: Dict[str, Any],
        summary: str,
        evidence: Optional[Dict[str, Any]] = None,
        status: str = "success",
    ) -> None:
        self.trace.add_step(
            step_name=step_name,
            tool_or_model=tool_or_model,
            duration_ms=duration_ms,
            parameters=parameters,
            summary=summary,
            evidence=evidence,
            status=status,
        )

    def finalize(self, final_confidence: float, status: str = "completed") -> ExecutionTrace:
        total_time = (time.perf_counter() - self._start_time) * 1000.0
        self.trace.total_latency_ms = round(total_time, 2)
        self.trace.final_confidence = round(float(final_confidence), 3)
        self.trace.status = status
        return self.trace
