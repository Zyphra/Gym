# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One bounded server-origin record, independent of HTTP outcome and retry policy."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from nemo_gym.http_connection_diagnostics import request_metadata
from nemo_gym.rollout_correlation import current_rollout_id


FAILURE_PROVENANCE_HEADER = "x-nemo-gym-failure-provenance"
MAX_FAILURE_PROVENANCE_HEADER = 1024
_CLASS_LABEL = re.compile(r"[A-Za-z_][A-Za-z0-9_.:()-]{0,127}\Z")
FailureStage = Literal["cleanup", "transport", "verification", "generation", "setup", "storage", "scoring", "unknown"]


class HTTPFailureProvenance(BaseModel):
    """Safe identifiers only; no exception messages, bodies, queries or headers.

    The class grammar and stage vocabulary match the native grader-component
    carrier. Operation is the existing route classifier, never a stage inference.
    A relay keeps the original record rather than appending an unbounded chain.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, revalidate_instances="always")

    schema_version: Literal["nemo_gym.http_failure.v1"] = "nemo_gym.http_failure.v1"
    exception_class: str
    failure_stage: FailureStage = "unknown"
    operation: Literal["other", "enroll", "release", "episode", "finalize", "progress", "snapshot", "run", "model"]
    rollout_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

    @field_validator("exception_class")
    @classmethod
    def safe_class_label(cls, value: str) -> str:
        if not _CLASS_LABEL.fullmatch(value) or value.lower().startswith(("http:", "https:", "file:", "ftp:")):
            raise ValueError("exception class must be a bounded code identifier")
        return value

    @classmethod
    def from_header(cls, value: str | None) -> "HTTPFailureProvenance | None":
        if not isinstance(value, str) or not value.isascii() or len(value) > MAX_FAILURE_PROVENANCE_HEADER:
            return None
        try:
            record = cls.model_validate_json(value)
            return record if "schema_version" in record.model_fields_set else None
        except (ValidationError, ValueError, RecursionError):
            return None


def failure_provenance_headers(error: BaseException, *, method: str, path: str, relay: bool = False) -> dict[str, str]:
    """Emit only owned, validated provenance; diagnostic failure leaves HTTP intact.

    A transport error without an origin record remains unknown: it must not turn
    into an invented initiating server error. The request path is used only by
    the existing safe route/rollout classifier and is never placed on the wire.
    """
    try:
        if relay:
            origin = getattr(error, "nemo_gym_failure_provenance", None)
            if not isinstance(origin, HTTPFailureProvenance):
                return {}
            origin = HTTPFailureProvenance.model_validate(origin)
        else:
            metadata = request_metadata(method, path, current_rollout_id())
            name = type(error).__name__
            try:
                name = HTTPFailureProvenance.safe_class_label(name)
            except ValueError:
                name = "unknown"
            stage = getattr(error, "failure_stage", "unknown")
            # Ask the model to validate the stage without guessing from the route.
            try:
                origin = HTTPFailureProvenance(
                    exception_class=name,
                    failure_stage=stage,
                    operation=metadata["operation"],
                    rollout_id=metadata["rollout_id"],
                )
            except ValidationError:
                origin = HTTPFailureProvenance(
                    exception_class=name,
                    operation=metadata["operation"],
                    rollout_id=metadata["rollout_id"],
                )
        value = origin.model_dump_json()
        return {FAILURE_PROVENANCE_HEADER: value} if len(value) <= MAX_FAILURE_PROVENANCE_HEADER else {}
    except Exception:
        return {}
