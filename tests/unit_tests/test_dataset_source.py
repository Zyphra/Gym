# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from contextlib import nullcontext

from pydantic import ValidationError
from pytest import mark, raises, warns

from nemo_gym.config_types import (
    BaseRunServerTypeConfig,
    DatasetConfig,
    GitlabDatasetSource,
    HuggingFaceDatasetSource,
)


def _dataset(**extra) -> dict:
    return {"name": "ds", "type": "example", "jsonl_fpath": "data.jsonl", **extra}


class TestDatasetSource:
    def test_source_gitlab_backfills_legacy_identifier(self) -> None:
        cfg = DatasetConfig.model_validate(
            _dataset(
                source={
                    "type": "gitlab",
                    "dataset_name": "my_dataset",
                    "version": "0.0.1",
                    "artifact_fpath": "train.jsonl",
                }
            )
        )

        assert isinstance(cfg.source, GitlabDatasetSource)
        # Existing consumers read the legacy field; it must be back-filled from `source`.
        assert cfg.gitlab_identifier is not None
        assert cfg.gitlab_identifier.dataset_name == "my_dataset"
        assert cfg.gitlab_identifier.version == "0.0.1"
        assert cfg.gitlab_identifier.artifact_fpath == "train.jsonl"
        assert cfg.huggingface_identifier is None

    def test_source_huggingface_backfills_legacy_identifier(self) -> None:
        cfg = DatasetConfig.model_validate(
            _dataset(source={"type": "huggingface", "repo_id": "org/dataset", "artifact_fpath": "train.jsonl"})
        )

        assert isinstance(cfg.source, HuggingFaceDatasetSource)
        assert cfg.huggingface_identifier is not None
        assert cfg.huggingface_identifier.repo_id == "org/dataset"
        assert cfg.huggingface_identifier.artifact_fpath == "train.jsonl"
        assert cfg.gitlab_identifier is None

    def test_legacy_gitlab_identifier_mirrors_into_source_with_warning(self) -> None:
        with warns(DeprecationWarning, match="gitlab_identifier"):
            cfg = DatasetConfig.model_validate(
                _dataset(
                    gitlab_identifier={
                        "dataset_name": "my_dataset",
                        "version": "0.0.1",
                        "artifact_fpath": "train.jsonl",
                    }
                )
            )

        assert isinstance(cfg.source, GitlabDatasetSource)
        assert cfg.source.dataset_name == "my_dataset"
        assert cfg.source.version == "0.0.1"
        assert cfg.source.artifact_fpath == "train.jsonl"
        # Legacy field stays populated so nothing that already reads it breaks.
        assert cfg.gitlab_identifier is not None

    def test_legacy_huggingface_identifier_mirrors_into_source_with_warning(self) -> None:
        with warns(DeprecationWarning, match="huggingface_identifier"):
            cfg = DatasetConfig.model_validate(_dataset(huggingface_identifier={"repo_id": "org/dataset"}))

        assert isinstance(cfg.source, HuggingFaceDatasetSource)
        assert cfg.source.repo_id == "org/dataset"
        assert cfg.source.artifact_fpath is None
        assert cfg.huggingface_identifier is not None

    def test_specifying_source_and_legacy_identifier_is_rejected(self) -> None:
        with raises(ValidationError, match="set only one"):
            DatasetConfig.model_validate(
                _dataset(
                    source={
                        "type": "gitlab",
                        "dataset_name": "my_dataset",
                        "version": "0.0.1",
                        "artifact_fpath": "train.jsonl",
                    },
                    gitlab_identifier={
                        "dataset_name": "my_dataset",
                        "version": "0.0.1",
                        "artifact_fpath": "train.jsonl",
                    },
                )
            )

    def test_both_legacy_identifiers_together_is_allowed(self) -> None:
        # A gitlab-primary / huggingface-fallback pair (backend chosen at download time) must stay
        # valid; the single discriminated `source:` can't represent both, so it is left unset.
        with warns(DeprecationWarning, match="gitlab_identifier"):
            cfg = DatasetConfig.model_validate(
                _dataset(
                    gitlab_identifier={
                        "dataset_name": "my_dataset",
                        "version": "0.0.1",
                        "artifact_fpath": "train.jsonl",
                    },
                    huggingface_identifier={"repo_id": "org/dataset"},
                )
            )

        assert cfg.source is None
        assert cfg.gitlab_identifier is not None
        assert cfg.huggingface_identifier is not None

    def test_no_source_is_allowed(self) -> None:
        cfg = DatasetConfig.model_validate(_dataset())

        assert cfg.source is None
        assert cfg.gitlab_identifier is None
        assert cfg.huggingface_identifier is None

    def test_source_discriminator_selects_backend(self) -> None:
        with raises(ValidationError):
            # Missing repo_id for the huggingface branch.
            DatasetConfig.model_validate(_dataset(source={"type": "huggingface", "artifact_fpath": "train.jsonl"}))

    @mark.parametrize("exclude_unset", [False, True])
    @mark.parametrize(
        "declaration",
        [
            {},
            {
                "source": {
                    "type": "gitlab",
                    "dataset_name": "my_dataset",
                    "version": "0.0.1",
                    "artifact_fpath": "train.jsonl",
                }
            },
            {"source": {"type": "huggingface", "repo_id": "org/dataset"}},
            {
                "gitlab_identifier": {
                    "dataset_name": "my_dataset",
                    "version": "0.0.1",
                    "artifact_fpath": "train.jsonl",
                }
            },
            {"huggingface_identifier": {"repo_id": "org/dataset", "artifact_fpath": "train.jsonl"}},
            {
                "gitlab_identifier": {
                    "dataset_name": "my_dataset",
                    "version": "0.0.1",
                    "artifact_fpath": "train.jsonl",
                },
                "huggingface_identifier": {"repo_id": "org/fallback"},
            },
        ],
    )
    def test_native_json_roundtrip_preserves_consumed_values(self, declaration, exclude_unset) -> None:
        legacy = "gitlab_identifier" in declaration or "huggingface_identifier" in declaration
        with warns(DeprecationWarning) if legacy else nullcontext():
            original = DatasetConfig.model_validate(_dataset(**declaration))
        dumped = original.model_dump_json(exclude_unset=exclude_unset)
        source_fields = json.loads(dumped)
        if original.source is not None:
            assert source_fields["source"] == original.source.model_dump(mode="json", exclude_unset=exclude_unset)
            assert "gitlab_identifier" not in source_fields
            assert "huggingface_identifier" not in source_fields
        if original.huggingface_identifier is not None:
            assert original.model_dump(include={"huggingface_identifier"}) == {
                "huggingface_identifier": original.huggingface_identifier.model_dump()
            }
        with warns(DeprecationWarning) if original.source is None and legacy else nullcontext():
            restored = DatasetConfig.model_validate_json(dumped)
        # Equality compares all stored native fields, including backfilled
        # legacy identifiers still read by dataset download consumers.
        assert restored == original
        assert restored.model_dump_json(exclude_unset=exclude_unset) == dumped

    def test_roundtrip_preserves_missing_and_explicit_null_source(self) -> None:
        missing = DatasetConfig.model_validate(_dataset())
        assert "source" not in missing.model_dump(exclude_unset=True)
        for field in ("source", "gitlab_identifier", "huggingface_identifier"):
            explicit = DatasetConfig.model_validate(_dataset(**{field: None}))
            dumped = explicit.model_dump_json(exclude_unset=True)
            assert json.loads(dumped)[field] is None
            restored = DatasetConfig.model_validate_json(dumped)
            assert restored.model_dump_json(exclude_unset=True) == dumped

    @mark.parametrize("backend", ["gitlab", "huggingface"])
    def test_conflicting_declared_source_stays_refused(self, backend) -> None:
        if backend == "gitlab":
            source = {"type": "gitlab", "dataset_name": "one", "version": "0.0.1", "artifact_fpath": "one.jsonl"}
            legacy = {"gitlab_identifier": {"dataset_name": "two", "version": "0.0.1", "artifact_fpath": "two.jsonl"}}
        else:
            source = {"type": "huggingface", "repo_id": "org/one"}
            legacy = {"huggingface_identifier": {"repo_id": "org/two"}}
        with raises(ValidationError, match="set only one"):
            DatasetConfig.model_validate(_dataset(source=source, **legacy))

    def test_native_server_catalog_roundtrip(self) -> None:
        with warns(DeprecationWarning):
            original = BaseRunServerTypeConfig.model_validate(
                {
                    "entrypoint": "app.py",
                    "datasets": [
                        _dataset(
                            type="train",
                            license="Apache 2.0",
                            huggingface_identifier={"repo_id": "org/dataset", "artifact_fpath": "train.jsonl"},
                        )
                    ],
                }
            )
        restored = BaseRunServerTypeConfig.model_validate_json(original.model_dump_json())
        assert restored == original
