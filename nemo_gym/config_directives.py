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
"""Native configuration directives without model or service dependencies."""

from copy import deepcopy
from typing import List

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict


INHERIT_FROM_KEY_NAME = "_inherit_from"
COPY_KEY_NAME = "_copy"
DELETE_KEY_KEY_NAME = "_delete_key"
_MISSING_REF = object()


def resolve_config_directives(config: DictConfig, *, missing_path_error: type[Exception] = ValueError) -> None:
    """Apply the parser's native move, copy and deletion directives in place."""
    _ConfigKeyComposer(missing_path_error)._recursively_swap_keys(config)


class _ConfigKeyComposer:
    def __init__(self, missing_path_error: type[Exception]) -> None:
        self._missing_path_error = missing_path_error

    def _recursively_swap_keys(self, dict_config: DictConfig) -> None:
        frozen_dict_config = deepcopy(dict_config)
        with open_dict(dict_config):
            self._recursively_swap_keys_helper(dict_config, dict_config, frozen_dict_config)

    def _recursively_swap_keys_helper(
        self, dict_config: DictConfig, original_dict_config: DictConfig, frozen_dict_config: DictConfig
    ) -> None:
        # items_ex(resolve=False) yields raw values: directive strings like "${inherit_from:...}"
        # come back unresolved (so the swap detection below still matches), and a missing ('???')
        # leaf is returned as-is instead of raising MissingMandatoryValue mid-swap. Any genuinely
        # unset values are reported together by raise_on_missing_values, which runs after this pass.
        for k, v in list(dict_config.items_ex(resolve=False)):
            is_delete_property = isinstance(v, DictConfig) and DELETE_KEY_KEY_NAME in v

            if is_delete_property:
                keys_to_delete = v.pop(DELETE_KEY_KEY_NAME).split(",")
                keys_to_delete = set(map(str.strip, keys_to_delete))

                # Delete first so we don't resolve the deleted keys
                # but only delete keys that are present in case the key-to-delete comes from a downstream inherit or swap
                existing_keys = set(k for k in keys_to_delete if k in v)
                for key in existing_keys:
                    v.pop(key)
                keys_to_delete -= existing_keys

            if isinstance(v, (DictConfig, dict)):
                self._recursively_swap_keys_helper(v, original_dict_config, frozen_dict_config)
            elif isinstance(v, (ListConfig, list)):
                # Iterate without resolving so a missing ('???') element doesn't raise mid-swap (it's
                # a scalar, so it's skipped here and reported later by raise_on_missing_values).
                for i in range(len(v)):
                    if isinstance(v, ListConfig) and OmegaConf.is_missing(v, i):
                        continue
                    inner_v = v[i]
                    if isinstance(inner_v, (DictConfig, dict)):
                        self._recursively_swap_keys_helper(inner_v, original_dict_config, frozen_dict_config)

            # e.g. ${inherit_from:grpo.num_prompts_per_step}
            is_swap_str = isinstance(v, str) and v.startswith("${inherit_from:")
            is_swap_property = isinstance(v, DictConfig) and INHERIT_FROM_KEY_NAME in v
            is_swap = is_swap_str or is_swap_property

            is_copy_str = isinstance(v, str) and v.startswith("${copy:")
            is_copy_property = isinstance(v, DictConfig) and COPY_KEY_NAME in v
            is_copy = is_copy_str or is_copy_property
            if not (is_swap or is_copy):
                continue

            if is_swap_str:
                path_to_swap = v.removeprefix("${inherit_from:").removesuffix("}")
            elif is_swap_property:
                path_to_swap = v.pop(INHERIT_FROM_KEY_NAME)

            if is_copy_str:
                path_to_swap = v.removeprefix("${copy:").removesuffix("}")
            elif is_copy_property:
                path_to_swap = v.pop(COPY_KEY_NAME)

            path_to_swap = path_to_swap.split(".")

            # Pop the swapped value. A '???' leaf or ancestor on the source path comes back as the
            # _MISSING_REF sentinel (not a dict/str), so guard the .pop()/.merge() that would crash on it.
            dict_containing_key_to_swap = self._recursive_index_dict_using_path(
                original_dict_config, path_to_swap[:-1]
            )
            if is_swap and dict_containing_key_to_swap is not _MISSING_REF:
                # Pop with a default since multiple configs may refer to the same path
                # We don't want to pop if it's just a copy
                dict_containing_key_to_swap.pop(path_to_swap[-1], None)

            swapped_value = self._recursive_index_dict_using_path(frozen_dict_config, path_to_swap)

            # If the source (leaf or an ancestor) is unset, the target inherits MISSING; skip the
            # property-merge and _delete_key handling and let raise_on_missing_values report it.
            if dict_containing_key_to_swap is _MISSING_REF or swapped_value is _MISSING_REF:
                dict_config[k] = "???"
                continue

            if is_swap_property or is_copy_property:
                swapped_value = OmegaConf.merge(swapped_value, v)

            dict_config[k] = swapped_value

            # TODO We may want to recurse again after swap since we are not guaranteed to traverse the swapped-from value before hitting this swap.

            if is_delete_property:
                # Enforce that every key-to-delete exists
                for key in keys_to_delete:
                    dict_config[k].pop(key)

    def _recursive_index_dict_using_path(self, dict_config: DictConfig, path: List[str]) -> "DictConfig | object":
        for k in path:
            # Use _get_node so a referenced value that is unset ('???') can be detected without
            # resolving it (indexing a MISSING leaf would raise an opaque error). A genuinely
            # absent key still errors clearly.
            node = dict_config._get_node(k) if isinstance(dict_config, DictConfig) else None
            if node is None:
                raise self._missing_path_error(f"Path specified does not exist in config: {path}")

            # The referenced value (or an ancestor of it) is unset. Return the _MISSING_REF sentinel
            # so the caller makes the swap/copy/inherit target '???' too (instead of calling .pop()/
            # OmegaConf.merge() on a bare string and crashing); raise_on_missing_values then reports
            # it with an actionable message instead of an opaque interpolation error.
            if OmegaConf.is_missing(dict_config, k):
                return _MISSING_REF

            dict_config = dict_config[k]

        return dict_config
