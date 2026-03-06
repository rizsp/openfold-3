# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import textwrap

from build.lib.openfold3.entry_points.experiment_runner import TrainingExperimentRunner
from openfold3.core.utils.checkpoint_loading_utils import load_checkpoint
from openfold3.entry_points.validator import TrainingExperimentConfig
import pytest  # noqa: F401  - used for pytest tmp fixture
import torch

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.core.utils.precision_utils import OF3DeepSpeedPrecision
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
from openfold3.tests import compare_utils
from openfold3.tests.config import consts
from openfold3.tests.data_utils import random_of3_features


class TestOF3ModelCheckpointing:
    def run_model(
        self,
        batch_size,
        n_token,
        n_msa,
        n_templ,
        tmp_path,
    ):
        mini_model_settings = textwrap.dedent("""
            model_config:
                architecture:
                    pairformer:
                        no_blocks: 4
                    diffusion_module:
                        diffusion_transformer:
                        no_blocks: 4
                    loss_module:
                        diffusion:
                        chunk_size: 16
                settings:
                    blocks_per_ckpt: 1
                    ckpt_intermediate_steps: true
                    train:
                        use_deepspeed_evo_attention: false
                    eval:
                        use_deepspeed_evo_attention: false
        """)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        expt_config = TrainingExperimentConfig.model_validate(mini_model_settings)
        expt_runner = TrainingExperimentRunner(expt_config)
        model = expt_runner.lightning_module.model

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=n_msa,
            n_templ=n_templ,
        )

        def to_device(t):
            return t.to(device=torch.device(device))

        batch = tensor_tree_map(to_device, batch)

        batch, outputs = model(batch=batch)

        # abbreviated pytorch lightning checkpoint
        fake_pytorch_lightning_checkpoint = {
            "epoch": 0,
            "global_step": 0,
            "state_dict": expt_runner.lightning_module.state_dict(),
            "ema": expt_runner.lightning_module.ema.state_dict(),  # required by on_load_checkpoint
        }
        torch.save(fake_pytorch_lightning_checkpoint, tmp_path)

    reloaded_model = load_checkpoint(tmp_path)

    ema_params = reloaded_model["ema"]["params"]
    expected_version_number = "1.0.0"
    actual_version = ema_params["version_tensor"].long().tolist()
    actual_version_number = (
        f"{actual_version[0]}.{actual_version[1]}.{actual_version[2]}"
    )

    assert actual_version_number == expected_version_number
