# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray
from ..extras.packages import is_safetensors_available


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key).clone() for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"),
                output_dir=output_dir,
                safe_serialization=getattr(args, "save_safetensors", True),
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=getattr(args, "save_safetensors", True))
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=getattr(args, "save_safetensors", True))
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=getattr(args, "save_safetensors", True),
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if "trackio" in args.report_to:
            import trackio

            trackio.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )


class PassRateEarlyStoppingCallback(TrainerCallback):
    r"""Early stopping based on pass rate. Runs model inference on an eval set,
    calls test_sft_python.py, and stops training if pass rate stops improving.

    Gracefully degrades: if eval file or test script is missing, simply logs a
    warning and does nothing (never crashes training).
    """

    _RE_ANSWER = re.compile(r"<answer>\s*```(?:python)?\s*\n(.*?)```\s*</answer>", re.DOTALL)
    _RE_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    _RE_PASS_RATE = re.compile(r"Pass Rate:\s*([\d.]+)%")

    def __init__(
        self,
        eval_file: str,
        test_sft_python_path: Optional[str] = None,
        eval_interval: int = 3,
        patience: int = 2,
        num_samples_per_problem: int = 10,
        max_samples: int = 50,
        vllm_gpu: Optional[str] = None,
        base_model_path: Optional[str] = None,
    ) -> None:
        self.eval_file = eval_file
        self.test_sft_python_path = test_sft_python_path
        self.eval_interval = eval_interval
        self.patience = patience
        self.num_samples_per_problem = max(1, int(num_samples_per_problem))
        self.max_samples = max_samples
        self.eval_count = 0
        self.best_pass_rate = -1.0
        self.no_improve_count = 0
        self._eval_samples: Optional[list] = None
        self._trainer: Optional[Any] = None
        self._disabled = False
        self._resolved_test_script: Optional[str] = None
        self._prompt_cache: Optional[list] = None
        self.vllm_gpu = vllm_gpu
        self.base_model_path = base_model_path

    def set_trainer(self, trainer: Any) -> None:
        self._trainer = trainer

    def _is_main_process(self, state: Any) -> bool:
        val = getattr(state, "is_world_process_zero", True)
        return bool(val)

    def _load_eval_samples(self) -> list:
        if self._eval_samples is not None:
            return self._eval_samples

        if not os.path.isfile(self.eval_file):
            return []

        samples = []
        with open(self.eval_file) as f:
            for line in f:
                if len(samples) >= self.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._eval_samples = samples
        return samples

    @classmethod
    def _extract_python(cls, content: str) -> str:
        m = cls._RE_ANSWER.search(content)
        if m:
            return m.group(1).strip()
        m = cls._RE_CODE_BLOCK.search(content)
        if m:
            return m.group(1).strip()
        return content.strip()

    def _resolve_test_script(self) -> Optional[str]:
        if self._resolved_test_script is not None:
            return self._resolved_test_script
        if self.test_sft_python_path and os.path.isfile(self.test_sft_python_path):
            self._resolved_test_script = self.test_sft_python_path
            return self._resolved_test_script
        candidates = [
            "/app/scripts/eval_pass_rate_verilogeval2.py",
            "/app/scripts/test_sft_python.py",
            "/ssd2/yichen/codev-r1/test_sft_python.py",
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                self._resolved_test_script = p
                return p
        return None

    def _prepare_prompts(self, tokenizer: Any) -> list:
        """Prepare prompts once and cache them across evaluations."""
        if self._prompt_cache is not None:
            return self._prompt_cache

        samples = self._load_eval_samples()
        prompt_data = []
        for sample in samples:
            question = sample.get("question", [])
            task_id = sample.get("task_id", "unknown")
            messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in question]
            try:
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                prompt = "\n".join(m.get("content", "") for m in question)
            prompt_data.append((task_id, prompt))

        self._prompt_cache = prompt_data
        return prompt_data

    def _distributed_inference(self, use_base_only: bool = False) -> Optional[list]:
        """Run inference across ALL DDP ranks, gather results to rank 0.

        use_base_only: if True, temporarily disable LoRA so step-0 eval uses pure base (matches standalone base pass@1).
        """
        trainer = self._trainer
        if trainer is None:
            return None

        tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        if tokenizer is None:
            return None

        model = getattr(trainer, "model", None)
        if model is None:
            return None

        prompt_data = self._prepare_prompts(tokenizer)
        if not prompt_data:
            return None

        # Expand: each (task_id, prompt) repeated num_samples_per_problem times for pass@1
        n = self.num_samples_per_problem
        expanded = [(tid, p) for (tid, p) in prompt_data for _ in range(n)]

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        # Each rank takes its shard (interleaved for balance)
        my_prompts = expanded[rank::world_size]
        # Sort by prompt length within shard to minimize padding waste
        my_prompts = sorted(my_prompts, key=lambda x: len(x[1]))

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        use_sample = n > 1
        gen_kwargs = {
            "max_new_tokens": 16384,
            "do_sample": use_sample,
            "temperature": 0.7 if use_sample else 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "pad_token_id": pad_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }

        unwrapped = getattr(trainer, "model_wrapped", model)
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        # Step 0: use pure base (disable LoRA) so pass@1 matches standalone base model
        adapter_disabled = False
        if use_base_only and hasattr(unwrapped, "disable_adapter"):
            unwrapped.disable_adapter()
            adapter_disabled = True
            if rank == 0:
                logger.info("PassRate step-0: using base model only (LoRA disabled for baseline).")

        was_training = unwrapped.training
        old_use_cache = getattr(unwrapped.config, "use_cache", True)
        unwrapped.config.use_cache = True
        unwrapped.eval()
        torch.cuda.empty_cache()

        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        # Smaller batch during step-0 to reduce peak VRAM and avoid OOM when training starts
        BATCH_SIZE = 4 if use_base_only else 8
        my_results: list[dict] = []
        t_infer_start = time.monotonic()
        try:
            with torch.no_grad():
                for batch_idx, batch_start in enumerate(range(0, len(my_prompts), BATCH_SIZE)):
                    batch = my_prompts[batch_start : batch_start + BATCH_SIZE]
                    prompts = [p for _, p in batch]

                    encoded = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=4096,
                    )
                    encoded = {k: v.to(unwrapped.device) for k, v in encoded.items()}
                    input_len = encoded["input_ids"].shape[1]

                    t_batch = time.monotonic()
                    try:
                        out = unwrapped.generate(**encoded, **gen_kwargs)
                        gen_len = out.shape[1] - input_len
                        for i, (task_id, _) in enumerate(batch):
                            text = tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
                            my_results.append({"task_id": task_id, "completion": text})
                        if rank == 0:
                            logger.info(
                                f"PassRate inference batch {batch_idx}: "
                                f"{len(batch)} samples, input_len={input_len}, gen_len={gen_len}, "
                                f"{time.monotonic() - t_batch:.1f}s"
                            )
                    except Exception as e:
                        if rank == 0:
                            logger.warning(f"PassRateEarlyStopping: batch generate failed: {e}")
                        for task_id, _ in batch:
                            my_results.append({"task_id": task_id, "completion": ""})
        finally:
            if adapter_disabled and hasattr(unwrapped, "enable_adapter"):
                unwrapped.enable_adapter()
            tokenizer.padding_side = orig_padding_side
            unwrapped.config.use_cache = old_use_cache
            if was_training:
                unwrapped.train()
            torch.cuda.empty_cache()

        # Gather results from all ranks to rank 0
        if world_size > 1:
            gathered: list[Optional[list]] = [None] * world_size
            torch.distributed.all_gather_object(gathered, my_results)
            if rank == 0:
                all_results: list[dict] = []
                for shard in gathered:
                    if shard:
                        all_results.extend(shard)
                t_total = time.monotonic() - t_infer_start
                logger.info(
                    f"PassRate distributed inference: {len(all_results)} samples "
                    f"across {world_size} GPUs in {t_total:.1f}s"
                )
                return all_results
            return None
        else:
            t_total = time.monotonic() - t_infer_start
            logger.info(f"PassRate inference total: {len(my_results)} samples in {t_total:.1f}s")
            return my_results

    def _vllm_inference(self, use_base_only: bool = False) -> Optional[list]:
        """Run inference using vLLM on a separate GPU. Only rank 0 does the work."""
        import shutil
        import subprocess
        import tempfile

        trainer = self._trainer
        if trainer is None:
            return None

        # Save current adapter to temp dir (rank 0 only)
        adapter_dir = None
        if not use_base_only:
            model = getattr(trainer, "model", None)
            if model is None:
                return None
            unwrapped = model
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            adapter_dir = tempfile.mkdtemp(prefix="pass_rate_adapter_")
            unwrapped.save_pretrained(adapter_dir)
            logger.info(f"PassRate vLLM: saved adapter to {adapter_dir}")

        # Get base model path
        base_model = self.base_model_path
        if not base_model:
            model = getattr(trainer, "model", None)
            if model is not None:
                unwrapped = model
                if hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module
                base_model = getattr(unwrapped.config, "_name_or_path", None)
        if not base_model:
            logger.warning("PassRate vLLM: cannot determine base model path")
            return None

        # Find the vLLM inference script
        script_candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "scripts", "vllm_pass_rate_infer.py"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "vllm_pass_rate_infer.py"),
            "/app/scripts/vllm_pass_rate_infer.py",
        ]
        vllm_script = None
        for p in script_candidates:
            if os.path.isfile(p):
                vllm_script = p
                break
        if vllm_script is None:
            logger.warning("PassRate vLLM: vllm_pass_rate_infer.py not found")
            return None

        output_dir = tempfile.mkdtemp(prefix="pass_rate_vllm_out_")
        output_file = os.path.join(output_dir, "results.jsonl")

        env = os.environ.copy()
        # Remove ALL distributed-training env vars so vLLM starts cleanly on its own GPU
        _dist_keys = {
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
            "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "GROUP_WORLD_SIZE",
            "ROLE_RANK", "ROLE_WORLD_SIZE", "OMP_NUM_THREADS",
        }
        for key in list(env.keys()):
            if key in _dist_keys or key.startswith(("TORCHELASTIC_", "TORCH_NPROC")):
                env.pop(key)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["CUDA_VISIBLE_DEVICES"] = self.vllm_gpu
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # run engine in-process to avoid TCPStore issues

        cmd = [
            sys.executable, vllm_script,
            "--base-model", base_model,
            "--eval-file", os.path.abspath(self.eval_file),
            "--output-file", output_file,
            "--max-samples", str(self.max_samples),
            "--num-samples-per-problem", str(self.num_samples_per_problem),
            "--max-new-tokens", "2048",
            "--no-think",
        ]
        if adapter_dir:
            cmd.extend(["--adapter-path", adapter_dir])

        t_start = time.monotonic()
        logger.info(f"PassRate vLLM: starting on GPU {self.vllm_gpu} ({'base only' if use_base_only else 'with adapter'})")

        results = None
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0:
                logger.warning(f"PassRate vLLM failed (exit {proc.returncode}):\nstderr: {proc.stderr[-3000:]}")
            else:
                if proc.stderr:
                    logger.info(f"PassRate vLLM: {proc.stderr.strip()}")
                results = []
                if os.path.isfile(output_file):
                    with open(output_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                results.append(json.loads(line))
        except subprocess.TimeoutExpired:
            logger.warning("PassRate vLLM: timed out (3600s)")
        except Exception as e:
            logger.warning(f"PassRate vLLM: error: {e}")
        finally:
            if adapter_dir:
                shutil.rmtree(adapter_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

        t_total = time.monotonic() - t_start
        logger.info(f"PassRate vLLM: {len(results) if results else 0} results in {t_total:.1f}s")
        return results if results else None

    def _run_test_script(self, results: list) -> Optional[float]:
        """Run eval script on gathered results (rank 0 only)."""
        import subprocess
        import tempfile

        test_sft_py = self._resolve_test_script()
        if test_sft_py is None:
            logger.warning_rank0("PassRateEarlyStopping: eval script not found, disabling.")
            self._disabled = True
            return None

        work_dir = tempfile.mkdtemp(prefix="pass_rate_eval_")
        temp_jsonl = os.path.join(work_dir, "eval_batch.jsonl")
        with open(temp_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        eval_file_abs = os.path.abspath(self.eval_file)
        trace_cache = os.path.join(os.path.dirname(eval_file_abs), "verilogeval_v2_traces")
        num_parallel = min(32, len(results))
        cmd = [
            sys.executable,
            test_sft_py,
            temp_jsonl,
            "--trace-cache-dir",
            trace_cache,
            "--parallel",
            str(num_parallel),
        ]
        pass_rate = None
        t_test_start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, cwd=os.path.dirname(test_sft_py)
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            m = self._RE_PASS_RATE.search(stdout)
            if m:
                pass_rate = float(m.group(1))
            else:
                logger.warning(
                    f"PassRateEarlyStopping: could not parse pass rate (rc={proc.returncode}):\n"
                    f"stdout: {stdout[:300]}\nstderr: {stderr[:300]}"
                )
        except subprocess.TimeoutExpired:
            logger.warning("PassRateEarlyStopping: test script timed out (600s).")
        except Exception as e:
            logger.warning(f"PassRateEarlyStopping: test script failed: {e}")
        finally:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)

        logger.info(f"PassRate test (cached traces): {time.monotonic() - t_test_start:.1f}s")
        return pass_rate

    def _broadcast_should_stop(self, should_stop: bool) -> bool:
        """Broadcast the stop decision from rank 0 to all ranks in DDP/FSDP."""
        if not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1):
            return should_stop

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        tensor = torch.tensor([int(should_stop)], dtype=torch.long, device=device)
        torch.distributed.broadcast(tensor, src=0)
        return tensor.item() == 1

    def _do_inference(self, use_base_only: bool = False) -> Optional[list]:
        """Route inference to vLLM (rank 0 only) or HF distributed generate."""
        if self.vllm_gpu:
            return self._vllm_inference(use_base_only=use_base_only)
        return self._distributed_inference(use_base_only=use_base_only)

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Run pass rate eval at step 0 (before any training) to establish baseline."""
        if self._disabled or self._trainer is None:
            return
        # Skip step-0 baseline when resuming from checkpoint (already past step 0)
        if getattr(state, "global_step", 0) > 0:
            logger.info_rank0("PassRate: skipping step-0 baseline (resuming from checkpoint)")
            return
        is_main = self._is_main_process(state)
        is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        all_results = None
        try:
            if self.vllm_gpu:
                # vLLM: only rank 0 does inference, others wait
                if is_main:
                    all_results = self._vllm_inference(use_base_only=True)
                if is_distributed:
                    torch.distributed.barrier()
            else:
                # HF generate: all ranks participate
                all_results = self._distributed_inference(use_base_only=True)
        except Exception as e:
            if is_main:
                logger.warning(f"PassRateEarlyStopping: step-0 inference error: {e}")
            return
        if is_main and all_results:
            try:
                pass_rate = self._run_test_script(all_results)
            except Exception as e:
                logger.warning(f"PassRateEarlyStopping: step-0 test error: {e}")
                return
            if pass_rate is not None:
                step0 = getattr(state, "global_step", 0)
                logger.info(
                    f"Pass rate at step 0 (baseline): {pass_rate:.2f}% "
                    f"(pass@1, {self.num_samples_per_problem} samples/problem)"
                )
                if pass_rate > self.best_pass_rate:
                    self.best_pass_rate = pass_rate
                if args.report_to and "wandb" in args.report_to:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({"eval/pass_rate": pass_rate}, step=step0)
                    except Exception:
                        pass
        # Aggressive GPU cleanup so first training step does not OOM (release inference cache)
        if not self.vllm_gpu and torch.cuda.is_initialized():
            if is_distributed:
                torch.distributed.barrier()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self.eval_count += 1
        if self.eval_count % self.eval_interval != 0:
            return

        is_main = self._is_main_process(state)
        is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        should_stop = False

        all_results = None
        if not self._disabled and self._trainer is not None:
            try:
                if self.vllm_gpu:
                    # vLLM: only rank 0 does inference on separate GPU
                    if is_main:
                        all_results = self._vllm_inference()
                    if is_distributed:
                        torch.distributed.barrier()
                else:
                    # HF generate: ALL ranks participate in distributed inference
                    all_results = self._distributed_inference()
            except Exception as e:
                if is_main:
                    logger.warning(f"PassRateEarlyStopping: inference error, disabling: {e}")
                self._disabled = True

        # Only rank 0 runs the test script and decides on early stopping
        if is_main and all_results:
            try:
                pass_rate = self._run_test_script(all_results)
            except Exception as e:
                logger.warning(f"PassRateEarlyStopping: test error: {e}")
                pass_rate = None

            if pass_rate is not None:
                logger.info(
                    f"Pass rate eval #{self.eval_count // self.eval_interval}: "
                    f"{pass_rate:.2f}% (pass@1, {self.num_samples_per_problem} samples/problem)  (best: {self.best_pass_rate:.2f}%)"
                )
                if args.report_to and "wandb" in args.report_to:
                    try:
                        import wandb

                        if wandb.run is not None:
                            wandb.log({"eval/pass_rate": pass_rate}, step=state.global_step)
                    except Exception:
                        pass

                if pass_rate > self.best_pass_rate:
                    self.best_pass_rate = pass_rate
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                    if self.no_improve_count >= self.patience:
                        logger.info(
                            f"Pass rate early stopping triggered: no improvement for "
                            f"{self.patience} consecutive pass-rate evals "
                            f"(best={self.best_pass_rate:.2f}%)."
                        )
                        should_stop = True

        if is_distributed:
            should_stop = self._broadcast_should_stop(should_stop)

        if should_stop:
            control.should_training_stop = True
