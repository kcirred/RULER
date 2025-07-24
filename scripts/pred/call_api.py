# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
from tqdm import tqdm
from pathlib import Path
import traceback
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from transformers.models.bamba.modeling_bamba import BambaMixer

import torch

SERVER_TYPES = (
    'trtllm',
    'vllm',
    'sglang',
    'openai',
    'gemini',
    'hf',
    'mamba',
    'fms',
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

# Server
parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5002')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo',
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')

# Inference
parser.add_argument("--fms_variant", type=str, default='llama_1b', help='provide the variant such as llama_1b, mamba_9.8b')
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--hook", type=bool, default=True)

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))
if args.server_type == 'hf' or args.server_type == 'gemini' or args.server_type == 'fms':
    args.threads = 1

class Hook4ssm_states(torch.nn.Module):
    """
    Hook to be called before recalculating/updating ssm_states.
    Note
    1. hook's call signature is different because of "with_kwargs=True"
    2. kwargs["cache_params"].ssm_states[mod.layer_idx] is the current ssm_states
    3. find current seq_len by a) kwargs["hidden_states"].shape = batch_size, seq_len, emb
        or b) len(kwargs["cache_position"])

    """
    def __init__(self, *args, **kwargs):
        pers_to_rec = [0.005, 0.25, 0.5, 0.75, 0.995]
        self.layer_name = kwargs.pop("lay_name", None)
        self.layer_idx = kwargs.pop("lay_idx", None)
        self.quant_opt = kwargs.pop("quant_opt", "no_quant")
        self.rec_scales = kwargs.pop("rec_scale", True)
        self.is_vllm = kwargs.pop("is_vllm", False)
        self.scales = []
        self.dist = {per:[] for per in pers_to_rec}
        self.calib_on_nth_token = kwargs.pop("calib_on_nth_token", -1)
        self.calib_scale = None
        super().__init__(*args, **kwargs)

    def __call__(self, mod, args, kwargs):
        # vllm's mixer is using pos args, "mamba_cache_params" is at [1]
        pers_to_rec = [0.005, 0.25, 0.5, 0.75, 0.995]
        if self.is_vllm:
            # TODO will need to update this part if vllm V1 is needed
            state_indices_tensor = args[1].state_indices_tensor
            attn_metadata = get_forward_context().attn_metadata
            num_prefills = attn_metadata.num_prefills  # request count
            num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
            state_indices_tensor_p, state_indices_tensor_d = torch.split(
                state_indices_tensor,
                [num_prefills, num_decodes],
                dim=0,
            )

            if args[1] is None or len(state_indices_tensor_p) == 0:
                return

            ssm_states_lay_i = args[1].ssm_state[state_indices_tensor_p]
        else:
            if kwargs["cache_params"] is None:
                return

            ssm_states_lay_i = kwargs["cache_params"].ssm_states[mod.layer_idx]

        # here we can manipulate ssm_states
        org_dtype = ssm_states_lay_i.dtype
        scale = self.calib_scale  # will be None unless calibration is triggered already
        tok_id = len(self.scales)
        if tok_id == self.calib_on_nth_token:
            self.calib_scale = torch.tensor(max(self.scales), device=ssm_states_lay_i.device)
        # dump the entire ssm_states at a given token pos for offline data analysis
        # if self.layer_idx == 0 and tok_id in dump_tensors_tok_id:
        #     torch.save(kwargs["cache_params"].ssm_states, f"bamba_ssm_states_tok{tok_id}.pt")

        tar_dtype = (torch.float8_e4m3fn if "fp8" in self.quant_opt else
                     torch.int8 if "int" in self.quant_opt else
                     org_dtype)
        match self.quant_opt:
            case "fp8_cast":
                # option 1: direct cast
                qmin, qmax = torch.finfo(tar_dtype).min, torch.finfo(tar_dtype).max
                scale = torch.tensor(1.0, device=ssm_states_lay_i.device)

            case "fp8_dyn_perT":
                # option 2: per tensor dynamic or static
                qmin, qmax = torch.finfo(tar_dtype).min, torch.finfo(tar_dtype).max
                if scale is None:
                    scale = (ssm_states_lay_i.abs().amax()/ qmax).clamp(min=1e-5)

            case "int4_dyn_perT":
                # option 3: int 4
                qmin, qmax = -8, 7  # in reality -7 to +7
                if scale is None:
                    scale = (ssm_states_lay_i.abs().amax()/ qmax).clamp(min=1e-5)

            case "clamp_to_1e-5":
                # debug only
                ssm_states_lay_i = ssm_states_lay_i.clamp(min=0, max=1e-5)
                scale = torch.tensor(1.0, device=ssm_states_lay_i.device)

            case "zero":
                # debug only
                ssm_states_lay_i = ssm_states_lay_i.fill_(0)
                scale = torch.tensor(1.0, device=ssm_states_lay_i.device)

            # case _:
            #     # default
        # NOTE for "no_quant" case, scale will still be None

        # manipulate ssm_states as needed
        if scale is not None:
            if tar_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                ssm_states_lay_i = (ssm_states_lay_i/scale).clamp(min=qmin, max=qmax).to(tar_dtype).to(org_dtype)
            elif tar_dtype in [torch.int8]:
                ssm_states_lay_i = (ssm_states_lay_i/scale).round().clamp(min=qmin, max=qmax)
            # no scaling/operation on ssm_states for other dtypes

            # NOTE record distributions in "quantized" range, i.e. FP8 or INT4 before *scale
            if self.rec_scales and tok_id<20000:
                self.scales.append(scale.item())
                quantiles = torch.quantile(ssm_states_lay_i.float(),
                                           torch.tensor(pers_to_rec, device=ssm_states_lay_i.device))
                for per, val in zip(pers_to_rec, quantiles.tolist()):
                    self.dist[per].append(val)

            ssm_states_lay_i *= scale

        # inplace update the cache_params before return
        if self.is_vllm:
            args[1].ssm_state[state_indices_tensor_p] = ssm_states_lay_i
        else:
            kwargs["cache_params"].ssm_states[mod.layer_idx].copy_(ssm_states_lay_i)

        # NOTE must return (args, kwargs) if we want to modify kwargs
        return args, kwargs



def get_llm(tokens_to_generate):
    if args.server_type == 'trtllm':
        from client_wrappers import TRTLLMClient
        llm = TRTLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
            max_attention_window_size=args.sliding_window_size,
        )

    elif args.server_type == 'vllm':
        from client_wrappers import VLLMClient
        llm = VLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'sglang':
        from client_wrappers import SGLClient
        llm = SGLClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'openai':
        from client_wrappers import OpenAIClient
        llm = OpenAIClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'gemini':
        from client_wrappers import GeminiClient
        llm = GeminiClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == 'hf':
        from model_wrappers import HuggingFaceModel
        llm = HuggingFaceModel(
            name_or_path=args.model_name_or_path,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )

    elif args.server_type == 'fms':
        from model_wrappers import FMSModel
        llm = FMSModel(
            name_or_path=args.model_name_or_path,
            variant=args.fms_variant,
            do_sample=args.temperature > 0,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )
    elif args.server_type == 'mamba':
        from model_wrappers import MambaModel
        # mamba uses its own generation function, do not pass in do_sample
        # https://github.com/state-spaces/mamba/blob/009bec5ee37f586844a3fc89c040a9c1a9d8badf/mamba_ssm/utils/generation.py#L121
        llm = MambaModel(
            name_or_path=args.model_name_or_path,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )

    else:
        raise RuntimeError(f'Unsupported server type {args.server_type}')

    return llm


def main():
    start_time = time.time()

    curr_folder = os.path.dirname(os.path.abspath(__file__))

    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')

    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'

    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'

    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample['index'] for sample in read_manifest(pred_file)]
        data = [sample for sample in read_manifest(task_file) if sample['index'] not in pred_index]
    else:
        data = read_manifest(task_file)

    # Load api
    llm = get_llm(config['tokens_to_generate'])


    if args.hook:
        hooks = []  # for accessing hook itself
        h_hooks = []  # for hook removal
        for n, m in llm.model.named_modules():
            if isinstance(m, BambaMixer):  # and quant_opt is not None:  # m.layer_idx not in layers_to_skip:
                lay_idx = getattr(m, "layer_idx", n.split(".")[2])
                hooks.append(
                    Hook4ssm_states(
                        lay_name=n,
                        lay_idx=lay_idx,
                        quant_opt='fp8_dyn_perT',
                        rec_scale=False,  # quant_opt is not None,
                        is_vllm=False,
                    )
                )
                h_hooks.append(m.register_forward_pre_hook(hooks[-1], with_kwargs=True,))

    def get_output(idx_list, index_list, input_list, outputs_list, others_list, truncation_list, length_list):
        nonlocal llm

        while True:
            try:
                pred_list = llm.process_batch(prompts=input_list)
                break
            except Exception as e:
                traceback.print_exc()

        zipped_iter = zip(pred_list, idx_list, index_list, input_list,
                          outputs_list, others_list, truncation_list, length_list)

        for pred, idx, index, input, outputs, others, truncation, length in zipped_iter:
            if isinstance(pred['text'], str):
                pred_text = pred['text']
            elif len(pred['text']) > 0:
                pred_text = pred['text'][0]
            else:
                pred_text = ''

            outputs_parallel[idx] = {
                'index': index,
                'pred': pred_text,
                'input': input,
                'outputs': outputs,
                'others': others,
                'truncation': truncation,
                'length': length,
            }

    threads = []
    outputs_parallel = [{} for _ in range(len(data))]

    batched_data = []
    batch = []
    for idx, data_point in enumerate(data):
        data_point['idx'] = idx

        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []

        batch.append(data_point)

    if len(batch):
        batched_data.append(batch)

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        # the data is processed sequentially, so we can store the start and end of current processing window
        start_idx = 0  # window: [start_idx, end_idx]

        for batch_idx, batch in tqdm(enumerate(batched_data), total=len(batched_data)):
            idx_list = [data_point['idx'] for data_point in batch]
            end_idx = idx_list[-1]  # the data in a batch is ordered

            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    idx_list=idx_list,
                    index_list=[data_point['index'] for data_point in batch],
                    input_list=[data_point['input'] for data_point in batch],
                    outputs_list=[data_point['outputs'] for data_point in batch],
                    others_list=[data_point.get('others', {}) for data_point in batch],
                    truncation_list=[data_point.get('truncation', -1) for data_point in batch],
                    length_list=[data_point.get('length', -1) for data_point in batch],
                ),
            )
            thread.start()
            threads.append(thread)

            is_last_batch = (batch_idx == len(batched_data) - 1)

            if (len(threads) == args.threads) or is_last_batch:
                for thread in threads:
                    thread.join()
                threads = []

                # dump the results in current processing window on disk
                for idx in range(start_idx, end_idx + 1):
                    if len(outputs_parallel[idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[idx]) + '\n')

                start_idx = end_idx + 1

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
    main()
