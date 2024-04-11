import argparse
import logging
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompt_optimization as prompt_opt

# Setup argument parser to get command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--discrete-optimizer", type=str, default="gcg")
parser.add_argument("--log-dir", type=str, default="experiments")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-410m")
parser.add_argument("--num-steps", type=int, default=200)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--topk", type=int, default=250)
args = parser.parse_args()

# Set randomness
if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Generate a unique ID for the run and create the experiments directory
run_id = 'example'
os.makedirs(f"outputs/", exist_ok=True)
# Setup logging configuration
logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s] %(message)s",
                    datefmt="%Y%m%d %H:%M:%S",
                    handlers=[logging.FileHandler(f"outputs/{run_id}.log"), logging.StreamHandler()])
logging.info(f"run id: {run_id}")
print(f"run id: {run_id}")
for arg, value in vars(args).items():
    logging.info(f"{arg}: {value}")

# Device, model, and tokenizer setup
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    model_args = dict(trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map="auto")
else:
    model_args = dict(trust_remote_code=False, low_cpu_mem_usage=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_args)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# model = model.to(device)

############################################################################################################
# Room to play around:
# the input_ids is constructed in prep_text() and is a concatenation in this order:
# [chat_template[0], system_prompt, input_str, free_tokens, chat_template[1], target_str]
# Prepare input and target tokens from strings
num_free_tokens = 10
input_str = " "
target_str = "To be or not to be, that is the question."
system_prompt = ""
chat_template = ("", "")
############################################################################################################

input_ids, free_token_slice, input_slice, target_slice, loss_slice = prompt_opt.prep_text(input_str,
                                                                                          target_str,
                                                                                          tokenizer,
                                                                                          system_prompt,
                                                                                          chat_template,
                                                                                          num_free_tokens,
                                                                                          device)
# Optimize the input tokens to generate the target string
if args.discrete_optimizer == "gcg":
    solution = prompt_opt.optimize_gcg(model, input_ids, input_slice, free_token_slice, target_slice,
                                       loss_slice, args.num_steps, batch_size=args.batch_size, topk=args.topk)
elif args.discrete_optimizer == "random_search":
    solution = prompt_opt.optimize_random_search(model, input_ids, input_slice, free_token_slice,
                                                 target_slice, loss_slice, args.num_steps, batch_size=args.batch_size)
else:
    raise ValueError("discrete_optimizer must be one of ['gcg', 'random_search']")

# Test the prompt and log the new generation with the target string
logging.info(f"Hard tokens returned:")
optimized_ids = solution["input_ids"]
output = model.generate(input_ids=optimized_ids[input_slice].unsqueeze(0), max_new_tokens=20, do_sample=False)
logging.info(f"solution: {tokenizer.decode(optimized_ids[input_slice], skip_special_tokens=True)}")
logging.info(f"goal: {tokenizer.decode(input_ids[target_slice], skip_special_tokens=True)}")
logging.info(f"output: {tokenizer.decode(output[0, target_slice], skip_special_tokens=True)}")

