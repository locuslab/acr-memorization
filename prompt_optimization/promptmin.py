"""
promptmin.py
an implementation of promptmin

developed in collaboration by: Avi Schwarzschild and Zhili Feng and Pratyush Maini in 2024
"""
import logging

import prompt_optimization as prompt_opt


def minimize_prompt(model, tokenizer, input_str, target_str, system_prompt, chat_template, device, optimization_args,
                    max_tokens=30):
    n_tokens_in_prompt = 5
    running_max = max_tokens
    running_min = 0
    success = False
    best_prompt = None
    done = False
    best_slices = (None, None, None, None)

    while not done:
        logging.info("\n------------------------------------\n")
        logging.info(f"{n_tokens_in_prompt} tokens in the prompt")
        input_ids, free_token_slice, input_slice, target_slice, loss_slice = prompt_opt.prep_text(input_str,
                                                                                                  target_str,
                                                                                                  tokenizer,
                                                                                                  system_prompt,
                                                                                                  chat_template,
                                                                                                  n_tokens_in_prompt,
                                                                                                  device)
        if running_max == -1:
            running_max = (target_slice.stop - target_slice.start) * 5
        if optimization_args["discrete_optimizer"] == "gcg":
            solution = prompt_opt.optimize_gcg(model, input_ids, input_slice, free_token_slice, target_slice,
                                               loss_slice, optimization_args["num_steps"],
                                               batch_size=optimization_args["batch_size"],
                                               topk=optimization_args["topk"],
                                               mini_batch_size=optimization_args["mini_batch_size"])
        elif optimization_args["discrete_optimizer"] == "random_search":
            solution = prompt_opt.optimize_random_search(model, input_ids, input_slice, free_token_slice,
                                                         target_slice, loss_slice, optimization_args["num_steps"],
                                                         batch_size=optimization_args["batch_size"],
                                                         mini_batch_size=optimization_args["mini_batch_size"])
        else:
            raise ValueError(
                "discrete_optimizer must be one of ['gcg', 'random_search']")

        target_acquired = prompt_opt.check_output_with_hard_tokens(model, solution["input_ids"].unsqueeze(0),
                                                                   target_slice,
                                                                   loss_slice)

        if target_acquired:
            logging.info(f"Target acquired with {n_tokens_in_prompt} tokens in the prompt")
            running_max = n_tokens_in_prompt
            success = True
            best_prompt = solution["input_ids"]
            new_num_tokens = n_tokens_in_prompt - 1
            best_slices = (free_token_slice, input_slice, target_slice, loss_slice)
        else:
            logging.info(f"Target NOT acquired with {n_tokens_in_prompt} tokens in the prompt")
            new_num_tokens = n_tokens_in_prompt + 5
            running_min = n_tokens_in_prompt
            optimization_args["num_steps"] = int(optimization_args["num_steps"] * 1.2)

        if (new_num_tokens >= running_max) or (new_num_tokens <= running_min):
            done = True
        else:
            n_tokens_in_prompt = new_num_tokens

    output = {"free_token_slice": best_slices[0] if best_slices[0] is not None else free_token_slice,
              "input_slice": best_slices[1] if best_slices[1] is not None else input_slice,
              "target_slice": best_slices[2] if best_slices[2] is not None else target_slice,
              "loss_slice": best_slices[3] if best_slices[3] is not None else loss_slice,
              "success": success,
              "num_free_tokens": running_max,
              "input_ids": best_prompt,
              }
    return output
