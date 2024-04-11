"""
random_search.py
an implementation of random search
Proposed for prompt optimization in
Adversarial attacks on gpt-4 via simple random search. 2023. by Maksym Andriushchenko

developed in collaboration by: Avi Schwarzschild and Zhili Feng and Pratyush Maini in 2024
"""
import logging

import torch


def optimize_random_search(model, input_ids, input_slice, free_token_slice, target_slice, loss_slice,
                           num_steps, batch_size=100, mini_batch_size=100):
    with torch.no_grad():
        # Get embedding matrix
        embedding_matrix = model.get_input_embeddings().weight

        best_loss = torch.inf
        best_input = input_ids.clone()

        # Random search optimization loop
        for i in range(num_steps):
            # Get random batch of single token perturbations for the free tokens
            free_token_ids = input_ids[free_token_slice]
            free_tokens_batch = free_token_ids.repeat(batch_size, 1)
            new_token_loc = torch.randint(0, free_token_ids.size(0), (batch_size,), device=input_ids.device)
            new_token_vals = torch.randint(0, embedding_matrix.size(0), (batch_size,), device=input_ids.device)
            free_tokens_batch[torch.arange(batch_size), new_token_loc] = new_token_vals
            batch_input_ids = input_ids.repeat(batch_size, 1)
            batch_input_ids[:, free_token_slice] = free_tokens_batch

            loss = torch.zeros(batch_size)
            for mini_batch in range(0, batch_size, mini_batch_size):
                output = model(input_ids=batch_input_ids[mini_batch:mini_batch + mini_batch_size])
                labels = input_ids[target_slice].repeat(output.logits.size(0), 1)
                loss_mini_batch = torch.nn.functional.cross_entropy(output.logits[:, loss_slice].transpose(1, 2),
                                                                    labels,
                                                                    reduction="none")
                loss[mini_batch:mini_batch + mini_batch_size] = loss_mini_batch.mean(dim=-1)
            best_candidate = torch.argmin(loss)

            input_ids = batch_input_ids[best_candidate]

            # compute test loss
            output_single = model(input_ids=input_ids.unsqueeze(0))
            match = (output_single.logits[0, loss_slice].argmax(-1) == input_ids[target_slice].squeeze())
            logging.info(f"step: {i:<4} | "
                         f"loss: {loss[best_candidate].mean().item():0.6f} | "
                         f"{match.int().tolist()} | ")
            if match.all():
                best_input = input_ids.clone()
                break
            if loss[best_candidate].mean().item() < best_loss:
                best_loss = loss[best_candidate].mean().item()
                best_input = input_ids.clone()

    return {"input_ids": best_input, "inputs_embeds": model.get_input_embeddings()(best_input).unsqueeze(0)}
