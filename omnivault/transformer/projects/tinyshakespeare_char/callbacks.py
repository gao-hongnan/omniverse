import os

import torch

from omnivault.transformer.core.trainer import Trainer


@torch.no_grad()
def evaluate_generate_and_save_on_train_batch_end(trainer: Trainer) -> None:
    """Evaluate, generate, and save the model on train batch end."""

    tokenizer = trainer.state.tokenizer
    generator_config = trainer.composer.generator

    if trainer.train_batch_index % 4 == 0:
        # Perform evaluation and sample generation
        model = trainer.model
        model.eval()

        context = "O God, O God!"
        starting_tokens = tokenizer.encode(context)
        starting_tokens = torch.tensor(starting_tokens, dtype=torch.long, device=trainer.device)

        generated_tokens = model.generate(starting_tokens, **generator_config.model_dump(mode="python"))
        generated_tokens = generated_tokens.squeeze(0)
        generated_tokens_decoded = tokenizer.decode(encoded_sequence=generated_tokens, remove_special_tokens=True)
        print(f"Generated: {generated_tokens_decoded}")

        # # Save the latest model
        ckpt_path = os.path.join(trainer.composer.trainer.save_dir, f"model_epoch_{trainer.epoch_index}_batch_{trainer.train_batch_index}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print("Model saved at", ckpt_path)

        # Revert model to training mode
        model.train()
