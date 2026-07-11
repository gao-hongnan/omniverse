import torch

from omnivault.transformer.core.trainer import Trainer
from omnivault.transformer.decoder.core import GPTDecoder


@torch.no_grad()
def evaluate_generate_on_train_batch_end(trainer: Trainer) -> None:
    """Evaluate and generate on train batch end."""

    tokenizer = trainer.state.tokenizer
    generator_config = trainer.composer.generator

    if trainer.train_batch_index % trainer.eval_every_n_steps == 0:
        # Perform evaluation and sample generation
        model = trainer.model
        assert isinstance(model, GPTDecoder)
        model.eval()

        context = "O God, O God!"
        encoded_tokens = tokenizer.encode(context)
        starting_tokens: torch.LongTensor = torch.tensor(encoded_tokens, dtype=torch.long, device=trainer.device)  # type: ignore[assignment]

        generated_tokens: torch.Tensor = model.generate(starting_tokens, **generator_config.model_dump(mode="python"))
        generated_tokens = generated_tokens.squeeze(0)
        generated_tokens_decoded = tokenizer.decode(encoded_sequence=generated_tokens, remove_special_tokens=True)
        trainer.logger.info("Generated text %s", generated_tokens_decoded)

        # Revert model to training mode
        model.train()
