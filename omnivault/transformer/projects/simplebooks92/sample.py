# type: ignore
# ruff: noqa


def next(prompt: tf.Tensor, cache, index):
    prompt = prompt.numpy()
    prompt = torch.from_numpy(prompt).to(composer.trainer.device)
    #     print(prompt)
    #     print(prompt.shape)
    #     print(model(prompt))
    #     print(index)

    index = int(index)
    with torch.no_grad():
        logits = model(prompt)[:, index - 1, :]
    logits = logits.detach().cpu().numpy()
    logits = tf.convert_to_tensor(logits)
    # Ignore hidden states for now; only needed for contrastive search.
    hidden_states = None
    return logits, hidden_states, cache


@torch.no_grad()
def generate_on_train_epoch_end(trainer: Trainer) -> None:
    """Evaluate and generate on train batch end."""

    def get_samplers() -> List[Sampler]:
        samplers = [
            keras_nlp.samplers.GreedySampler(temperature=1.0),
            keras_nlp.samplers.BeamSampler(num_beams=10, temperature=1.0),
            keras_nlp.samplers.RandomSampler(temperature=1.0),
            keras_nlp.samplers.TopKSampler(k=10, temperature=1.0),
            keras_nlp.samplers.TopPSampler(p=0.5, temperature=1.0),
        ]
        return samplers

    samplers = get_samplers()

    # The "packer" layers adds the [BOS] token for us.
    prompt_tokens = start_packer(tokenizer([""]))

    for sampler in samplers:
        generated_tokens = sampler(
            next=next,
            prompt=prompt_tokens,
            index=1,  # Start sampling immediately after the [BOS] token.
        )
        generated_tokens_decoded = tokenizer.detokenize(output_tokens)
        sampler_name = sampler.__class__.__name__
        # generated_tokens_decoded = tokenizer.decode(encoded_sequence=generated_tokens, remove_special_tokens=True)
        trainer.logger.info("%s search Generated text %s", sampler_name, generated_tokens_decoded)
    # Revert model to training mode
    model.train()


trainer.add_callback(TrainerEvent.ON_TRAIN_EPOCH_END.value, generate_on_train_epoch_end)
