from __future__ import annotations

import copy
import logging
import sys
import time

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from rich.pretty import pprint
from tqdm import tqdm

from omnivault._types._alias import Missing
from omnivault._types._sentinel import MISSING
from omnivault.core.logger import RichLogger
from omnivault.transformer.config.composer import Composer, DataConfig
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.criterion import CRITERION_REGISTRY
from omnivault.transformer.config.decoder import DecoderConfig
from omnivault.transformer.config.generator import GeneratorConfig
from omnivault.transformer.config.global_ import MaybeGlobal
from omnivault.transformer.config.logger import LoggerConfig
from omnivault.transformer.config.optim import OPTIMIZER_REGISTRY
from omnivault.transformer.config.scheduler import SCHEDULER_REGISTRY, LambdaLRConfig
from omnivault.transformer.config.trainer import TrainerConfig
from omnivault.transformer.core.dataset import AdderDataset, create_loader, split_dataset
from omnivault.transformer.core.optim import apply_weight_decay_to_different_param_groups
from omnivault.transformer.core.scheduler import noam_lr_decay
from omnivault.transformer.core.state import State
from omnivault.transformer.core.tokenizer import AdderTokenizer
from omnivault.transformer.core.trainer import Trainer, TrainerEvent
from omnivault.transformer.core.vocabulary import AdderVocabulary
from omnivault.transformer.decoder.core import GPTDecoder
from omnivault.transformer.utils.config_utils import load_yaml_config, merge_configs
from omnivault.transformer.utils.reproducibility import seed_all

# TODO: I have a callable instead of _target_ field for me to use importlib to parse.
# so maybe consider using my own code base


@torch.no_grad()
def evaluate_and_generate_on_valid_epoch_end(
    trainer: Trainer,
    num_batches_to_eval: int | None = None,
) -> None:
    generator_config = trainer.composer.generator
    assert (
        generator_config.max_tokens == trainer.composer.constants.NUM_DIGITS + 1 + 1  # type: ignore[attr-defined]
    ), "In this dataset, the max tokens to generate is fixed and derived from the number of digits. If we add two 2-digits together, it does not make sense for us to keep generating since the max digits for answer is 3 digits, with an optional `<EOS` token if it is in our vocabulary."
    assert generator_config.greedy is True, "We should use greedy generation for this task in particular."

    vocabulary = trainer.state.vocabulary
    assert isinstance(vocabulary, AdderVocabulary)
    EQUAL = vocabulary.token_to_index[vocabulary.EQUAL]
    EOS = vocabulary.token_to_index[vocabulary.EOS]
    eos_token = torch.tensor([EOS], device=trainer.device)

    model = trainer.model
    model.eval()

    dataloader = trainer.test_loader
    assert dataloader is not None
    total_samples = 0
    num_batches = len(dataloader)

    total_correct_across_samples = 0
    progress_bar = tqdm(
        enumerate(dataloader, start=1), total=num_batches, desc="Evaluating and Generation.", leave=False
    )

    all_predictions = []
    for _batch_index, batch in progress_bar:
        inputs, _, _, _ = batch
        inputs = inputs.to(trainer.device)

        # inputs:   [14, 9, 8, 10, 9, 9, 13, 1, 9, 7]
        # targets:  [    9, 8, 10, 9, 9, 13, 1, 9, 7, 15]
        # equation: [14, 9, 8, 10, 9, 9, 13, 1, 9, 7, 15]
        # equal_index: 6'th position
        # starting_tokens: [14, 9, 8, 10, 9, 9, 13] -> <BOS>98+99=
        # generated_tokens: [14, 9, 8, 10, 9, 9, 13, 1, 9, 7, 15]
        # generated_tokens_decoded: 98+99=197

        batch_correct_predictions = 0
        batch_size = inputs.size(0)

        for input in inputs:
            total_samples += 1  # for sure it is 1 sample anyways

            equation = torch.cat((input, eos_token), 0)  # this is the answer also
            equation_decoded = trainer.state.tokenizer.decode(encoded_sequence=equation, remove_special_tokens=True)

            equal_mask = equation == EQUAL
            equal_index = torch.where(equal_mask)[0][0]
            starting_tokens = equation[: equal_index + 1].unsqueeze(0)

            generated_tokens = model.generate(
                starting_tokens=starting_tokens,
                **generator_config.model_dump(mode="python"),
            )
            generated_tokens = generated_tokens.squeeze(0)
            generated_tokens_decoded = trainer.state.tokenizer.decode(
                encoded_sequence=generated_tokens, remove_special_tokens=True
            )

            is_correct = torch.all(torch.eq(generated_tokens, equation))

            prediction_details = {
                "epoch": trainer.epoch_index,
                "batch_index": _batch_index,
                "equation": equation_decoded,
                "generated": generated_tokens_decoded,
                "correct": is_correct.item(),
            }

            if is_correct:
                total_correct_across_samples += 1
                batch_correct_predictions += 1

            batch_accuracy = batch_correct_predictions / batch_size
            progress_bar.set_postfix_str(f"accuracy: {batch_accuracy:.4f}")

            all_predictions.append(prediction_details)

        if num_batches_to_eval and _batch_index >= num_batches_to_eval:
            trainer.logger.info("Early stopping evaluation.")
            break

    trainer.logger.info("Correct/Total Samples: %d/%d", total_correct_across_samples, total_samples)
    accuracy = total_correct_across_samples / total_samples
    df = pd.DataFrame(all_predictions)

    trainer.logger.info("Accuracy: %s", accuracy)
    pprint(df)


def main(cfg: DictConfig | ListConfig) -> None:
    """Main driver."""
    seed_all(cfg.global_.seed)

    constants = MaybeConstant(**cfg.constants) if cfg.constants is not None else MaybeConstant()
    logger_pydantic_config = LoggerConfig(**cfg.logger)
    global_ = MaybeGlobal(**cfg.global_)
    data = DataConfig(**cfg.data)
    trainer_config = TrainerConfig(**cfg.trainer)
    generator_config = GeneratorConfig(**cfg.generator)

    # logger
    logger = RichLogger(**logger_pydantic_config.model_dump(mode="python")).logger
    assert isinstance(logger, logging.Logger)

    vocabulary = AdderVocabulary.from_tokens(tokens=constants.TOKENS, num_digits=constants.NUM_DIGITS)  # type: ignore[attr-defined]
    tokenizer = AdderTokenizer(vocabulary=vocabulary)

    # assign back model.vocab_size from ??? to vocabulary.vocab_size
    cfg.model.vocab_size = vocabulary.vocab_size
    model_config = instantiate(cfg.model)
    model_pydantic_config = DecoderConfig(**model_config)

    optimizer_config_cls = OPTIMIZER_REGISTRY[cfg.optimizer.name]
    optimizer_pydantic_config = optimizer_config_cls(**cfg.optimizer)

    criterion_config_cls = CRITERION_REGISTRY[cfg.criterion.name]
    criterion_pydantic_config = criterion_config_cls(**cfg.criterion)

    composer = Composer(
        constants=constants,
        logger=logger_pydantic_config,
        global_=global_,
        data=data,
        model=model_pydantic_config,
        optimizer=optimizer_pydantic_config,
        criterion=criterion_pydantic_config,
        trainer=trainer_config,
        generator=generator_config,
    )
    assert composer.model is not MISSING and not isinstance(composer.model, Missing)
    assert composer.optimizer is not MISSING and not isinstance(composer.optimizer, Missing)
    assert composer.criterion is not MISSING and not isinstance(composer.criterion, Missing)

    # TODO: consider classmethod from file_path
    assert composer.data.dataset_path is not None
    with open(composer.data.dataset_path, "r") as file:
        sequences = [line.strip() for line in file]

    dataset = AdderDataset(data=sequences, tokenizer=tokenizer)
    if composer.data.split:
        train_dataset, valid_dataset, test_dataset = split_dataset(
            dataset=dataset, split=composer.data.split, seed=composer.global_.seed
        )
    else:
        # no need to cater to mypy as either Subset or Dataset is fine.
        train_dataset = dataset  # type: ignore[assignment]

    # you do these asserts to make sure that the loaders are not None
    # because create loader expects non-None loaders and collate_fn.
    # if you don't do these asserts, mypy cannot guarantee that the loaders are not None
    # so they cannot infer properly.
    assert composer.data.train_loader is not None
    assert composer.data.valid_loader is not None
    assert composer.data.test_loader is not None
    assert composer.data.collate_fn is not None

    train_loader = create_loader(
        dataset=train_dataset,
        loader_config=composer.data.train_loader,
        collate_fn_config=composer.data.collate_fn,
    )

    if valid_dataset is not None:
        valid_loader = create_loader(
            dataset=valid_dataset,
            loader_config=composer.data.valid_loader,
            collate_fn_config=composer.data.collate_fn,
        )

    if test_dataset is not None:
        test_loader = create_loader(  # noqa: F841
            dataset=test_dataset,
            loader_config=composer.data.test_loader,
            collate_fn_config=composer.data.collate_fn,
        )

    # Create model
    model = GPTDecoder(model_pydantic_config).to(composer.trainer.device)

    # Create optimizer based on model parameters
    if composer.trainer.apply_weight_decay_to_different_param_groups:
        assert hasattr(composer.optimizer, "weight_decay")
        optimizer = optimizer_pydantic_config.build(
            params=apply_weight_decay_to_different_param_groups(
                model=model, weight_decay=composer.optimizer.weight_decay
            )
        )
    else:
        optimizer = optimizer_pydantic_config.build(params=model.parameters())

    # Create criterion
    criterion = criterion_pydantic_config.create_instance()
    assert criterion.ignore_index == vocabulary.token_to_index[vocabulary.PAD]

    # Create Scheduler Noam
    # TODO: this part is hardcoded in a way since we are using LambdaLR.
    # I do not have time to make it more "automated" so this is anti-config-pattern.
    warmup_steps = 3 * len(train_loader)

    # lr first increases in the warmup steps, and then decays
    noam = lambda step: noam_lr_decay(step, d_model=composer.model.d_model, warmup_steps=warmup_steps)  # noqa: E731

    scheduler_config_cls = SCHEDULER_REGISTRY[cfg.scheduler.name]

    if issubclass(scheduler_config_cls, LambdaLRConfig):
        scheduler_pydantic_config = scheduler_config_cls(lr_lambda=noam, **cfg.scheduler)
    else:
        scheduler_pydantic_config = scheduler_config_cls(**cfg.scheduler)  # type: ignore[assignment]

    assert composer.scheduler is MISSING  # now it is MISSING for us to fill up.
    composer.scheduler = scheduler_pydantic_config
    scheduler = scheduler_pydantic_config.build(optimizer=optimizer)

    composer.pretty_print()
    time.sleep(1)

    state = State(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
    )
    state.pretty_print()
    time.sleep(1)

    device = composer.trainer.device

    # train
    trainer = Trainer(
        state=state,
        composer=composer,
        logger=logger,
        device=device,  # type: ignore[arg-type]
    )
    trainer.add_callback(
        TrainerEvent.ON_VALID_EPOCH_END.value,
        lambda trainer: evaluate_and_generate_on_valid_epoch_end(trainer, num_batches_to_eval=None),
    )
    _trained_state = trainer.fit(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader)
    _trained_state.pretty_print()

    loaded_state = State.load_snapshots(
        filepath=trainer.best_checkpoint_path,
        device=device,  # type: ignore[arg-type]
        model=copy.deepcopy(model),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert _trained_state == loaded_state, "Loading the last saved state, should be the same as the trained state."


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    args_list = sys.argv[2:]

    yaml_cfg = load_yaml_config(yaml_path)
    cfg = merge_configs(yaml_cfg, args_list)
    om.resolve(cfg)  # inplace ops

    main(cfg)
    # epoch 2 : 1.38283, 1.15267
    # epoch 20: 0.11087, 0.04235
