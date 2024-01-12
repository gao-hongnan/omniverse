from __future__ import annotations

import copy
import logging
import sys
import time
from typing import List

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om

from omnivault._types._alias import Missing
from omnivault._types._sentinel import MISSING
from omnivault.core.logger import RichLogger
from omnivault.transformer.config.composer import Composer, DataConfig
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.criterion import CRITERION_REGISTRY
from omnivault.transformer.config.decoder import DecoderConfig
from omnivault.transformer.config.global_ import MaybeGlobal
from omnivault.transformer.config.logger import LoggerConfig
from omnivault.transformer.config.optim import OPTIMIZER_REGISTRY
from omnivault.transformer.config.scheduler import SCHEDULER_REGISTRY, LambdaLRConfig
from omnivault.transformer.config.trainer import TrainerConfig
from omnivault.transformer.core.dataset import (
    AdderDataset,
    construct_dummy_batch_future_masks,
    construct_dummy_batch_target_padding_masks,
    create_loader,
    split_dataset,
)
from omnivault.transformer.core.optim import apply_weight_decay_to_different_param_groups
from omnivault.transformer.core.scheduler import noam_lr_decay
from omnivault.transformer.core.state import State
from omnivault.transformer.core.tokenizer import AdderTokenizer, Vocabulary_t
from omnivault.transformer.core.trainer import Trainer
from omnivault.transformer.core.vocabulary import AdderVocabulary
from omnivault.transformer.decoder.core import GPTDecoder
from omnivault.transformer.utils.config_utils import load_yaml_config, merge_configs
from omnivault.transformer.utils.reproducibility import seed_all

# TODO: I have a callable instead of _target_ field for me to use importlib to parse.
# so maybe consider using my own code base?


def decode_equation(vocab: Vocabulary_t, equation: torch.Tensor | List[int]) -> str:
    """
    Convert an equation in list format to string format.

    Parameters
    ----------
    equation : List[int]
        The equation in list format.

    Returns
    -------
    str
        The equation in string format.
    """
    if isinstance(equation, torch.Tensor):
        equation = equation.tolist()

    index_to_token = vocab.index_to_token

    UNK = vocab.token_to_index[vocab.UNK]
    decoded_equation = "".join([str(index_to_token.get(x, UNK)) for x in equation])
    return decoded_equation.replace(vocab.BOS, "").replace(vocab.EOS, "")

@torch.no_grad()
def compute_sum(model, x, num_digits=2, EOS=15):
    # x=[[15,  9,  8, 10,  3,  5, 13]]
    for _ in range(num_digits + 2):
        #print(x)
        #print(decode_equation(vocab=vocab, equation=x[0]))
        # print(x.shape)
        batch_size, seq_len = x.size()
        #pad_mask = (x != PAD).view(1, 1, 1, x.size(-1)).to(DEVICE)
        pad_mask = construct_dummy_batch_target_padding_masks(batch_size=batch_size, seq_len=seq_len)
        future_mask = construct_dummy_batch_future_masks(batch_size=batch_size, seq_len=seq_len)
        #print(future_mask.shape)
        #print(pad_mask.shape)

        #future_mask = future_mask.view(1, seq_len, seq_len).expand(size=(batch_size, -1, -1)).unsqueeze(1)
        #print(pad_mask.shape, future_mask.shape)
        #inputs, targets, target_padding_masks, future_masks = construct_batches(x)
        #print(target_padding_masks.shape, future_masks.shape)
        logits = model(input_tokens=x, target_padding_masks=pad_mask, future_masks=future_mask)

        #logits = model(inputs, target_padding_masks=target_padding_masks, future_masks=future_masks)

        last_output = logits.argmax(-1)[:, -1].view(1, 1)
        x = torch.cat((x, last_output), 1)
        # STOPPING CONDITION!
        if last_output.item() == EOS:
            break
        #return
    return x[0]

# def evaluate(model, dataloader, num_batch=None):
#     """
#     Function for evaluation the model.

#     This function take equations, and truncate them up to the equal-sign, and feed
#     them to the model to get the predictions, compare them with the correct answers,
#     and output the accuracy.
#     """
#     model.eval()
#     acc, count = 0, 0
#     num_wrong_to_display = 5
#     for idx, batch in enumerate(dataloader):
#         (
#             inputs,
#             targets,
#             target_padding_masks,
#             future_masks,
#         ) = batch  # construct_batches(batch)
#         for equation in inputs:
#             # pprint(equation)
#             # add EOS behind equation
#             equation = torch.cat((equation, torch.tensor([EOS])), 0) # TODO: PLEASE DO NOT DO THIS - DO NOT MODIFY LIKE THIS.
#             # fmt: off
#             loc_equal_sign = equation.tolist().index(EQUAL)
#             loc_EOS        = equation.tolist().index(EOS)
#             input          = equation[0 : loc_equal_sign + 1].view(1, -1).to(DEVICE)
#             ans            = equation[: loc_EOS + 1].tolist()
#             ans_pred       = compute_sum(model, input)
#             count += 1
#             # fmt: on

#             if ans == ans_pred.tolist():
#                 acc += 1
#             else:
#                 if num_wrong_to_display > 0:
#                     print(
#                         f'correct equation: {decode_equation(vocab=vocab, equation=equation).replace("<PAD>","")}'
#                     )
#                     print(f"wrongly predicted as:        {decode_equation(vocab=vocab, equation=ans_pred)}")
#                     num_wrong_to_display -= 1
#         if num_batch and idx > num_batch:
#             break
#     return acc / count


@torch.no_grad()
def generate_evaluation_samples(trainer: Trainer, num_samples=5) -> None:
    """
    Generates and logs samples from the evaluation/validation dataset.

    Args:
        trainer (Trainer): The trainer instance containing the model and dataloaders.
        num_samples (int): Number of samples to generate and log.
    """
    EQUAL = 13
    EOS = 15
    BOS = 14
    PAD = 16

    model = trainer.model
    dataloader = trainer.valid_dataloader  # Assuming this is your validation dataloader
    model.eval()

    acc, count = 0, 0
    num_wrong_to_display = 5


    for index, batch in enumerate(dataloader):
        (
            inputs,
            targets,
            target_padding_masks,
            future_masks,
        ) = batch
        inputs = inputs.to(trainer.device)

        for equation in inputs:
            count += 1
            # pprint(equation)
            # add EOS behind equation
            eos_token = torch.tensor([EOS], device=trainer.device)

            equation = torch.cat((equation, eos_token), 0)
            loc_equal_sign = equation.tolist().index(EQUAL)
            loc_EOS = equation.tolist().index(EOS)
            input = equation[0 : loc_equal_sign + 1].view(1, -1).to(trainer.device)
            ans = equation[: loc_EOS + 1].tolist()
            ans_pred = compute_sum(model, input)
            print(f"ans_pred: {ans_pred}")
            answer_decoded = decode_equation(vocab=trainer.state.vocabulary, equation=ans_pred)
            print(f"answer_decoded: {answer_decoded}")
            if count > 5:
                break


def main(cfg: DictConfig | ListConfig) -> None:
    """Main driver."""
    seed_all(cfg.global_.seed)

    constants = MaybeConstant(**cfg.constants) if cfg.constants is not None else MaybeConstant()
    logger_pydantic_config = LoggerConfig(**cfg.logger)
    global_ = MaybeGlobal(**cfg.global_)
    data = DataConfig(**cfg.data)
    trainer_config = TrainerConfig(**cfg.trainer)

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
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset=dataset, split=composer.data.split, seed=composer.global_.seed
        )

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

    valid_loader = create_loader(
        dataset=val_dataset,
        loader_config=composer.data.valid_loader,
        collate_fn_config=composer.data.collate_fn,
    )

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
    trainer.add_callback("on_valid_epoch_end", generate_evaluation_samples)
    _trained_state = trainer.fit(train_loader=train_loader, valid_loader=valid_loader)
    # _trained_state.pretty_print()

    loaded_state = State.load_snapshots(
        filepath=f"{composer.trainer.save_dir}/model_checkpoint_epoch_2.pt",
        device=device,  # type: ignore[arg-type]
        model=copy.deepcopy(model),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert _trained_state == loaded_state, "Cherry picked 2 epochs, so the last trained state should be the same."


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    args_list = sys.argv[2:]

    yaml_cfg = load_yaml_config(yaml_path)
    cfg = merge_configs(yaml_cfg, args_list)
    om.resolve(cfg)  # inplace ops

    main(cfg)
    # 1.38283, 1.15584
