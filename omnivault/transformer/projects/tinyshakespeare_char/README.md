```bash
python omnivault/transformer/projects/tinyshakespeare_char/main.py \
    omnivault/transformer/projects/tinyshakespeare_char/config.yaml \
    global_.debug=true \
    trainer.max_epochs=5 \
    generator.max_tokens=100 \
    trainer.device=cpu
```

```bash
python omnivault/transformer/projects/tinyshakespeare_char/main.py \
    omnivault/transformer/projects/tinyshakespeare_char/config.yaml \
    global_.debug=False \
    trainer.max_epochs=5 \
    trainer.eval_every_n_steps=4000 \
    trainer.log_every_n_steps=10000 \
    trainer.device=cpu \
    data.context_length=256 \
    generator.max_tokens=100
```

## GPU

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NPROC=$(nproc)
nohup python omnivault/transformer/projects/tinyshakespeare_char/main.py \
    omnivault/transformer/projects/tinyshakespeare_char/config.yaml \
    global_.debug=False \
    data.train_loader.batch_size=128 \
    data.train_loader.num_workers=8 \
    data.context_length=256 \
    optimizer.name=torch.optim.AdamW \
    optimizer.lr=0.0005 \
    optimizer.weight_decay=0.01 \
    criterion.name=torch.nn.CrossEntropyLoss \
    criterion.reduction=mean \
    criterion.label_smoothing=0.0 \
    scheduler.name=torch.optim.lr_scheduler.CosineAnnealingLR \
    trainer.device=auto \
    trainer.max_epochs=5 \
    trainer.log_every_n_steps=10000 \
    trainer.eval_every_n_steps=4000 \
    trainer.step_scheduler_on_batch_or_epoch=epoch \
    trainer.use_amp=True \
    trainer.autocast_config.enabled=True \
    trainer.autocast_config.dtype=float16 \
    trainer.scaler_config.enabled=True \
    trainer.scaler_config.init_scale=65536.0 \
    trainer.scaler_config.growth_factor=2.0 \
    trainer.scaler_config.backoff_factor=0.5 \
    trainer.scaler_config.growth_interval=2000 \
    trainer.gradient_accumulation_steps=1 \
    trainer.clip_grad_norm.max_norm=1.0 \
    trainer.clip_grad_norm.norm_type=2.0 \
    trainer.clip_grad_norm.error_if_nonfinite=False \
    trainer.clip_grad_norm.foreach=None \
    trainer.apply_weight_decay_to_different_param_groups=False \
    trainer.save_dir=./data/tinyshakespeare_char/checkpoints \
    trainer.save_every_epoch=False \
    trainer.save_best_only=True \
    trainer.monitor=train_this_epoch_average_loss \
    trainer.mode=min \
    model.context_length=256 \
    model.d_model=128 \
    model.dropout=0.1 \
    model.num_decoder_blocks=5 \
    model.decoder_block.masked_self_attention_mha.H=8 \
    model.decoder_block.masked_self_attention_mha.dropout=0.1 \
    model.decoder_block.feed_forward.dropout=0.1 \
    model.decoder_block.add_norm_1.dropout=0.1 \
    model.decoder_block.add_norm_1.dropout=0.1 \
    model.decoder_block.feed_forward.d_ff=512 \
    generator.max_tokens=1000 \
    generator.temperature=1.0 \
    generator.greedy=False \
    generator.top_k=10 > nohup.log 2>&1 &
```

```
│   optimizer=AdamWConfig(
│   │   name='torch.optim.AdamW',
│   │   lr=0.0005,
│   │   betas=(0.9, 0.999),
│   │   eps=1e-08,
│   │   weight_decay=0.01,
│   │   amsgrad=False
│   ),
│   criterion=CrossEntropyLossConfig(
│   │   name='torch.nn.CrossEntropyLoss',
│   │   weight=None,
│   │   size_average=None,
│   │   ignore_index=-100,
│   │   reduction='mean',
│   │   label_smoothing=0.0
│   ),
```

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python omnivault/transformer/projects/tinyshakespeare_char/main.py \
    omnivault/transformer/projects/tinyshakespeare_char/config.yaml \
    global_.debug=False \
    data.train_loader.batch_size=128 \
    data.train_loader.num_workers=8 \
    data.context_length=256 \
    trainer.device=auto \
    trainer.max_epochs=5 \
    trainer.log_every_n_steps=10000 \
    trainer.eval_every_n_steps=4000 \
    trainer.step_scheduler_on_batch_or_epoch=epoch \
    trainer.use_amp=True \
    trainer.autocast_config.enabled=True \
    trainer.autocast_config.dtype=float16 \
    trainer.scaler_config.enabled=True \
    trainer.scaler_config.init_scale=65536.0 \
    trainer.scaler_config.growth_factor=2.0 \
    trainer.scaler_config.backoff_factor=0.5 \
    trainer.scaler_config.growth_interval=2000 \
    trainer.gradient_accumulation_steps=1 \
    trainer.clip_grad_norm.max_norm=1.0 \
    trainer.clip_grad_norm.norm_type=2.0 \
    trainer.clip_grad_norm.error_if_nonfinite=False \
    trainer.clip_grad_norm.foreach=None \
    trainer.apply_weight_decay_to_different_param_groups=False \
    trainer.save_dir=./data/tinyshakespeare_char/checkpoints \
    trainer.save_every_epoch=False \
    trainer.save_best_only=True \
    trainer.monitor=train_this_epoch_average_loss \
    trainer.mode=min \
    model.context_length=256 \
    model.d_model=384 \
    model.dropout=0.2 \
    model.num_decoder_blocks=6 \
    model.decoder_block.masked_self_attention_mha.H=6 \
    model.decoder_block.masked_self_attention_mha.dropout=0.2 \
    model.decoder_block.feed_forward.dropout=0.2 \
    model.decoder_block.add_norm_1.dropout=0.2 \
    model.decoder_block.add_norm_1.dropout=0.2 \
    model.decoder_block.feed_forward.d_ff=1536 \
    generator.max_tokens=1000 \
    generator.temperature=1.0 \
    generator.greedy=False \
    generator.top_k=10
```
