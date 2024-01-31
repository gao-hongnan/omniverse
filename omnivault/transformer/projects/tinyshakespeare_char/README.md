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

```bash
python omnivault/transformer/projects/tinyshakespeare_char/main.py \
    omnivault/transformer/projects/tinyshakespeare_char/config.yaml \
    global_.debug=False \
    data.train_loader.batch_size=128 \
    data.train_loader.num_workers=8 \
    trainer.max_epochs=5 \
    trainer.eval_every_n_steps=4000 \
    trainer.log_every_n_steps=10000 \
    data.context_length=256 \
    trainer.use_amp=True \
    trainer.autocast_config.enabled=True \
    trainer.autocast_config.dtype=float16 \
    trainer.scaler_config.enabled=True \
    generator.max_tokens=100
```

```bash
python omnivault/transformer/projects/tinyshakespeare_char/main.py \
    omnivault/transformer/projects/tinyshakespeare_char/config.yaml \
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
    model.d_model=128 \
    model.context_length=256 \
    model.num_decoder_blocks=5
```
