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
