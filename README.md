# chatterbox-es-latam

Text-to-Speech for Latin American Spanish.

## Training Logger

Simple, non-verbose logging for training runs. Shows loss, learning rate, and steps at configurable intervals.

### Usage

```python
from chatterbox_es_latam import TrainingLogger

# Log every 100 batches (default)
logger = TrainingLogger(log_every_n_batches=100)

for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    lr = scheduler.get_last_lr()[0]
    logger.log(step=step, loss=loss, lr=lr)
```

### Output Example

```
15:02:08 | step 0 | loss: 2.0000 | lr: 1.00e-04
15:02:08 | step 100 | loss: 1.5000 | lr: 5.99e-05
15:02:08 | step 200 | loss: 1.0000 | lr: 3.58e-05
```

With `total_steps` for progress display:

```python
logger.log(step=50, loss=0.5, lr=1e-4, total_steps=100)
# Output: 15:02:08 | [50/100] | loss: 0.5000 | lr: 1.00e-04
```