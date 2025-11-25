"""Simple training logger for displaying loss, LR, and steps.

This module provides non-verbose logging functionality for training runs,
displaying metrics at configurable intervals without generating too much noise.
"""

import logging
from typing import Optional


class TrainingLogger:
    """Simple logger for training metrics (loss, LR, steps).

    Logs training progress at specified intervals to avoid excessive output.

    Example:
        >>> logger = TrainingLogger(log_every_n_batches=100)
        >>> for step, batch in enumerate(dataloader):
        ...     loss = train_step(batch)
        ...     lr = scheduler.get_last_lr()[0]
        ...     logger.log(step=step, loss=loss, lr=lr)
    """

    def __init__(
        self,
        log_every_n_batches: int = 100,
        logger: Optional[logging.Logger] = None,
        name: str = "training",
    ) -> None:
        """Initialize the training logger.

        Args:
            log_every_n_batches: Log metrics every N batches. Default is 100.
            logger: Optional custom logger. If None, creates a default logger.
            name: Name for the logger. Default is "training".
        """
        if log_every_n_batches < 1:
            raise ValueError("log_every_n_batches must be at least 1")

        self._log_every_n_batches = log_every_n_batches
        self._logger = logger or self._create_default_logger(name)

    @staticmethod
    def _create_default_logger(name: str) -> logging.Logger:
        """Create a default logger with simple formatting."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def log(
        self,
        step: int,
        loss: float,
        lr: float,
        total_steps: Optional[int] = None,
    ) -> None:
        """Log training metrics if at a logging interval.

        Args:
            step: Current training step/batch number (0-indexed).
            loss: Current loss value.
            lr: Current learning rate.
            total_steps: Optional total number of steps for progress display.
        """
        # Log at step 0 and then every N batches
        if step % self._log_every_n_batches != 0:
            return

        if total_steps is not None:
            progress = f"[{step}/{total_steps}]"
        else:
            progress = f"step {step}"

        self._logger.info(
            "%s | loss: %.4f | lr: %.2e",
            progress,
            loss,
            lr,
        )

    @property
    def log_every_n_batches(self) -> int:
        """Return the logging interval."""
        return self._log_every_n_batches

    @log_every_n_batches.setter
    def log_every_n_batches(self, value: int) -> None:
        """Set the logging interval."""
        if value < 1:
            raise ValueError("log_every_n_batches must be at least 1")
        self._log_every_n_batches = value
