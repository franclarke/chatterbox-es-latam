"""Tests for the training logger module."""

import logging
import pytest

from chatterbox_es_latam.training_logger import TrainingLogger


class TestTrainingLogger:
    """Tests for TrainingLogger class."""

    def test_init_default_values(self):
        """Test default initialization."""
        logger = TrainingLogger()
        assert logger.log_every_n_batches == 100

    def test_init_custom_interval(self):
        """Test custom logging interval."""
        logger = TrainingLogger(log_every_n_batches=50)
        assert logger.log_every_n_batches == 50

    def test_init_invalid_interval_raises(self):
        """Test that invalid interval raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            TrainingLogger(log_every_n_batches=0)

        with pytest.raises(ValueError, match="at least 1"):
            TrainingLogger(log_every_n_batches=-1)

    def test_log_at_interval(self, caplog):
        """Test that logging occurs at correct intervals."""
        logger = TrainingLogger(log_every_n_batches=10)

        with caplog.at_level(logging.INFO):
            # Should log at step 0
            logger.log(step=0, loss=1.5, lr=0.001)
            assert len(caplog.records) == 1

            # Should not log at step 5
            logger.log(step=5, loss=1.4, lr=0.001)
            assert len(caplog.records) == 1

            # Should log at step 10
            logger.log(step=10, loss=1.3, lr=0.001)
            assert len(caplog.records) == 2

    def test_log_format_without_total_steps(self, caplog):
        """Test log format without total steps."""
        logger = TrainingLogger(log_every_n_batches=1)

        with caplog.at_level(logging.INFO):
            logger.log(step=5, loss=0.1234, lr=1e-4)

        assert "step 5" in caplog.text
        assert "loss: 0.1234" in caplog.text
        assert "lr: 1.00e-04" in caplog.text

    def test_log_format_with_total_steps(self, caplog):
        """Test log format with total steps."""
        logger = TrainingLogger(log_every_n_batches=1)

        with caplog.at_level(logging.INFO):
            logger.log(step=50, loss=0.5, lr=2e-5, total_steps=100)

        assert "[50/100]" in caplog.text
        assert "loss: 0.5000" in caplog.text
        assert "lr: 2.00e-05" in caplog.text

    def test_log_every_n_batches_setter(self):
        """Test setting log_every_n_batches property."""
        logger = TrainingLogger(log_every_n_batches=100)
        logger.log_every_n_batches = 50
        assert logger.log_every_n_batches == 50

    def test_log_every_n_batches_setter_invalid_raises(self):
        """Test that invalid setter value raises ValueError."""
        logger = TrainingLogger()
        with pytest.raises(ValueError, match="at least 1"):
            logger.log_every_n_batches = 0

    def test_custom_logger(self, caplog):
        """Test with custom logger."""
        custom_log = logging.getLogger("custom_test")
        logger = TrainingLogger(log_every_n_batches=1, logger=custom_log)

        with caplog.at_level(logging.INFO, logger="custom_test"):
            logger.log(step=0, loss=1.0, lr=0.01)

        assert "loss: 1.0000" in caplog.text
