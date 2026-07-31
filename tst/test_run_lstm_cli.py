"""Regression test for a real bug found in review: src/training/run_lstm.py
called set_global_seed(42) hard-coded, silently ignoring the parsed --seed
CLI argument (default 1337). This broke reproducibility end-to-end: the
seed logged in lstm_test_report.json / MLflow never matched the seed that
was actually used for training.
"""

from __future__ import annotations

import sys

import pytest

import src.training.run_lstm as run_lstm_mod


class _StopEarly(Exception):
    pass


def _run_main_capturing_seed(monkeypatch, argv):
    captured = {}

    def fake_set_global_seed(seed):
        captured["seed"] = seed

    def fake_prepare_dataset(cfg, seq_len):
        raise _StopEarly()

    monkeypatch.setattr(run_lstm_mod, "set_global_seed", fake_set_global_seed)
    monkeypatch.setattr(run_lstm_mod, "prepare_dataset", fake_prepare_dataset)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(_StopEarly):
        run_lstm_mod.main()
    return captured


def test_run_lstm_uses_the_cli_seed(monkeypatch):
    captured = _run_main_capturing_seed(monkeypatch, ["run_lstm.py", "--seed", "777"])
    assert captured["seed"] == 777


def test_run_lstm_falls_back_to_documented_default_seed(monkeypatch):
    captured = _run_main_capturing_seed(monkeypatch, ["run_lstm.py"])
    assert captured["seed"] == 1337
