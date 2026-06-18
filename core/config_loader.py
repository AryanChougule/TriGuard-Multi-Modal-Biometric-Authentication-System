"""
core/config_loader.py
Loads config.json, validates it, and handles dynamic weight redistribution
for disabled modules.
"""

import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

REQUIRED_MODULES = ["face", "voice", "lip"]


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load and validate config.json. Returns the full config dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    _validate_config(config)
    config = _redistribute_weights(config)

    logger.info("Config loaded successfully.")
    logger.info(f"Active modules: {_active_modules(config)}")
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Basic structural validation of the config."""
    assert "system" in config, "Missing 'system' section in config."
    assert "modules" in config, "Missing 'modules' section in config."
    assert "scoring" in config, "Missing 'scoring' section in config."

    for mod in REQUIRED_MODULES:
        assert mod in config["modules"], f"Missing module '{mod}' in config."
        assert "enabled" in config["modules"][mod], f"Module '{mod}' missing 'enabled' field."
        assert "weight" in config["modules"][mod], f"Module '{mod}' missing 'weight' field."

    active = _active_modules(config)
    if len(active) == 0:
        raise ValueError("All modules are disabled. Enable at least one module in config.json.")

    raw_total = sum(config["modules"][m]["weight"] for m in REQUIRED_MODULES)
    if not (0.99 <= raw_total <= 1.01):
        raise ValueError(
            f"Module weights must sum to 1.0. Current sum: {raw_total:.2f}. "
            "Please fix the weights in config.json."
        )


def _active_modules(config: Dict[str, Any]) -> list:
    return [m for m in REQUIRED_MODULES if config["modules"][m]["enabled"]]


def _redistribute_weights(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redistribute weights of disabled modules equally among active ones.
    Modifies and returns the config in-place with effective_weight added per module.
    """
    active = _active_modules(config)
    disabled = [m for m in REQUIRED_MODULES if not config["modules"][m]["enabled"]]

    if not disabled:
        # All modules active — effective weight == configured weight
        for mod in REQUIRED_MODULES:
            config["modules"][mod]["effective_weight"] = config["modules"][mod]["weight"]
        return config

    # Total weight to redistribute from disabled modules
    disabled_weight_total = sum(config["modules"][m]["weight"] for m in disabled)
    bonus_per_active = disabled_weight_total / len(active)

    for mod in REQUIRED_MODULES:
        if config["modules"][mod]["enabled"]:
            effective = config["modules"][mod]["weight"] + bonus_per_active
            config["modules"][mod]["effective_weight"] = round(effective, 6)
        else:
            config["modules"][mod]["effective_weight"] = 0.0

    effective_total = sum(config["modules"][m]["effective_weight"] for m in active)
    logger.info(
        f"Weight redistribution applied. Disabled: {disabled}. "
        f"Effective weights: { {m: config['modules'][m]['effective_weight'] for m in active} }. "
        f"Effective total: {effective_total:.4f}"
    )

    return config


def get_module_config(config: Dict[str, Any], module_name: str) -> Dict[str, Any]:
    """Convenience accessor for a specific module's config block."""
    return config["modules"][module_name]


def get_system_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience accessor for the system config block."""
    return config["system"]
