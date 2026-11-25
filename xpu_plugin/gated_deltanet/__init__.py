"""
Copyright (c) 2026 Baidu.com, Inc. All Rights Reserved
This module provides GatedDeltaNet XPU mock.
"""

from .gated_deltanet import MockGatedDeltaNet, _install_gdn_mock_hook

__all__ = ["MockGatedDeltaNet", "_install_gdn_mock_hook"]
