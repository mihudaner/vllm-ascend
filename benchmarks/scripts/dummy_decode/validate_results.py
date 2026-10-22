# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the MR2 performance validator."""

from validate_residual_performance import main, validate_directory  # noqa: F401

if __name__ == "__main__":
    main()
