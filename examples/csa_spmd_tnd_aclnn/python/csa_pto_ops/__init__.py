# SPDX-License-Identifier: Apache-2.0
from .bootstrap import ensure_vendor

ensure_vendor()

from .api import PreparedAclnnCSA

__version__ = "0.1.0"
__all__ = ["PreparedAclnnCSA", "ensure_vendor"]
