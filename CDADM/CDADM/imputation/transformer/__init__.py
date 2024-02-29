"""
The package of the partially-observed time-series imputation model Transformer.

Refer to the paper "Du, W., Cote, D., & Liu, Y. (2023). SAITS: Self-Attention-based Imputation for Time Series.
Expert systems with applications."

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


from .model import Transformer

__all__ = [
    "Transformer",
]
