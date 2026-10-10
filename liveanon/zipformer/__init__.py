"""Streaming Zipformer encoder from icefall (Apache-2.0).

https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer
k2 / icefall dependencies are replaced by `_compat`.
"""
from .zipformer import Zipformer2  # noqa: F401
from .subsampling import Conv2dSubsampling  # noqa: F401
