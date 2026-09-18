# kaggle_environments (vendored, one function)

This directory exists so that `kaggriculture.py` can be imported **unmodified**.

The environment file's only dependency outside the standard library is:

```python
from kaggle_environments.utils import resolve_episode_seed
```

`resolve_episode_seed` is copied here from
[`Kaggle/kaggle-environments`](https://github.com/Kaggle/kaggle-environments)
(`kaggle_environments/utils.py`), which is Apache License 2.0. It is copied rather than
installed because installing the full `kaggle-environments` package pulls in jax,
transformers, open_spiel, pettingymnasium and pygame, and pygame has no wheel for Python
3.14 and will not compile here without SDL2 development headers.

**One function is vendored. Kaggle's environment is not edited, reimplemented or
approximated** — `env/kaggriculture.py` is a byte-for-byte copy of the official file, and
the interpreter that runs here is therefore the interpreter that runs on Kaggle.

Attribution and licence: see `../NOTICE`.
