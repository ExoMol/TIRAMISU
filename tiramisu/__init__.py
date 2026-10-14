import hashlib
import pathlib
import shutil

import numba

# numba's cache=True keys each kernel on its own file only, so editing a jitted helper or a module constant in another
# file leaves cached callers silently running the old code. Key the whole cache on the package source instead: any edit
# recompiles every kernel once. Set here, before any submodule decorates a kernel.
_package_dir = pathlib.Path(__file__).parent
_source_hash = hashlib.sha1(b"".join(
    path.relative_to(_package_dir).as_posix().encode() + path.read_bytes()
    for path in sorted(_package_dir.rglob("*.py"))
)).hexdigest()[:12]
_cache_root = _package_dir / "__pycache__" / "numba"
for _stale in _cache_root.glob("*"):
    if _stale.name != _source_hash:
        # ponytail: a run of an older source version still going loses its cache and recompiles what it has not yet.
        shutil.rmtree(_stale, ignore_errors=True)
numba.config.CACHE_DIR = str(_cache_root / _source_hash)
