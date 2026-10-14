import ast
import hashlib
import pathlib
import shutil

import numba

# numba's cache=True keys each kernel on its own file only, so editing a jitted helper or a module constant in another
# file leaves cached callers silently running the old code. Key the whole cache on the package's code instead: any code
# edit recompiles every kernel once. The hash is of each module's syntax tree without docstrings, so comments,
# docstrings and line numbers don't count; editing those in a file still recompiles that file's kernels (numba's own
# per-file check). Set here, before any submodule decorates a kernel.
_package_dir = pathlib.Path(__file__).parent
_digest = hashlib.sha1()
for _path in sorted(_package_dir.rglob("*.py")):
    _tree = ast.parse(_path.read_bytes())
    for _node in ast.walk(_tree):
        _body = getattr(_node, "body", None)
        if (isinstance(_node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and _body
                and isinstance(_body[0], ast.Expr) and isinstance(_body[0].value, ast.Constant)
                and isinstance(_body[0].value.value, str)):
            _node.body = _body[1:]
    _digest.update(_path.relative_to(_package_dir).as_posix().encode() + ast.dump(_tree).encode())
_source_hash = _digest.hexdigest()[:12]
_cache_root = _package_dir / "__pycache__" / "numba"
for _stale in _cache_root.glob("*"):
    if _stale.name != _source_hash:
        # ponytail: a run of an older source version still going loses its cache and recompiles what it has not yet.
        shutil.rmtree(_stale, ignore_errors=True)
numba.config.CACHE_DIR = str(_cache_root / _source_hash)
