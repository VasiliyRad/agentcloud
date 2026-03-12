"""
Root conftest: ensure src/ is first on sys.path so local modules
(ag2, crew, chat, ...) take precedence over any installed packages
with the same name.
"""
import sys
import os

src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# If the installed 'ag2' (AutoGen) package was already cached in sys.modules
# before our src/ag2 could be found, evict it so the local module wins.
for key in list(sys.modules.keys()):
    if key == "ag2" or key.startswith("ag2."):
        mod = sys.modules[key]
        if hasattr(mod, "__file__") and mod.__file__ and src_path not in mod.__file__:
            del sys.modules[key]
