from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location("heima_snapshot",Path(__file__).parent/"snapshot_files.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
snapshot=module.snapshot
