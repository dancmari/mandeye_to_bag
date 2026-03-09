#!/usr/bin/env python3
"""
mandeye_check_deps.py — Validate the Python environment for the mandeye toolset.

Checks:
  • Python version (>= 3.8)
  • Required packages: numpy, rosbags, laspy
  • Optional packages: Pillow (image export in mandeye_bag_extract.py)
  • Presence of companion scripts in the same directory
  • Functional smoke-tests (ROS1/ROS2 reader/writer round-trip imports)

Usage:
  python mandeye_check_deps.py
  python mandeye_check_deps.py --install-missing   # auto-install with pip
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OK   = "[OK]  "
_WARN = "[WARN]"
_FAIL = "[FAIL]"
_INFO = "[INFO]"

_PASS_COLOR  = "\033[32m"
_WARN_COLOR  = "\033[33m"
_FAIL_COLOR  = "\033[31m"
_INFO_COLOR  = "\033[36m"
_RESET_COLOR = "\033[0m"


def _enable_ansi_windows() -> bool:
    """Try to enable VT100/ANSI processing on Windows console. Returns True on success."""
    try:
        import ctypes
        import ctypes.wintypes
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.wintypes.DWORD()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return bool(kernel32.SetConsoleMode(
                handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            ))
    except Exception:
        pass
    return False


if sys.platform == "win32":
    _use_color = sys.stdout.isatty() and _enable_ansi_windows()
else:
    _use_color = sys.stdout.isatty()


def _colored(prefix: str, msg: str) -> str:
    if not _use_color:
        return f"{prefix} {msg}"
    if prefix.startswith("[OK"):
        return f"{_PASS_COLOR}{prefix}{_RESET_COLOR} {msg}"
    if prefix.startswith("[WARN"):
        return f"{_WARN_COLOR}{prefix}{_RESET_COLOR} {msg}"
    if prefix.startswith("[FAIL"):
        return f"{_FAIL_COLOR}{prefix}{_RESET_COLOR} {msg}"
    return f"{_INFO_COLOR}{prefix}{_RESET_COLOR} {msg}"


def ok(msg: str)   -> None: print(_colored(_OK,   msg))
def warn(msg: str) -> None: print(_colored(_WARN, msg))
def fail(msg: str) -> None: print(_colored(_FAIL, msg))
def info(msg: str) -> None: print(_colored(_INFO, msg))


def _pip_install(package: str) -> bool:
    """Run pip install for *package*.  Returns True on success."""
    print(f"  Running: pip install {package}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        ok(f"Installed: {package}")
        return True
    else:
        fail(f"pip install failed for '{package}':\n{result.stderr.strip()}")
        return False


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_python_version() -> bool:
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 8):
        fail(f"Python {ver_str} — requires >= 3.8")
        return False
    ok(f"Python {ver_str}")
    return True


def check_package(
    import_name: str,
    pip_name: str,
    version_attr: str = "__version__",
    required: bool = True,
    install_if_missing: bool = False,
) -> bool:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, version_attr, "n/a")
        ok(f"{pip_name} {version}")
        return True
    except ImportError:
        level = fail if required else warn
        label = "required" if required else "optional"
        level(f"{pip_name} not found ({label})")
        if install_if_missing:
            return _pip_install(pip_name)
        if required:
            info(f"  Install with:  pip install \"{pip_name}\"")
        return not required   # optional missing → still OK overall


def check_rosbags_internals() -> bool:
    """Verify the specific sub-modules used by mandeye_bag_common."""
    modules = [
        ("rosbags.rosbag1",         "Reader"),
        ("rosbags.rosbag2",         "Reader"),
        ("rosbags.rosbag1",         "Writer"),
        ("rosbags.rosbag2",         "Writer"),
        ("rosbags.typesys",         "Stores"),
    ]
    all_ok = True
    for mod_path, attr in modules:
        try:
            mod = importlib.import_module(mod_path)
            getattr(mod, attr)
        except (ImportError, AttributeError) as exc:
            fail(f"rosbags internal: {mod_path}.{attr} — {exc}")
            all_ok = False
    if all_ok:
        ok("rosbags internals  (rosbag1/rosbag2/typesys)")
    return all_ok


def check_laspy_lazrs() -> bool:
    """Check that laspy is present and a LAZ compression backend is available."""
    try:
        import laspy
    except ImportError:
        fail("laspy not found (required)")
        info("  Install with:  pip install \"laspy[lazrs]\"")
        return False

    # Detect available LAZ backends — try each known package separately.
    backends = []
    try:
        import lazrs  # noqa: F401
        backends.append("lazrs")
    except ImportError:
        pass
    try:
        import laszip  # noqa: F401
        backends.append("laszip")
    except ImportError:
        pass

    # Fallback: laspy 2.x may expose backend info via LasData / compression
    if not backends:
        try:
            from laspy.compression import LazrsPayloadCompressor  # noqa: F401
            backends.append("lazrs (internal)")
        except ImportError:
            pass

    if backends:
        ok(f"laspy {laspy.__version__}  (LAZ backends: {', '.join(backends)})")
        return True

    warn(f"laspy {laspy.__version__} — no LAZ backend found.")
    info("  Install lazrs with:  pip install \"laspy[lazrs]\"")
    info("  Note: reading/writing .laz files will fail without a backend.")
    return False


def check_companion_scripts() -> bool:
    here = Path(sys.argv[0]).resolve().parent
    scripts = [
        "mandeye_bag_common.py",
        "mandeye_bag_audit.py",
        "mandeye_bag_convert.py",
        "mandeye_bag_extract.py",
        "mandeye_imu_rescale.py",
    ]
    all_present = True
    for name in scripts:
        p = here / name
        if p.exists():
            ok(f"Script present: {name}")
        else:
            warn(f"Script missing:  {name}  (expected alongside this file)")
            all_present = False
    return all_present


def check_mandeye_common_importable() -> bool:
    """Try importing mandeye_bag_common — catches any syntax/import errors there."""
    try:
        import mandeye_bag_common  # noqa: F401
        ok("mandeye_bag_common imports cleanly")
        return True
    except ImportError as exc:
        fail(f"mandeye_bag_common import failed: {exc}")
        info("  Make sure all required packages are installed first.")
        return False
    except Exception as exc:
        fail(f"mandeye_bag_common error: {type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Python environment for the mandeye toolset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Without --install-missing this script is read-only — it only inspects,
never modifies, the environment.

Required packages:
  pip install numpy rosbags "laspy[lazrs]"

Optional packages:
  pip install Pillow          # image export in mandeye_bag_extract.py
""",
    )
    parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Automatically pip-install any missing required packages",
    )
    args = parser.parse_args()

    install = args.install_missing
    failures = 0

    print()
    print("=" * 58)
    print("  mandeye toolset — dependency check")
    print("=" * 58)

    # --- Python version ---
    print("\n-- Runtime -----------------------------------------------")
    if not check_python_version():
        failures += 1

    # --- Required packages ---
    print("\n-- Required packages --------------------------------------")
    for import_name, pip_name in [
        ("numpy",   "numpy"),
        ("rosbags", "rosbags"),
    ]:
        if not check_package(import_name, pip_name,
                             install_if_missing=install):
            failures += 1

    if not check_laspy_lazrs():
        if install:
            _pip_install("laspy[lazrs]")
        failures += 1

    # --- rosbags sub-modules ---
    print("\n-- rosbags internals --------------------------------------")
    if not check_rosbags_internals():
        failures += 1

    # --- Optional packages ---
    print("\n-- Optional packages --------------------------------------")
    check_package(
        "PIL", "Pillow",
        version_attr="__version__",
        required=False,
        install_if_missing=False,
    )
    if not any(
        importlib.util.find_spec(m) for m in ("PIL",)
        if importlib.util.find_spec(m) is not None
    ):
        info("  Pillow is only needed for image export (mandeye_bag_extract.py).")

    # --- Companion scripts ---
    print("\n-- Companion scripts --------------------------------------")
    check_companion_scripts()

    # --- Import smoke-test ---
    print("\n-- Import smoke-test --------------------------------------")
    if failures == 0:
        check_mandeye_common_importable()
    else:
        warn("Skipping import smoke-test — fix required packages first.")

    # --- Summary ---
    print()
    print("=" * 58)
    if failures == 0:
        ok("All required dependencies satisfied.")
        info("You can now use the mandeye toolset.")
    else:
        fail(f"{failures} required check(s) failed.")
        info("Run with --install-missing to auto-install missing packages:")
        info("  python mandeye_check_deps.py --install-missing")
    print()


if __name__ == "__main__":
    main()
