"""D launcher: receive P's config and KV, decode on its assigned GPU; no HTTP port."""
import argparse

from pd.ipc import default_ipc_dir
from pd.standalone import install_decode_signal_handler, run_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc-dir", default=default_ipc_dir(),
                        help="Same private local directory as p_server.py")
    parser.add_argument("--startup-timeout-s", type=float, default=300,
                        help="Maximum time waiting for P's IPC endpoint")
    args = parser.parse_args()
    if args.startup_timeout_s <= 0:
        parser.error("--startup-timeout-s must be positive")
    install_decode_signal_handler()
    try:
        return run_decode(args.ipc_dir, args.startup_timeout_s)
    except KeyboardInterrupt:
        return 0
    except (TimeoutError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"D startup failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
