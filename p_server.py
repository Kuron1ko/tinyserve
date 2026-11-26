"""P launcher: HTTP API + GPU prefill; run d_server.py on this same host/user."""
from functools import partial

import uvicorn

from pd.ipc import default_ipc_dir
from pd.standalone import PrefillRuntime
from pd_server import build_parser, config_from_args, create_app


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--ipc-dir", default=default_ipc_dir(),
                        help="Private local directory shared with d_server.py")
    args = parser.parse_args()
    cfg = config_from_args(parser, args)
    factory = partial(PrefillRuntime, ipc_dir=args.ipc_dir)
    uvicorn.run(create_app(cfg, args.trace, runtime_factory=factory), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
