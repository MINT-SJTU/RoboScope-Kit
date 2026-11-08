#!/usr/bin/env python3
"""Experimental adapter: use PyRUA's own SAM3 HTTP client in each Policy process."""

import argparse
import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit

EXPECTED_POLICY = "359d0ed3734bce2bc7713fe34dd55044ed5f495fc7452838c0af9422168404cd"
EXPECTED_HTTP = "93a55e38e5d51aaa302c5fb4edb5d58635bee26d77d65d6a74f93668afda29a8"
EXPECTED_LOCAL = "b33d89de366930daafb7cf789155101a4d41cb3f1f526e809989c6c0ea74629c"
HTTP_SOURCE = Path(__file__).resolve().parent / "pyrua-shared-sam3/sam3_http.py"


def install_shared_sam3(endpoint):
    parsed = urlsplit(endpoint if "://" in endpoint else "http://" + endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("SAM3 service must be a local HTTP endpoint")
    from pyrualean.hosts import sam3_local

    if hashlib.sha256(Path(sam3_local.__file__).read_bytes()).hexdigest() != EXPECTED_LOCAL:
        raise ValueError("PyRUA's installed local SAM3 implementation changed")
    if hashlib.sha256(HTTP_SOURCE.read_bytes()).hexdigest() != EXPECTED_HTTP:
        raise ValueError("the later PyRUA SAM3 HTTP implementation changed")
    module_name = "pyrualean.hosts._sam3_http_shared_experiment"
    spec = importlib.util.spec_from_file_location(module_name, HTTP_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the checked SAM3 HTTP source")
    sam3_http = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = sam3_http
    spec.loader.exec_module(sam3_http)
    # Do not let machine-wide HTTP proxy settings carry local camera images off host.
    # Replace only this isolated module's urllib binding, not urllib globally.
    direct = urllib_request.build_opener(urllib_request.ProxyHandler({}))
    sam3_http.urllib = SimpleNamespace(
        request=SimpleNamespace(Request=urllib_request.Request, urlopen=direct.open),
        error=urllib_error,
    )

    class StrictClient(sam3_http.Sam3Http):
        def segment(self, *args, **kwargs):
            result = super().segment(*args, **kwargs)
            reason = getattr(result, "reason", None)
            if isinstance(reason, str) and reason.startswith(
                ("SAM3 server error ", "SAM3 server unreachable at ")
            ):
                raise RuntimeError(reason)
            return result

    client = StrictClient(endpoint)
    if not client.alive():
        raise RuntimeError("shared SAM3 service is not ready")

    original = sam3_local.shared

    def shared(checkpoint=None, cuda_device=None):
        if checkpoint:
            raise ValueError("shared SAM3 checkpoint must be configured at the service")
        return client

    sam3_local.shared = shared
    return original, client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-endpoint", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    policy = args.policy.resolve(strict=True)
    if hashlib.sha256(policy.read_bytes()).hexdigest() != EXPECTED_POLICY:
        raise ValueError("original PyRUA Policy source differs from the evaluated version")
    install_shared_sam3(args.sam3_endpoint)
    sys.argv = [str(policy), *remaining]
    runpy.run_path(str(policy), run_name="__main__")


if __name__ == "__main__":
    main()
