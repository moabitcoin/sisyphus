import sys
import base64
import binascii

import numpy as np
from einops import rearrange

from flask import Flask, request, jsonify, abort

from sfi.index import Index, IndexQueryError


app = Flask(__name__)
index = None


@app.route("/", methods=["POST"])
def query():
    if not index:
        return abort(503)

    req = request.get_json(force=True, silent=False, cache=False)

    if not all(v in req for v in ["feature", "shape", "dtype"]):
        return abort(400)

    try:
        feature = base64.b64decode(req["feature"])
    except binascii.Error:
        return abort(400)

    N, C = req["shape"]
    dtype = req["dtype"]

    try:
        vs = np.frombuffer(feature, dtype=dtype)
        vs = rearrange(vs, "(n c) -> n c", n=N, c=C)
    except ValueError:
        return abort(400)

    num_results = req.get("num_results", 1)

    try:
        results = index.query(vs, num_results=num_results)
    except IndexQueryError:
        return abort(400)

    return jsonify([{"distance": d, "path": p} for d, p in results])


def main(args):
    print("Loading index from disk", file=sys.stderr)

    global index
    index = Index(path=args.index, metadata=args.index.with_suffix(".json"),
                  features_size=args.features_size, num_probes=args.num_probes)

    app.run(host=args.host, port=args.port, debug=False)
