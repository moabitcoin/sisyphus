import sys
import json
import base64

import requests
from einops import rearrange

from sfi.io import ArrayIO


def main(args):
    query = ArrayIO.load(args.query)

    if len(query.shape) == 1:  # handle (C,) as (1, C)
        query = rearrange(query, "n -> () n")

    N, C = query.shape
    dtype = str(query.dtype)
    feature = base64.b64encode(query.ravel()).decode("utf-8")

    url = "http://{}:{}".format(args.host, args.port)

    payload = {"num_results": args.num_results,
               "feature": feature,
               "shape": [N, C],
               "dtype": dtype}

    res = requests.post(url, data=json.dumps(payload))

    if res.status_code != requests.codes.ok:
        sys.exit("Error: unable to query server")

    print(json.dumps(res.json()))
