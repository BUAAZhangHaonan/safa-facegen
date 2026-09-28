"""PRDC from individual EXISTING Inception features; not from mean/covariance.
No feature extractor/weights or images are downloaded. k=5, original coordinates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist


def compute_prdc(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict:
    real, fake = np.asarray(real), np.asarray(fake)
    if real.ndim!=2 or fake.ndim!=2 or real.shape[1]!=fake.shape[1]:
        raise ValueError("Expected matching individual feature matrices [N,D]")
    if not 1<=k<min(len(real),len(fake)) or not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise ValueError("Invalid k or features")
    if max(len(real),len(fake))>2048:
        raise ValueError("This helper is bounded to 2048 rows; use the registered 1024 protocol")
    rr = cdist(real,real,metric="sqeuclidean")
    ff = cdist(fake,fake,metric="sqeuclidean")
    rf = cdist(real,fake,metric="sqeuclidean")
    r_radius = np.partition(rr,k,axis=1)[:,k]
    f_radius = np.partition(ff,k,axis=1)[:,k]
    return {"precision":float((rf<r_radius[:,None]).any(axis=0).mean()),
            "recall":float((rf<f_radius[None,:]).any(axis=1).mean()),
            "density":float((rf<r_radius[:,None]).sum(axis=0).mean()/k),
            "coverage":float((rf.min(axis=1)<r_radius).mean()),
            "k":k,"real_count":len(real),"fake_count":len(fake),"feature_dim":real.shape[1],
            "not_identity_or_privacy_certification":True}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real",type=Path,required=True);parser.add_argument("--fake",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(compute_prdc(np.load(args.real,allow_pickle=False),np.load(args.fake,allow_pickle=False)),indent=2))
