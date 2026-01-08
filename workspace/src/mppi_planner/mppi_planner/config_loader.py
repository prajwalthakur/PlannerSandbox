#!/usr/bin/env python3
import yaml
def configLoader(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg 
