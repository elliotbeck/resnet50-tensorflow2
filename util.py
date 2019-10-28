import csv
import json
from collections import namedtuple
from shutil import make_archive
from datetime import datetime
import os
import tensorflow as tf

import local_settings

def copy_source(code_directory, model_dir):
    now = datetime.now().strftime('%Y-%m-%d')
    make_archive(os.path.join(model_dir, "code_%s.tar.gz" % now), 'tar', code_directory)



def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value)
        return namedtuple('GenericDict', obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else:
        return obj


def get_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)

    return config


def update_config(config, args):
    for entry in config:
        if hasattr(args, entry):
            if eval("args.{}".format(entry)) is not None:
                config[entry] = eval("args.{}".format(entry))
    return config

