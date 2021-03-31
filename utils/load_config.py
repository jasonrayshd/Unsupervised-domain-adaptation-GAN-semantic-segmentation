#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import yaml
import re
import logging.config


def init_logging(log_path, config_path='config/logging_config.yaml'):
    """
    initial logging module with config
    :param config_path:
    :return:
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
            # print(config)
            for i, handler in enumerate(config["handlers"].keys()):
                # print(handler)
                if "error" in handler:
                    config["handlers"][handler]["filename"] = f"{log_path}/errors.log"
                elif "handler" in handler:
                    config["handlers"][handler]["filename"] = f"{log_path}/debug.log"
        # print(config)
        logging.config.dictConfig(config)

    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        logging.basicConfig(level=logging.DEBUG)

def read_config(config_path='config/global_config.yaml'):
    """
    store the global parameters in the project
    :param config_path:
    :return:
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config

    except IOError:
        sys.stderr.write('logging config file "%s" not found' % config_path)
        exit(-1)
