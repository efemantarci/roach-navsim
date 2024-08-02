from importlib import import_module
import json
from pathlib import Path
import socket
import xml.etree.ElementTree as ET
import numpy as np
import hydra

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def load_obs_configs(agent_configs_dict):
    obs_configs = {}
    for actor_id, cfg in agent_configs_dict.items():
        obs_configs[actor_id] = json.load(open(cfg['path_to_conf_file'], 'r'))['obs_configs']
    return obs_configs


def init_agents(agent_configs_dict, **kwargs):
    agents_dict = {}
    for actor_id, cfg in agent_configs_dict.items():
        AgentClass = load_entry_point(cfg['entry_point'])
        agents_dict[actor_id] = AgentClass(cfg['path_to_conf_file'], **kwargs)
    return agents_dict

def to_camel_case(snake_str, init_capital=False):
    # agent_class_str = to_camel_case(agent_module_str.split('.')[-1], init_capital=True)
    components = snake_str.split('_')
    if init_capital:
        init_letter = components[0].title()
    else:
        init_letter = components[0]
    return init_letter + ''.join(x.title() for x in components[1:])


def get_free_tcp_port():
    s = socket.socket()
    s.bind(("", 0))  # Request the sys to provide a free port dynamically
    server_port = s.getsockname()[1]
    s.close()
    # 2000 works fine for now
    server_port = 2000
    return server_port
