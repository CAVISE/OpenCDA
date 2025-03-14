# -*- coding: utf-8 -*-
"""
Script to run different scenarios.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modified by CAVISE team.

import os
import sys
import errno
import pathlib
import logging
import argparse
import omegaconf
import importlib

from opencda.version import __version__

try:
    coloredlogs = importlib.import_module('coloredlogs')
except ModuleNotFoundError:
    coloredlogs = None
    print('could not find coloredlogs module! Your life will look pale.')
    print('if you are interested in improving it: https://pypi.org/project/coloredlogs')


# Handle cavise log creation, obtain this logger later with a call to
# logging.getLogger('cavise'). Use for our (cavise) code only.
def create_logger(level: int, fmt: str = '- [%(asctime)s][%(name)s] %(message)s', datefmt: str = '%H:%M:%S') -> logging.Logger:
    logger = logging.getLogger('cavise')
    if coloredlogs is not None:
        coloredlogs.install(level=level, logger=logger, fmt=fmt, datefmt=datefmt)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handler.setLevel(level)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


# Parse command line args.
def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    # opencda basic args
    parser.add_argument('-t', '--test-scenario', required=True, type=str,
                        help='Define the name of the scenario you want to test. The given name must'
                             'match one of the testing scripts(e.g. single_2lanefree_carla) in '
                             'opencda/scenario_testing/ folder'
                             ' as well as the corresponding yaml file in opencda/scenario_testing/config_yaml.')
    parser.add_argument('--record', action='store_true', help='whether to record and save the simulation process to .log file')
    # NOTICE: temporary disabled until we update yolo models.
    # parser.add_argument("--apply-ml", action='store_true',
    #                     help='whether ml/dl framework such as sklearn/pytorch is needed in the testing. '
    #                          'Set it to true only when you have installed the pytorch/sklearn package.')
    parser.add_argument('-v', "--version", type=str, default='0.9.15', help='Specify the CARLA simulator version.')
    parser.add_argument('--free-spectator', action='store_true', help='Enable free movement for the spectator camera.')
    parser.add_argument('-x', '--xodr', action='store_true', help='Run simulation using a custom map from an XODR file.')
    parser.add_argument('-c', '--cosim', action='store_true', help='Enable co-simulation with SUMO.')
    parser.add_argument('--with-capi', action='store_true', help='wether to run a communication manager instance in this simulation.')

    # Coperception models parameters
    parser.add_argument('--with-coperception', action='store_true', help='Whether to enable the use of cooperative perception models in this simulation.')
    parser.add_argument('--model-dir', type=str, help='Continued training path')
    parser.add_argument('--fusion-method', type=str, default='late', help='late, early or intermediate')
    parser.add_argument('--show-vis', action='store_true', help='whether to show image visualization result')
    parser.add_argument('--show-sequence', action='store_true', help='whether to show video visualization result. It can note be set true with show_vis together.')
    parser.add_argument('--save-vis', action='store_true', help='whether to save visualization result')
    parser.add_argument('--save-npy', action='store_true', help='whether to save prediction and gt result in npy_test file')
    parser.add_argument('--save-vis-n', type=int, default=10, help='save how many numbers of visualization result?')
    parser.add_argument('--global-sort-detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    return parser.parse_args()

# TODO: python setup.py develop для OpenCOOD на проверку присутствия файлов cython
def main() -> None:
    opt = arg_parse()
    logger = create_logger(logging.DEBUG)
    logger.info(f'OpenCDA Version: {__version__}')

    cwd = pathlib.PurePath(os.getcwd())
    default_yaml = config_yaml = cwd.joinpath('opencda/scenario_testing/config_yaml/default.yaml')
    config_yaml = cwd.joinpath(f'opencda/scenario_testing/config_yaml/{opt.test_scenario}.yaml')
    if not os.path.isfile(config_yaml):
        logger.error(f'opencda/scenario_testing/config_yaml/{opt.test_scenario}.yaml not found!')
        sys.exit(errno.EPERM)

    # allow opencood imports
    sys.path.append(str(cwd.joinpath('OpenCOOD')))

    # set the yaml file for the specific testing scenario
    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = omegaconf.OmegaConf.load(str(default_yaml))
    scene_dict = omegaconf.OmegaConf.load(str(config_yaml))
    scene_dict = omegaconf.OmegaConf.merge(default_dict, scene_dict)
    opt.apply_ml = False
    testing_scenario = importlib.import_module('opencda.scenario_testing.scenario')

    scenario_runner = getattr(testing_scenario, 'run_scenario')
    if scenario_runner is None:
        logger.error('Failed to get \'run_scenario\' method from scenario module. Ensure it exists and follows specified template')
        sys.exit(errno.EPERM)
    scenario_runner(opt, scene_dict)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
