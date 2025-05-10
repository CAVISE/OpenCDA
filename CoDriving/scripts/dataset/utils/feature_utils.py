#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

import numpy as np

def get_intention_from_vehicle_id(vehicle_id: str)-> np.ndarray:
    """
    Parse the vehicle id to distinguish its intention.
    """
    intention = np.zeros(4)

    from_path, to_path, _ = vehicle_id.split('_')
    if from_path == 'left':
        if to_path == 'right':
            intention_str = 'straight'
        elif to_path == 'up':
            intention_str = 'left'
        elif to_path == 'down':
            intention_str = 'right'

    elif from_path == 'right':
        if to_path == 'left':
            intention_str = 'straight'
        elif to_path == 'up':
            intention_str = 'right'
        elif to_path == 'down':
            intention_str = 'left'

    elif from_path == 'up':
        if to_path == 'down':
            intention_str = 'straight'
        elif to_path == 'left':
            intention_str = 'right'
        elif to_path == 'right':
            intention_str = 'left'

    elif from_path == 'down':
        if to_path == 'up':
            intention_str = 'straight'
        elif to_path == 'right':
            intention_str = 'right'
        elif to_path == 'left':
            intention_str = 'left'

    else:
        raise Exception('Wrong vehicle id')

    if intention_str == 'left':
        intention[0] = 1
    elif intention_str == 'straight':
        intention[1] = 1
    elif intention_str == 'right':
        intention[2] = 1
    
    return intention
