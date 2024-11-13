# %% load modules
from typing import Optional, List

import numpy as np

#%% 

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#%% 

def check_type(
        *args,
        type_of_reference: Optional[int] = None,
        from_function: Optional[str] = 'check_type',
        flag: Optional[bool] = True) -> bool:
    """Check that all args have the same type. 
    The expected type can be given in input. Otherwise, the first input's length is taken as the reference one.

    :param length_of_reference: expected length, defaults to None
    :type length_of_reference: Optional[int]
    :return: return True in case of success
    :rtype: bool
    """
    if type_of_reference == None:
        type_of_reference = type(args[0])

    for arg_idx, arg in enumerate(args):
        if not isinstance(arg, type_of_reference):
            if flag:
                print(
                    f'{bcolors.WARNING}\n [error: {from_function}] Input parameter #{arg_idx+1} should be {type_of_reference} (not {type(arg)}).\n{bcolors.ENDC}')
            return False
    return True

def check_shape(
        *args,
        shape_of_reference: Optional[np.ndarray.shape] = None, # Optional[object] = None
        from_function: Optional[str] = 'check_shape',
        flag: Optional[bool] = True) -> bool:
    """Check that all args have the same shape. 
    The expected shape can be given in input. Otherwise, the first input's shape is taken as the reference one.

    :param shape_of_reference: expected shape, defaults to None
    :type shape_of_reference: Optional[np.ndarray.shape]
    :return: return True in case of success
    :rtype: bool
    """
    if shape_of_reference == None:
        shape_of_reference = np.shape(args[0])

    for arg_idx, arg in enumerate(args):
        dims = np.shape(arg)
        if dims != shape_of_reference:
            if flag:
                print(
                    f'{bcolors.WARNING}\n [error: {from_function}] Input parameter #{arg_idx+1} should be of shape {shape_of_reference} (not {dims}).\n{bcolors.ENDC}')
            return False
    return True

def check_length(
        *args,
        length_of_reference: Optional[int] = None,
        from_function: Optional[str] = 'check_length',
        flag: Optional[bool] = True) -> bool:
    """Check that all args have the same length. 
    The expected length can be given in input. Otherwise, the first input's length is taken as the reference one.

    :param length_of_reference: expected length, defaults to None
    :type length_of_reference: Optional[int]
    :return: return True in case of success
    :rtype: bool
    """
    if length_of_reference == None:
        length_of_reference = len(args[0])

    for arg_idx, arg in enumerate(args):
        if len(arg) != length_of_reference:
            if flag:
                print(
                    f'{bcolors.WARNING}\n [error: {from_function}] Input parameter #{arg_idx+1} should have {length_of_reference} elements (not {len(arg)}).\n{bcolors.ENDC}')
            return False
    return True


#%% 
def from_float_to_string(
        value: float,
        format: Optional[str] = 'float',
        width_precision: Optional[int] = 14,
        digit_precision: Optional[int] = 9) -> str:
    if format == 'float':
        return '{0:<{1}.{2}f}'.format(value, width_precision, digit_precision)
    if format == 'exp':
        return '{0:<{1}.{2}e}'.format(value, width_precision, digit_precision)

def from_list_to_array(
        obj: List,
        inhomogeneous: Optional[bool] = False) -> np.ndarray:
    """Change a list to np.ndarray by preserving the type of list elements.

    :param obj: list
    :type obj: list
    :return: obj as a np.ndarray object
    :rtype: np.ndarray
    """
    if inhomogeneous:
        return np.array(obj, dtype=type(obj[0]))
    else:
        return np.array(obj, dtype=np.float64)

