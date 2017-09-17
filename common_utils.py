import functools
import warnings
from argparse import ArgumentParser
from optparse import OptionParser

import os
import time


def set_proc_name(newname):
    from ctypes import cdll, byref, create_string_buffer
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(len(newname)+1)
    buff.value = newname.encode("utf-8")
    libc.prctl(15, byref(buff), 0, 0, 0)


def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as err:
        if err.errno!=17:
            raise


def parse_dict(parser, dic, prefix=()):
    option_cmd = list(prefix)
    for k, v in dic.items():
        assert isinstance(k, str)
        if v is True:
            option_cmd.append("--" + k)
        elif v is False:
            continue
        else:
            option_cmd.append("--" + k)
            if isinstance(v, list):
                if isinstance(parser, OptionParser):
                    option_cmd.append(",".join(str(i) for i in v))
                else:
                    assert isinstance(parser, ArgumentParser)
                    option_cmd.extend(v)
            else:
                option_cmd.append(str(v))

    return parser.parse_args(option_cmd)



def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning) #turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning) #reset filter
        return func(*args, **kwargs)

    return new_func


def under_construction(func):
    """This is a decorator which can be used to mark functions
    as under construction. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn("Call to under construction function {}.".format(func.__name__), category=UserWarning, stacklevel=2)
        return func(*args, **kwargs)

    return new_func


def add_common_arguments(parser):
    parser.add_argument("--dynet-seed", type=int, dest="seed", default=0)
    parser.add_argument("--dynet-mem", type=int, dest="mem", default=0)
    parser.add_argument("--dynet-l2", type=float, dest="l2", default=0.0)
    parser.add_argument("--dynet-weight-decay", type=float, dest="weight_decay", default=0.0)


def add_train_arguments(parser):
    parser.add_argument("--title", type=str, dest="title", default="default")
    parser.add_argument("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", required=True)
    parser.add_argument("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", nargs="+", required=True)
    parser.add_argument("--outdir", type=str, dest="output", required=True)
    parser.add_argument("--max-save", type=int, dest="max_save", default=2)
    parser.add_argument("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model.")


def add_predict_arguments(parser):
    parser.add_argument("--output", type=str, dest="out_file", required=True)
    parser.add_argument("--model", dest="model", help="Load/Save model file", metavar="FILE", required=True)
    parser.add_argument("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", required=True)


def add_train_and_predict_arguments(parser):
    parser.add_argument("--output-scores", action="store_true", dest="output_scores", default=False)

class Timer(object):
    def __init__(self):
        self.time = time.time()

    def tick(self):
        oldtime = self.time
        self.time = time.time()
        return self.time - oldtime
