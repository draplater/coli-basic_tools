import argparse
import gzip
import pickle
from collections import UserDict, OrderedDict
from io import open
import contextlib
import functools
import warnings
from argparse import ArgumentParser
from optparse import OptionParser

import os
import time

import sys

from itertools import islice

import dataclasses
import numpy as np
from dataclasses import is_dataclass

from logger import logger


def set_proc_name(newname):
    from ctypes import cdll, byref, create_string_buffer
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(len(newname) + 1)
    buff.value = newname.encode("utf-8")
    libc.prctl(15, byref(buff), 0, 0, 0)


def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as err:
        if err.errno != 17:
            raise


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


@deprecated
def parse_dict(parser, dic, prefix=()):
    from training_scheduler import dict_to_commandline
    return parser.parse_args(dict_to_commandline(dic, prefix))


def under_construction(func):
    """This is a decorator which can be used to mark functions
    as under construction. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn("Call to under construction function {}.".format(func.__name__), category=UserWarning,
                      stacklevel=2)
        return func(*args, **kwargs)

    return new_func


class Timer(object):
    def __init__(self):
        self.time = time.time()

    def tick(self):
        oldtime = self.time
        self.time = time.time()
        return self.time - oldtime


@contextlib.contextmanager
def smart_open(filename, mode="r", *args, **kwargs):
    if filename != '-':
        fh = open(filename, mode, *args, **kwargs)
    else:
        if mode.startswith("r"):
            fh = sys.stdin
        elif mode.startswith("w") or mode.startswith("a"):
            fh = sys.stdout
        else:
            raise ValueError("invalid mode " + mode)

    try:
        yield fh
    finally:
        if fh is not sys.stdout and fh is not sys.stdin:
            fh.close()


def split_to_batches(iterable, batch_size):
    iterator = iter(iterable)
    sent_id = 0
    batch_id = 0

    while True:
        piece = list(islice(iterator, batch_size))
        if not piece:
            break
        yield sent_id, batch_id, piece
        sent_id += len(piece)
        batch_id += 1


class AttrDict(dict):
    """A dict whose items can also be accessed as member variables.

    >>> d = AttrDict(a=1, b=2)
    >>> d['c'] = 3
    >>> print d.a, d.b, d.c
    1 2 3
    >>> d.b = 10
    >>> print d['b']
    10

    # but be careful, it's easy to hide methods
    >>> print d.get('c')
    3
    >>> d['get'] = 4
    >>> print d.get('a')
    Traceback (most recent call last):
    TypeError: 'int' object is not callable
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    @property
    def __dict__(self):
        return self


class IdentityDict(object):
    """ A dict like IdentityHashMap in java"""

    def __init__(self, seq=None):
        self.dict = dict(seq=((id(key), value) for key, value in seq))

    def __setitem__(self, key, value):
        self.dict[id(key)] = value

    def __getitem__(self, item):
        return self.dict[id(item)]

    def get(self, key, default=None):
        return self.dict.get(id(key), default)

    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return repr(self.dict)

    def __getstate__(self):
        raise NotImplementedError("Cannot pickle this.")


def dict_key_action_factory(choices):
    class DictKeyAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # noinspection PyUnresolvedReferences
            setattr(namespace, self.dest, dataclasses.replace(choices[values]))

    return DictKeyAction


def dataclasses_trace_origin(klass, result_container=None):
    if result_container is None:
        result_container = OrderedDict()
    if not dataclasses.is_dataclass(klass):
        return result_container
    for var in klass.__dataclass_fields__:
        result_container[var] = klass
    for base in klass.__bases__:
        dataclasses_trace_origin(base, result_container)
    return result_container


class DictionarySubParser(argparse._ArgumentGroup):
    def __init__(self, sub_namespace, original_parser, choices=None,
                 title=None,
                 description=None,
                 default_key="default"):
        for i in choices.values():
            assert is_dataclass(i)
        super(DictionarySubParser, self).__init__(
            original_parser, title=title, description=description)
        self.sub_namespace = sub_namespace
        self.original_parser = original_parser
        default_obj = choices[default_key]
        if choices is not None:
            self.original_parser.add_argument("--" + self.sub_namespace,
                                              action=dict_key_action_factory(choices),
                                              choices=choices.keys(),
                                              default=default_obj
                                              )
        params_dict = dataclasses.asdict(default_obj)
        origin_class_map = dataclasses_trace_origin(default_obj.__class__)
        # docs
        class_to_groups = {i: original_parser.add_argument_group(title=i.__qualname__)
                           for i in set(origin_class_map.values())}
        for key, value in params_dict.items():
            # TODO: resursive dataclasses
            # if dataclasses.is_dataclass(value):
            #     DictionarySubParser(self.sub_namespace + "." + key, self)
            # else:
            default_list = " (default: {}".format(value)
            for choice_key, choice_dict in choices.items():
                alt_value = getattr(choice_dict, key)
                if alt_value != value:
                    default_list += ", {}: {}".format(choice_key, alt_value)
            default_list += ")"

            original_class = origin_class_map[key]
            option_choices = original_class.__dataclass_fields__[key].metadata.get("choices")
            help = original_class.__annotations__.get(key)
            self.add_argument(
                "--" + key.replace("_", "-"), type=value.__class__,
                help="{}{}".format(help, default_list),
                choices=option_choices,
                original_parser=class_to_groups[original_class]
            )

    def add_argument(self, *args, **kwargs):
        original_parser = kwargs.get("original_parser") or self.original_parser
        if "original_parser" in kwargs:
            kwargs.pop("original_parser")

        def modify_names(name):
            last_hyphen = -1
            for i, char in enumerate(name):
                if char == "-":
                    last_hyphen = i
                else:
                    break
            last_hyphen += 1
            return name[:last_hyphen] + self.sub_namespace + "." + name[last_hyphen:]

        if "dest" in kwargs:
            kwargs["dest"] = self.sub_namespace + "." + kwargs["dest"]

        original_action_input = kwargs.get("action")
        if original_action_input is None or \
                isinstance(original_action_input, (str, bytes)):
            original_action_class = self._registry_get(
                "action", original_action_input, original_action_input)
        else:
            original_action_class = original_action_input
        kwargs["action"] = group_action_factory(self.sub_namespace, original_action_class)
        kwargs["default"] = argparse.SUPPRESS
        original_parser.add_argument(
            *[modify_names(i) for i in args],
            **kwargs)


def group_action_factory(group_name, original_action_class):
    class GroupAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            assert dest.startswith(group_name + ".")
            dest = dest[len(group_name) + 1:]
            super(GroupAction, self).__init__(option_strings, dest, **kwargs)
            self.original_action_obj = original_action_class(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            groupspace = getattr(namespace, group_name)
            self.original_action_obj(parser, groupspace, values, option_string)

    return GroupAction


def read_embedding(embedding_filename, encoding):
    if embedding_filename.endswith(".gz"):
        external_embedding_fp = gzip.open(embedding_filename, 'rb')
    else:
        external_embedding_fp = open(embedding_filename, 'rb')

    def embedding_gen():
        for line in external_embedding_fp:
            fields = line.decode(encoding).strip().split(' ')
            if len(fields) <= 2:
                continue
            token = fields[0]
            vector = [float(i) for i in fields[1:]]
            yield token, vector

    external_embedding = list(embedding_gen())
    external_embedding_fp.close()
    return external_embedding


def cache_result_to(file_name_func, enable=True):
    if not enable:
        return lambda func: func

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            file_name = file_name_func(*args, **kwargs)
            if os.path.exists(file_name):
                with open(file_name, "rb") as f:
                    logger.info("Use cached file: {}".format(file_name))
                    return pickle.load(f)
            else:
                result = func(*args, **kwargs)
                with open(file_name, "wb") as f:
                    pickle.dump(result, f)
                    logger.info("Cached file generated: {}".format(file_name))
                return result

        return wrapped

    return wrapper


def cache_result(file_name, enable=True):
    return cache_result_to(lambda *args, **kwargs: file_name, enable=enable)


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, log_func=None):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose
        self.log_func = log_func

    def update(self, current, values=(), exact=(), strict=()):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        if self.log_func is not None:
            log_string = "Step: {}/{} ".format(current, self.target)
            if values:
                log_string += " values: {}".format(values)
            if exact:
                log_string += " exact: {}".format(exact)
            if strict:
                log_string += " strict: {}".format(strict)
            self.log_func(log_string)

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def finish(self):
        sys.stdout.write("\n")

    def add(self, n, values=()):
        self.update(self.seen_so_far + n, values)


class SmartDefaultDict(UserDict):
    def __init__(self, default_factory, seq=()):
        self.default_factory = default_factory
        super(SmartDefaultDict, self).__init__(seq)

    def __missing__(self, key):
        value = self[key] = self.default_factory(key)
        return value


def set_default_attr(obj, attr, value):
    if not hasattr(obj, attr):
        setattr(obj, attr, value)


def combine_sub_options(sub_classes_with_option, name="Options", extra=None):
    if extra is None:
        extra = {}
    base_classes = tuple(getattr(i, "Options")
                         for i in sub_classes_with_option.values()
                         if hasattr(i, "Options")
                         )
    return dataclasses.dataclass(type(name, base_classes, extra))
