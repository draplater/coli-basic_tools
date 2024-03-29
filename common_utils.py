import collections
import pickle
import queue
import typing
from collections import UserDict, Counter, _count_elements
from io import open
import contextlib
import functools
import warnings

import os
import time

import sys

from itertools import islice
from threading import Thread
from typing import TypeVar, Dict, Generic

import dataclasses
import numpy as np
from wrapt import CallableObjectProxy

from .logger import logger

T = TypeVar("T")


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
    from coli.parser_tools.training_scheduler import dict_to_commandline
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

    def __setattr__(self, key, value):
        self[key] = value

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


# can be enabled by setting common_utils.cache_keeper = some_dict
cache_keeper = None


def try_cache_keeper(key):
    """
    Store objects globally to speed up debugging
    """

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if cache_keeper is None:
                return func(*args, **kwargs)
            else:
                result = cache_keeper.get(key)
                if result is None:
                    result = func(*args, **kwargs)
                    cache_keeper[key] = result
                    logger.info("Add into cache keeper: {}".format(key))
                else:
                    logger.info("Load from cache keeper: {}".format(key))
                return result

        return wrapped

    return wrapper


def cache_result_to(file_name_func, enable=True):
    if not enable:
        return lambda func: func

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            file_name = file_name_func(*args, **kwargs)
            if cache_keeper is not None and file_name in cache_keeper:
                logger.info("Use cache keeper: {}".format(file_name))
                ret = cache_keeper[file_name]
            elif os.path.exists(file_name):
                with open(file_name, "rb") as f:
                    logger.info("Use cached file: {}".format(file_name))
                    ret = pickle.load(f)
            else:
                ret = func(*args, **kwargs)
                with open(file_name, "wb") as f:
                    pickle.dump(ret, f, protocol=4)
                    logger.info("Cached file generated: {}".format(file_name))

            if cache_keeper is not None and file_name not in cache_keeper:
                cache_keeper[file_name] = ret
            return ret

        return wrapped

    return wrapper


def cache_result(file_name, enable=True):
    return cache_result_to(lambda *args, **kwargs: file_name, enable=enable)


# TODO: use tqdm instead
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
                    if isinstance(self.sum_values[k][0], int):
                        info += ' - %s: %d' % (k,
                                               self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    else:
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
                    if isinstance(self.sum_values[k][0], int):
                        info += ' - %s: %d' % (
                            k,
                            self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    else:
                        info += ' - %s: %.4f' % (
                            k,
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


def combine_sub_options(
        sub_classes_with_option: Dict[str, T],
        name="Options",
        extra=None):
    if extra is None:
        extra = {}
    base_classes = tuple(set(getattr(i, "Options")
                             for i in sub_classes_with_option.values()
                             if hasattr(i, "Options")
                             ))
    return dataclasses.dataclass(type(name, base_classes, extra))


def add_slots(cls):
    # https://raw.githubusercontent.com/ericvsmith/dataclasses/master/dataclass_tools.py
    # Need to create a new class, since we can't set __slots__
    #  after a class has been created.

    # Make sure __slots__ isn't already set.
    if '__slots__' in cls.__dict__:
        raise TypeError(f'{cls.__name__} already specifies __slots__')

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(f.name for f in dataclasses.fields(cls))
    cls_dict['__slots__'] = field_names
    for field_name in field_names:
        # Remove our attributes, if present. They'll still be
        #  available in _MARKER.
        cls_dict.pop(field_name, None)
    # Remove __dict__ itself.
    cls_dict.pop('__dict__', None)
    # And finally create the class.
    qualname = getattr(cls, '__qualname__', None)
    cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
    if qualname is not None:
        cls.__qualname__ = qualname
    return cls


class NoPickle(CallableObjectProxy):
    """
    An ObjectProxy that make an unpicklable object picklable,
    (but you will get `None` when restoring)
    """

    @classmethod
    def return_none(cls):
        return None

    def __reduce__(self):
        return self.return_none, ()

    def __reduce_ex__(self, version):
        return self.return_none, ()

    def __repr__(self):
        return "(NoPickle): " + repr(self.__wrapped__)

    def __deepcopy__(self, memo):
        # TODO: is it correct?
        return None


class IdentityGetAttr(object):
    """
    create an object a such that a.xxx = "xxx", a.yyy = "yyy"
    """

    def __getattr__(self, item):
        return item


def identity(arg): return arg


class UserCounterBase(UserDict):
    def __init__(*args, **kwds):
        if not args:
            raise TypeError("descriptor '__init__' of 'Counter' object "
                            "needs an argument")
        self, *args = args
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        super(UserCounterBase, self).__init__()
        self.update(*args, **kwds)

    def update(*args, **kwds):
        if not args:
            raise TypeError("descriptor 'update' of 'Counter' object "
                            "needs an argument")
        self, *args = args
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        iterable = args[0] if args else None
        if iterable is not None:
            if isinstance(iterable, collections.Mapping):
                if self:
                    self_get = self.get
                    for elem, count in iterable.items():
                        self[elem] = count + self_get(elem, 0)
                else:
                    super(UserCounterBase, self).update(iterable)  # fast path when counter is empty
            else:
                _count_elements(self, iterable)
        if kwds:
            self.update(kwds)


UserCounter: typing.Type[typing.Counter] = type("UserCounter", (UserCounterBase,),
                                                {k: v for k, v in Counter.__dict__.items()
                                                 if k not in {"__dict__", "__weakref__", "__reduce__", "__init__",
                                                              "update"}}
                                                )
UserCounter.__module__ = UserCounterBase.__module__


class ValueContainer(Generic[T]):
    def __init__(self, default: T):
        self.value: T = default

    def get(self) -> T:
        return self.value

    def set(self, value: T):
        self.value = T


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


class Singleton(type):
    """Metaclass which implements the singleton pattern"""

    _instances = {}

    def __call__(self, *args, **kwargs):
        if self not in self._instances:
            self._instances[self] = super(Singleton, self).__call__(*args, **kwargs)
        return self._instances[self]

    def __copy__(cls, instance):
        return instance


class RepeatableGenerator(object):
    def __init__(self, generator_factory):
        self.generator_factory = generator_factory

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    def __iter__(self):
        return iter(self.generator_factory(*self.args, **self.kwargs))


def repeatable(generator):
    return RepeatableGenerator(generator)


def generate_to_queue(gen, args, kwargs, q):
    try:
        for item in gen(*args, **kwargs):
            q.put(("item", item))
    except Exception as e:
        q.put(("exception", e))
    q.put((StopIteration, None))


def run_generator_in_thread(gen, *args, **kwargs):
    q = queue.Queue(maxsize=1024)

    thread = Thread(target=generate_to_queue, args=(gen, args, kwargs, q))
    thread.start()
    while True:
        info, i = q.get()
        if info == StopIteration:
            break
        elif info == "exception":
            raise i
        else:
            assert info == "item"
            print("qsize", q.qsize())
            yield i


def run_in_thread(gen):
    @functools.wraps(gen)
    def wrapped(*args, **kwargs):
        yield from run_generator_in_thread(gen, *args, **kwargs)

    return wrapped


def run_generator_in_process(gen, *args, **kwargs):
    import multiprocessing
    from multiprocessing.managers import SyncManager
    manager = SyncManager()
    manager.start()
    q = manager.Queue(maxsize=1024)

    process = multiprocessing.Process(
        target=generate_to_queue, args=(gen, args, kwargs, q))
    process.start()
    while True:
        info, i = q.get()
        if info == StopIteration:
            break
        elif info == "exception":
            raise i
        else:
            assert info == "item"
            print("qsize", q.qsize())
            yield i
    manager.shutdown()


def run_in_process(gen):
    @functools.wraps(gen)
    def wrapped(*args, **kwargs):
        yield from run_generator_in_process(gen, *args, **kwargs)

    return wrapped


class WritersProxy(object):
    def __init__(self, *writers):
        self.writers = writers

    def write(self, content):
        for i in self.writers:
            i.write(content)

    def flush(self):
        for i in self.writers:
            i.flush()


def screen_and_file(file_stream, screen_stream=sys.stdout):
    if file_stream != sys.stderr:
        return WritersProxy(file_stream, screen_stream)
    return screen_stream

