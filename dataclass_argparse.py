import argparse
import inspect
from abc import ABCMeta, abstractproperty
from collections import OrderedDict
from operator import itemgetter
from pprint import pformat
from typing import List, Any, Union, NewType, Optional, Iterable

import dataclasses
from dataclasses import MISSING, dataclass
from typeguard import check_type

from coli.basic_tools.common_utils import NoPickle, Singleton
from coli.basic_tools.logger import default_logger

ExistFile = NewType("ExistFile", str)

meta_key = "__ARGPARSE__"


def bool_convert(input_bool):
    if input_bool == "False":
        return False
    elif input_bool == "True":
        return True
    else:
        raise Exception("Unknown bool value {}".format(input_bool))


def dataclasses_trace_origin(klass, result_container=None):
    if result_container is None:
        result_container = OrderedDict()
    if not dataclasses.is_dataclass(klass):
        return result_container
    for var in dataclasses.fields(klass):
        result_container[var] = klass
    for base in klass.__bases__:
        dataclasses_trace_origin(base, result_container)
    return result_container


def dict_key_action_factory(choices):
    class DictKeyAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # noinspection PyUnresolvedReferences
            setattr(namespace, self.dest, dataclasses.replace(choices[values]))

    return DictKeyAction


class DataClassArgParser(argparse._ArgumentGroup):
    def __init__(self, sub_namespace, original_parser, choices=None,
                 title=None,
                 description=None,
                 default_key="default",
                 mode="train"
                 ):
        for i in choices.values():
            assert dataclasses.is_dataclass(i)
        super(DataClassArgParser, self).__init__(
            original_parser, title=title, description=description)
        self.original_parser = original_parser

        instance_or_class = choices[default_key]
        # FIXME: consider the container itself
        if sub_namespace and "." not in sub_namespace:
            self.original_parser.add_argument("--" + sub_namespace,
                                              action=dict_key_action_factory(choices),
                                              choices=choices.keys(),
                                              default=instance_or_class
                                              )

        self.sub_namespace = sub_namespace
        if self.sub_namespace:
            self.sub_namespace += "."

        origin_class_map = {k.name: v for k, v in
                            dataclasses_trace_origin(instance_or_class.__class__).items()}
        # docs
        class_to_groups = {i: original_parser.add_argument_group(title=i.__qualname__)
                           for i in set(origin_class_map.values())}
        for field in dataclasses.fields(instance_or_class):
            properties = field.metadata.get(meta_key) or default_arg_properties
            key = field.name
            if not isinstance(instance_or_class, type):
                if isinstance(instance_or_class, OptionsBase):
                    value = instance_or_class.get_value(key)
                else:
                    default_logger.warning(f"{self.sub_namespace} ({instance_or_class.__class__.__qualname__})"
                                           f"is a dataclass but not OptionsBase.")
                    value = getattr(instance_or_class, key)
            else:
                value = field.default
            train_value = value
            if mode == "predict":
                value = properties.predict_default

            if mode == "predict" and not dataclasses.is_dataclass(train_value) and not properties.predict_time:
                continue
            if mode == "train" and not properties.train_time:
                continue

            # solve nested dataclass
            if dataclasses.is_dataclass(train_value):
                sub_choices = properties.choices or \
                              field.metadata.get("choices") or \
                              {"default": train_value}
                DataClassArgParser(self.sub_namespace + key, original_parser,
                                   choices=sub_choices, mode=mode)
                continue

            default_list = " (default: {}".format(value)
            for choice_key, choice_dict in choices.items():
                alt_value = getattr(choice_dict, key)
                if alt_value != value:
                    default_list += ", {}: {}".format(choice_key, alt_value)
            default_list += ")"

            original_class = origin_class_map[key]
            arg_type = value.__class__
            help_str = None

            # get help or type from annotations
            annotation = original_class.__annotations__.get(key)
            this_annotation = instance_or_class.__class__.__annotations__.get(key)
            if this_annotation is not None and this_annotation is not Any:
                annotation = this_annotation

            if isinstance(annotation, str):
                help_str = annotation
            elif hasattr(annotation, "__args__") \
                    and len(annotation.__args__) == 1:
                # for generic type annotation like List[int]
                arg_type = annotation.__args__[0]
            elif getattr(annotation, "__origin__", None) == Union and \
                    annotation.__args__[1] is type(None) \
                    and callable(annotation.__args__[0]):
                # for optional type annotation like Optional[int]
                # ignore ForwardRef
                arg_type = annotation.__args__[0]
            elif isinstance(annotation, type):
                arg_type = annotation
            elif annotation is ExistFile:
                arg_type = str
            elif properties.type == MISSING:
                raise Exception(
                    f"Cannot determine type for argument \"{key}\" "
                    f"when annotation is {annotation} ")

            option_choices = properties.choices or field.metadata.get("choices")
            if properties.help != MISSING:
                help_str = properties.help
            if properties.type != MISSING:
                arg_type = properties.type

            if help_str is None:
                help_str = arg_type

            if self.sub_namespace:
                self.add_argument(
                    "--" + key.replace("_", "-"),
                    type=arg_type if arg_type != bool else bool_convert,
                    help="{}{}".format(help_str, default_list),
                    choices=option_choices,
                    original_parser=class_to_groups[original_class]
                )
            else:
                class_to_groups[original_class].add_argument(
                    "--" + key.replace("_", "-"),
                    type=arg_type if arg_type != bool else bool_convert,
                    help="{}".format(help_str),
                    default=value if value is not REQUIRED else None,
                    required=value is REQUIRED,
                    choices=option_choices,
                    nargs=properties.nargs
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
            return name[:last_hyphen] + self.sub_namespace + name[last_hyphen:]

        if "dest" in kwargs:
            kwargs["dest"] = self.sub_namespace + kwargs["dest"]

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
            assert dest.startswith(group_name)
            self.group_name = group_name
            dest = dest[len(group_name):]
            super(GroupAction, self).__init__(option_strings, dest, **kwargs)
            self.original_action_obj = original_action_class(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            group_name = self.group_name.rstrip(".")
            while True:
                parts = group_name.split(".", 1)
                if len(parts) == 1:
                    break
                sub_namespace_key, group_name = parts
                namespace = getattr(namespace, sub_namespace_key)
            groupspace = getattr(namespace, group_name)
            self.original_action_obj(parser, groupspace, values, option_string)

    return GroupAction


@dataclass
class ArgProperties(object):
    choices: Optional[Iterable]
    required: bool
    type: Any
    help: str
    metavar: Optional[str]
    nargs: Any
    train_time: bool  # whether this arg is used when training
    predict_time: bool  # whether this arg is used when predicting
    predict_default: Any  # use the same value as training if MISSING


# used when argument is create with field(..) instead of argfield(..)
default_arg_properties = ArgProperties(choices=None, required=False,
                                       type=MISSING, help=MISSING, metavar=None,
                                       nargs=None, train_time=True, predict_time=False,
                                       predict_default=MISSING)


class Required(metaclass=Singleton):
    def __str__(self):
        return "(Required)"

    def __repr__(self):
        return "(Required)"

    def __deepcopy__(self, memo=None):
        return self


class AsTraining(metaclass=Singleton):
    def __str__(self):
        return "(Use the same value as training)"

    def __repr__(self):
        return "(Use the same value as training)"

    def __deepcopy__(self, memo=None):
        return self


REQUIRED = Required()
AS_TRAINING = AsTraining()


def argfield(default=REQUIRED, *, default_factory=MISSING,
             choices=None, help=MISSING, metavar=None, nargs=None, type=MISSING,
             train_time=True, predict_time=False, predict_default=MISSING,
             init=True, repr=True, hash=None, compare=True, metadata=None,
             ):
    if default_factory is not MISSING and default is REQUIRED:
        default = MISSING
    required = default is REQUIRED

    if predict_default is MISSING:
        if predict_time:
            if not train_time:
                predict_default = REQUIRED
            else:
                predict_default = AS_TRAINING

    metadata_ = {meta_key: ArgProperties(choices, required, type, help, metavar,
                                         nargs, train_time,
                                         predict_time, predict_default)}
    if metadata is not None:
        metadata_.update(metadata)
    return dataclasses.field(default=default, default_factory=default_factory,
                             init=init, repr=repr, hash=hash, compare=compare,
                             metadata=metadata_)


def check_argparse_result(args, namespace=""):
    if namespace:
        namespace += "."

    for name, value in dict(args.__dict__).items():
        if value is REQUIRED:
            raise Exception(f"Parameter {namespace}{name} is required")
        elif value is AS_TRAINING:
            delattr(args, name)


def check_options(op, is_training=True, namespace="options"):
    assert dataclasses.is_dataclass(op)
    if isinstance(op, OptionsBase):
        # noinspection PyDataclass
        for field in op.generate_valid_fields():
            value = getattr(op, field.name)
            full_name = f"{namespace}.{field.name}"
            if dataclasses.is_dataclass(value):
                check_options(value, is_training, namespace=full_name)
            else:
                argparse_metadata = field.metadata.get(meta_key) or default_arg_properties
                if is_training and argparse_metadata.train_time and value is REQUIRED:
                    raise ValueError(f"{full_name} is required when training")
                if (not is_training) and argparse_metadata.predict_default and value is REQUIRED:
                    raise ValueError(f"{full_name} is required when training")
    else:
        default_logger.warning(f"{op.__class__.__qualname__} should inherent OptionsBase")


def pretty_format(obj, indent=0, is_training=True):
    if dataclasses.is_dataclass(obj):
        if not isinstance(obj, OptionsBase):
            default_logger.warning(f"{obj.__class__.__qualname__} should inherent OptionsBase")
            return pformat(obj.__dict__)
        else:
            return obj.pretty_format(indent, is_training)
    elif isinstance(obj, argparse.Namespace):
        return "\n".join(f"{k}={pretty_format(v)}" for k, v in obj.__dict__.items())
    else:
        return pformat(obj)


def merge_predict_time_options(train_options, predict_options, prefix=""):
    if isinstance(train_options, OptionsBase):
        valid_fields = {i.name for i in train_options.generate_valid_fields(False)}
    else:
        valid_fields = train_options.__dict__.keys()

    for k in valid_fields:
        if not hasattr(predict_options, k):
            continue
        v = getattr(predict_options, k)
        if isinstance(v, AsTraining) or v is MISSING:
            continue
        if dataclasses.is_dataclass(v):
            assert isinstance(v, OptionsBase)
            merge_predict_time_options(getattr(train_options, k), v, prefix + k + ".")
        else:
            if hasattr(train_options, k):
                setattr(train_options, k, v)
            else:
                default_logger.info(f"Redundant option {prefix}{k}")


FIELDS_TO_ORIGIN = "__fields_to_origin__"
NAMES_TO_FIELDS = "__names_to_fields__"


class OptionsBase(object):
    def generate_valid_fields(self, is_training=True):
        # noinspection PyDataclass
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            argparse_metadata = field.metadata.get(meta_key) or default_arg_properties
            if isinstance(value, OptionsBase) or \
                    is_training and argparse_metadata.train_time \
                    or (not is_training) and argparse_metadata.predict_time:
                yield field

    def pretty_format(self, indent=0, is_training=True):
        ret = f'{self.__class__.__qualname__}(\n'
        field_values = []
        is_empty = True
        for field in self.generate_valid_fields(is_training):
            value = getattr(self, field.name)
            if value is MISSING:
                continue
            if dataclasses.is_dataclass(value):
                if not isinstance(value, OptionsBase):
                    default_logger.warning(f"{value.__class__.__qualname__} should inherent OptionsBase")
                    value_str = pformat(value.__dict__)
                else:
                    value_str = value.pretty_format(indent + 2, is_training)
                if value_str:
                    is_empty = False
                    field_values.append((field.name, value_str, True))
            else:
                is_empty = False
                value_str = repr(value)
                field_values.append((field.name, value_str, False))
        if is_empty:
            return ""
        else:
            ret += ",\n".join(f'{" " * (indent + 2)}{key}={value}'
                              for key, value, _ in sorted(field_values, key=itemgetter(2)))
            ret += f'\n{" " * indent})'
            return ret

    def _get_original_fields(self):
        assert dataclasses.is_dataclass(self)
        fields_to_origin = getattr(self, FIELDS_TO_ORIGIN, None)
        names_to_fields = getattr(self, NAMES_TO_FIELDS, None)
        if fields_to_origin is None:
            fields_to_origin = dataclasses_trace_origin(self.__class__)
            setattr(self, FIELDS_TO_ORIGIN, NoPickle(fields_to_origin))
        else:
            fields_to_origin = fields_to_origin.__wrapped__
        if names_to_fields is None:
            names_to_fields = {i.name: i for i in fields_to_origin}
            setattr(self, NAMES_TO_FIELDS, NoPickle(names_to_fields))
        else:
            names_to_fields = names_to_fields.__wrapped__
        return fields_to_origin, names_to_fields

    def check_key(self, key, value):
        fields_to_origin, names_to_fields = self._get_original_fields()
        if key not in names_to_fields:
            raise KeyError(f"{self.__class__.__qualname__} has no attribute \"{key}\"")
        field = names_to_fields[key]
        annotation = fields_to_origin[field].__annotations__.get(key)
        argparse_metadata = field.metadata.get(meta_key) or default_arg_properties
        if argparse_metadata.choices is not None and value not in argparse_metadata.choices:
            raise KeyError(f'Invalid value "{value}" for {self.__class__.__qualname__}.{key}. '
                           f'Must chosen from {"{" + ",".join(argparse_metadata.choices) + "}"}')
        if argparse_metadata.type is not MISSING:
            annotation = argparse_metadata.type
        check_type(key, value, annotation)

    def get_value(self, key):
        """
        get value without checking
        """
        return self.__getattribute__(key)

    def __setattr__(self, key, value):
        # ignore setattr from self
        current_frame = inspect.currentframe()
        if current_frame.f_back.f_locals.get("self") is self:
            return super(OptionsBase, self).__setattr__(key, value)

        self.check_key(key, value)
        super(OptionsBase, self).__setattr__(key, value)

    @classmethod
    def get_default(cls):
        return cls()

    def to_predict_default(self):
        fields_to_origin, names_to_fields = self._get_original_fields()
        for key, field in names_to_fields.items():
            properties = field.metadata.get(meta_key) or default_arg_properties
            value = getattr(self, key)
            if dataclasses.is_dataclass(value):
                assert isinstance(value, OptionsBase)
                value.to_predict_default()
            else:
                if not properties.predict_time:
                    setattr(self, key, MISSING)
                else:
                    setattr(self, key, properties.predict_default)
        return self


class BranchSelect(metaclass=ABCMeta):
    branches = abstractproperty()

    @dataclass
    class Options(OptionsBase, metaclass=ABCMeta):
        type = abstractproperty()

        def generate_valid_fields(self, is_training=True):
            for field in super().generate_valid_fields(is_training):
                if not isinstance(self.type, AsTraining) and self.type != MISSING and \
                        field.name.endswith("_options") and field.name != self.type + "_options":
                    continue
                yield field

        def __getattribute__(self, key):
            # ignore getattribute from self
            current_frame = inspect.currentframe()
            if current_frame.f_back.f_locals.get("self") is self:
                return super().__getattribute__(key)

            if self.type is not MISSING and not isinstance(self.type, AsTraining) \
                    and key.endswith("_options") and key != self.type + "_options":
                raise KeyError(f'try to use {key} when type is "{self.type}"')
            return super().__getattribute__(key)

        def __repr__(self):
            return f"{self.__class__.__name__}(type={self.type}," \
                f'{self.type}_options={getattr(self, self.type + "_options")})'

    @classmethod
    def get(cls, options: Options, **kwargs):
        contextual_unit_class = cls.branches[options.type]
        if contextual_unit_class is None:
            return None
        branch_kwargs = {}
        branch_options = getattr(options, f"{options.type}_options")
        assert dataclasses.is_dataclass(branch_options)
        branch_kwargs.update({i.name: getattr(branch_options, i.name)
                              for i in dataclasses.fields(branch_options)})
        branch_kwargs.update(kwargs)
        return contextual_unit_class(**branch_kwargs)
