import argparse
from collections import OrderedDict
from typing import List, Any

import dataclasses
from dataclasses import field, MISSING, dataclass

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
    for var in klass.__dataclass_fields__:
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
        if sub_namespace:
            self.original_parser.add_argument("--" + sub_namespace,
                                              action=dict_key_action_factory(choices),
                                              choices=choices.keys(),
                                              default=instance_or_class
                                              )

        self.sub_namespace = sub_namespace
        if self.sub_namespace:
            self.sub_namespace += "."

        origin_class_map = dataclasses_trace_origin(instance_or_class.__class__)
        # docs
        class_to_groups = {i: original_parser.add_argument_group(title=i.__qualname__)
                           for i in set(origin_class_map.values())}
        for field in dataclasses.fields(instance_or_class):
            properties = field.metadata.get(meta_key)
            key = field.name
            if mode == "predict" and properties and not properties.predict_time:
                continue
            if mode == "train" and properties and not properties.train_time:
                continue
            if not isinstance(instance_or_class, type):
                value = getattr(instance_or_class, key)
            else:
                value = field.default
            if mode == "predict" and properties:
                value = properties.predict_default

            # solve nested dataclass
            if dataclasses.is_dataclass(value):
                if properties:
                    sub_choices = properties.choices
                else:
                    sub_choices = field.metadata.get("choices")
                if sub_choices is None:
                    sub_choices = {"default": value}
                DataClassArgParser(self.sub_namespace + key, original_parser,
                                   choices=sub_choices)
                continue

            default_list = " (default: {}".format(value)
            for choice_key, choice_dict in choices.items():
                alt_value = getattr(choice_dict, key)
                if alt_value != value:
                    default_list += ", {}: {}".format(choice_key, alt_value)
            default_list += ")"

            original_class = origin_class_map[key]
            option_choices = field.metadata.get("choices")
            arg_type = value.__class__
            help_str = None

            # get help or type from annotations
            annotation = original_class.__annotations__.get(key)
            if isinstance(annotation, str):
                help_str = annotation
            elif isinstance(annotation, type):
                arg_type = annotation
            elif hasattr(annotation, "__origin__") and \
                    hasattr(annotation.__origin__, "__mro__"):
                # for generic type annotation like List[int]
                arg_type = annotation.__origin__.__mro__[1]
            elif properties.type == MISSING:
                raise Exception(
                    f"Cannot determine type for argument {key} "
                    f"when annotation is {annotation} ")

            if properties is not None:
                option_choices = properties.choices
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
                    nargs=properties.nargs if properties else None
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
    choices: List[Any]
    required: bool
    type: Any
    help: str
    metavar: str
    nargs: Any
    train_time: bool  # whether this arg is used when training
    predict_time: bool  # whether this arg is used when predicting
    predict_default: Any  # use the same value as training if MISSING


class Required(object):
    def __str__(self):
        return "(Required)"

    def __repr__(self):
        return "(Required)"


class AsTraining(object):
    def __str__(self):
        return "(Use the same value as training)"

    def __repr__(self):
        return "(Use the same value as training)"


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
    return field(default=default, default_factory=default_factory,
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
