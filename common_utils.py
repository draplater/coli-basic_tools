from io import open

import graph_utils
from conll_reader import OldSDPSentence


def set_proc_name(newname):
    from ctypes import cdll, byref, create_string_buffer
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(len(newname)+1)
    buff.value = newname
    libc.prctl(15, byref(buff), 0, 0, 0)


def parse_dict(parser, dic):
    option_cmd = []
    for k, v in dic.items():
        assert isinstance(k, str)
        if v is True:
            option_cmd.append("--" + k)
        elif v is False:
            continue
        else:
            option_cmd.append("--" + k)
            option_cmd.append(str(v))

    return parser.parse_args(option_cmd)


def get_graphs(file_name, use_edge=True):
    with open(file_name, "r") as f:
        graphs = [graph_utils.Graph.from_sdp(i, use_edge)
                  for i in OldSDPSentence.get_all_sentences(f)]
    return graphs