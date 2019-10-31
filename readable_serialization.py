from ast import literal_eval

literal_magic = "::literal::"
def readable_deserialize(s):
    if s.startswith(literal_magic):
        return literal_eval(s[len(literal_magic):])
    else:
        return s


def readable_serialize(s):
    if isinstance(s, str):
        assert not s.startswith(literal_magic)
        return s
    else:
        return literal_magic + repr(s)