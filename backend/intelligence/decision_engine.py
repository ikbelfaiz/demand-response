"""Future DR decision contract; historical labels never activate operations."""


def propose_event(*_args, **_kwargs):
    raise NotImplementedError("Automated DR event decisions are not implemented.")
