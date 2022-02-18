from typing import NamedTuple, Optional


class SampleDDLaneParam(NamedTuple):
    extra: Optional[dict] = {}


if __name__ == '__main__':
    d = {"extra": {"haha":1}}
    args =SampleDDLaneParam(**d,w=2)
    print(args.extra)

