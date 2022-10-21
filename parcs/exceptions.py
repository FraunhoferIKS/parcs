class DataError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class GraphError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class GuidelineError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class DescriptionFileError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class RandomizerError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def parcs_assert(condition, action, msg):
    if not condition:
        raise action(msg)


if __name__ == '__main__':
    parcs_assert(False, RandomizerError, 'hi man')