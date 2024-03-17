import sys


if sys.platform == "darwin":
    from hook_darwin import *
else:
    raise Exception(f'Platform "{sys.platform}" is not yet supported.')
