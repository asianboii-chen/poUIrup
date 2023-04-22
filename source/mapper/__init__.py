import sys
import typing

if sys.platform == 'darwin':
    from . import hook_darwin as hook
else:
    raise Exception(f'Platform "{sys.platform}" is not yet supported.')


class Context(typing.NamedTuple):
    hook_context: hook.Context


def create() -> Context:
    hook_context = hook.create()
    return Context(hook_context)


def process(context: Context) -> None:
    hook.process(context.hook_context)
