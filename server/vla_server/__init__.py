# Lazy imports to avoid circular import issues when running as module
def __getattr__(name):
    if name == 'OpenVLAWrapper':
        from .openvla_wrapper import OpenVLAWrapper
        return OpenVLAWrapper
    elif name == 'VLAServer':
        from .server import VLAServer
        return VLAServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['OpenVLAWrapper', 'VLAServer']
