import sys
import platform

# format: ('_major', '_minor', '_patch')
watermark = dict(python=f"{sys.version_info.major}.{sys.version_info.minor}")
platform = platform.system()
