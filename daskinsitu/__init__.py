__version__ = '0.1.0'

from .daskinsitu import from_h5dataset, from_h5group, \
                        compute, close_open_files, \
                        NotGroupError, NotDatasetError
