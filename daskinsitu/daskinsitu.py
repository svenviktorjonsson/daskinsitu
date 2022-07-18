"""Library for loading data as dask arrays without keeping files open.

The opening of the file is delayed until the `compute` method is called on the
dask arrays. If you use `dask.compute` you need to close open files with the
method `close_open_files` if you don't want files to remain open.

Examples:

>>> import daskinsitu as di
>>> array = di.from_h5dataset("path/to/file.h5","/dataset/path","real")
>>> array.shape
(11, 31, 2, 181, 3)

>>> array[:3,:2,:1].shape
(3, 2, 1, 181, 360)

>>> di.compute(array[0,0,:,0,0])
(array([-0.03268581, -0.01254339]),)

>>> group = di.from_h5group("path/to/file.h5","/grouppath")

>>> list(group.keys())
['RET 1', 'Frequency', 'Polarization', 'Theta', 'Phi', 'real', 'imag']
"""


import dask.array as da
import dask
import h5py
import os
from typing import Any


_OPEN_FILES_ = {}


class NotGroupError(RuntimeError):
    pass


class NotDatasetError(RuntimeError):
    pass


def from_h5dataset(file_path: str,
                   dataset_path: str,
                   field_name: str=None,
                   **kwargs) -> da.Array:
    """Loads a dataset as a dask array from a h5-file without having files open.

    The opening of the file is delayed until the `compute` method is executed
    on the resulting array. Only the part that has been indexed is retrieved 
    from the file and after this all files that where opened during the 
    `compute` call are closed.


    Args:
        file_path: The path to the h5-file.
        dataset_path: The key that accesses the dataset in the h5-file.
        field_name: The key that accesses the field for structured arrays.
        kwargs: Extra key value arguments passed to `dask.array.from_delayed`.

    Returns:
        A dask array representing the h5 dataset.

    Raises:
        FileNotFoundError: If `file_path` is not valid and dtype or shape is
                           not given.
        NotDatasetError: If ` is not found and dtype or shape is not given.

    """

    delayed = dask.delayed(_get_dataset)(file_path, dataset_path, field_name)
    if not {"dtype","shape"} <= set(kwargs):
        ds_info = _get_ds_info(file_path, dataset_path, field_name)
        kwargs.update(ds_info)
    return da.from_delayed(delayed, **kwargs)


def from_h5group(file_path: str, group_path: str) -> dict[str, da.Array]:
    """Loads all datasets in a h5-group via the function `from_h5dataset`.

    In contrast to `from_h5dataset` this function raises an error if the file
    or group can't be found.

    Args:
        file_path: The path to the h5-file.
        group_path: The key that accesses the group in the h5-file.

    Returns:
        A dictionary of dask arrays with keys equal to the dataset names.
        Each field of structured datasets are accessed by the field name.
    
    Raises:
        FileNotFoundError: If `file_path` does not exist.
        NotGroupError: If the `group_path` does not lead to a h5-group.
    
    """

    _raise_error_if_file_not_found(file_path)
    with h5py.File(file_path) as file:
        if not isinstance((group:=file[group_path]), h5py.Group):
            raise NotGroupError
        output = {}
        for ds_name, ds  in group.items():
            if not isinstance(ds, h5py.Dataset):
                continue
            keys_dtypes = [(name, ds.dtype[name]) for name in ds.dtype.names]\
                          if ds.dtype.names else [(ds_name, ds.dtype)]
            ds_path = f"{group_path}/{ds_name}"
            for key, dtype in keys_dtypes:
                output[key] = from_h5dataset(file_path, ds_path,
                                             shape = ds.shape,
                                             dtype = dtype)
        return output


def compute(*args: Any) -> tuple:
    """Analogous to dask.array.compute but will close open files afterwards.

    See documentation for `dask.compute` for more information. If you use
    dask.compute you need to manually close files with `close_open_files`.
    
    Args:
        args: Any number of arguments that we want to make concrete.

    Returns:
        A tuple of computed values.

    Raises: 
        FileNotFoundError: If any of the arguments accesses a file that
                           no longer is available.
        NotDatasetError: If any of the arguments was created from a group path
                         rather than a dataset path.
    """
    is_wrapper = lambda x: isinstance(x, h5py._hl.dataset.FieldsWrapper)
    result = tuple([x[:] if is_wrapper else x for x in da.compute(*args)])
    close_open_files()
    return result


def close_open_files() -> None:
    """Closes files that have been opened using this module."""

    while _OPEN_FILES_:
        _, file = _OPEN_FILES_.popitem()
        file.close()


def _get_dataset(file_path, dataset_path, field_name):
    _raise_error_if_file_not_found(file_path)
    if file_path not in _OPEN_FILES_:
        _OPEN_FILES_[file_path] = h5py.File(file_path)
    ds = _OPEN_FILES_[file_path][dataset_path]
    _raise_error_if_not_dataset(ds)
    return ds.fields(field_name) if field_name else ds


def _get_ds_info(file_path, dataset_path, field_name):
    _raise_error_if_file_not_found(file_path)
    with h5py.File(file_path) as file:
        ds = file[dataset_path]
        _raise_error_if_not_dataset(ds)
        return {"shape" : ds.shape, 
                "dtype" : ds.dtype[field_name] if field_name else ds.dtype}


def _raise_error_if_file_not_found(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError


def _raise_error_if_not_dataset(obj):
    if not isinstance(obj, h5py.Dataset):
        raise NotDatasetError
