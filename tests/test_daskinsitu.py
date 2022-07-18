from daskinsitu import __version__
import pytest
import dask.array as da
import dask
import time

import daskinsitu.daskinsitu as di

file_path = "tests/resources/test.h5"
gr_path = "/Radiation Pattern"
ds_path = "/Radiation Pattern/Theta"
structured_ds_path = "/Radiation Pattern/Data"
field_name = "real"
ds_name = "Frequency"


def test_version():
    assert __version__ == '0.1.0'


@pytest.fixture
def invalid_array():
    return di.from_h5dataset("none","none", shape=(), dtype=int)


@pytest.fixture
def valid_array():
    return di.from_h5dataset(file_path, ds_path)


@pytest.fixture
def valid_field():
    return di.from_h5dataset(file_path, structured_ds_path, field_name)


def test_from_h5dataset(valid_array):
    assert isinstance(valid_array, da.Array)


def test_compute_array_with_invalid_file_name_raises_no_errors(invalid_array):
    assert isinstance(invalid_array, da.Array)


def test_raises_error_when_using_invalid_file_without_info():
    with pytest.raises(FileNotFoundError):
        di.from_h5dataset("none","none")


def test_raises_error_when_computing_with_invalid_file(invalid_array):
    with pytest.raises(FileNotFoundError):
        di.compute(invalid_array)


def test_from_h5group_with_wrong_path_raises_error():
    with pytest.raises(FileNotFoundError):
        di.from_h5group("wrong path", "none")


def test_from_h5dataset_with_group_path_and_no_info_raises_error():
    with pytest.raises(di.NotDatasetError):
        di.from_h5dataset(file_path,gr_path)


def test_from_h5group_with_big_ds_path_raises_error():
    with pytest.raises(di.NotGroupError):
        di.from_h5group(file_path, ds_path)


def test_from_h5group_returns_dict():
    group = di.from_h5group(file_path, gr_path)
    assert isinstance(group, dict)
    assert ds_name in group
    assert isinstance(group[ds_name],da.Array)


def test_no_open_files_before_and_after_compute(valid_array):
    assert not di._OPEN_FILES_
    di.compute(valid_array)
    assert not di._OPEN_FILES_


def test_close_open_files_after_dask_compute(valid_array):
    assert not di._OPEN_FILES_
    dask.compute(valid_array)
    assert di._OPEN_FILES_
    di.close_open_files()
    assert not di._OPEN_FILES_


def test_accessing_datasubset_is_much_faster(valid_field):
    t0 = time.perf_counter()
    full, = di.compute(valid_field)
    t1 = time.perf_counter()
    partial, = di.compute(valid_field[0,0,0,:2,:2])
    t2 = time.perf_counter()
    assert t1-t0>50*(t2-t1)

