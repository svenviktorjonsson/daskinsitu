from daskinsitu import __version__
import pytest
import dask.array as da
import dask
import h5py

import daskinsitu.daskinsitu as dais

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
    return dais.from_h5dataset("none","none", shape=(), dtype=int)


@pytest.fixture
def valid_array():
    return dais.from_h5dataset(file_path, ds_path)


@pytest.fixture
def valid_field():
    return dais.from_h5dataset(file_path, ds_path, field_name)


def test_from_h5dataset(valid_array):
    assert isinstance(valid_array, da.Array)


def test_compute_array_with_invalid_file_name_raises_no_errors(invalid_array):
    assert isinstance(invalid_array, da.Array)


def test_raises_error_when_using_invalid_file_without_info():
    with pytest.raises(FileNotFoundError):
        dais.from_h5dataset("none","none")


def test_raises_error_when_computing_with_invalid_file(invalid_array):
    with pytest.raises(FileNotFoundError):
        dais.compute(invalid_array)


def test_from_h5group_with_wrong_path_raises_error():
    with pytest.raises(FileNotFoundError):
        dais.from_h5group("wrong path", "none")


def test_from_h5dataset_with_group_path_and_no_info_raises_error():
    with pytest.raises(dais.NotDatasetError):
        dais.from_h5dataset(file_path,gr_path)


def test_from_h5group_with_big_ds_path_raises_error():
    with pytest.raises(dais.NotGroupError):
        dais.from_h5group(file_path, ds_path)


def test_from_h5group_returns_dict():
    group = dais.from_h5group(file_path, gr_path)
    assert isinstance(group, dict)
    assert ds_name in group
    assert isinstance(group[ds_name],da.Array)


def test_no_open_files_before_and_after_compute(valid_array):
    assert not dais._OPEN_FILES_
    dais.compute(valid_array)
    assert not dais._OPEN_FILES_


def test_close_open_files_after_dask_compute(valid_array):
    assert not dais._OPEN_FILES_
    dask.compute(valid_array)
    assert dais._OPEN_FILES_
    dais.close_open_files()
    assert not dais._OPEN_FILES_

