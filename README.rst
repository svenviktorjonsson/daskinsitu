WHY

I created this project to make it easy to create arrays that can be managed
symbolically and Dask Arrays are perfect for this. I was concerned about having
a lot of files open while using the arrays and this module only openes file when
actual data is required. This is why I named the project daskinsitu.

HOW

I used dask.from_delayed with a function that opens the file only when the
compute method is executed on your dask arrays. By including a conveniance
method called compute in the module you don't have to worry about closing the
files afterwords.

WHAT

The modules includes the following public functions:

from_h5dataset:
    Used to get a dask array from a h5-file.

from_h5group:
    Used to get a dictionary of all the datasets in group within a h5-file.

close_open_files:
    Used to close open files. Use this if you apply the dask.compute on your
    dask arrays in order to close the file.
