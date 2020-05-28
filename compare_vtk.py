
import argparse
import vtk
import pytest
import warnings
from vtk.numpy_interface import dataset_adapter as dsa


def compare_vtk(*, reference_filename, test_filename):

    warnings.formatwarning = warning_on_one_line

    wrapper_1 = get_vtk_wrapper(filename=reference_filename)
    wrapper_2 = get_vtk_wrapper(filename=test_filename)

    keys_1 = wrapper_1.PointData.keys()
    keys_2 = wrapper_2.PointData.keys()

    print("""==============================
 Checking for missing data
==============================""")
    common_keys = []
    missing_data = False
    for k in keys_1:
        if k in keys_2:
            common_keys.append(k)
        else:
            missing_data = True
            warnings.warn(f"The key, {k}, is missing from {test_filename}", RuntimeWarning)

    if not missing_data:
        print("No missing data.")

    print("""\n==============================
 Comparing Common Data
==============================""")

    data_match = True
    for k in common_keys:
        are_approx_equal = (wrapper_1.PointData[k] - wrapper_2.PointData[k] == pytest.approx(0.0))
        if not are_approx_equal:
            data_match = False
            warnings.warn(f"The data corresponding to {k} do not match!")
    
    if data_match:
        print("All data match.")

def get_vtk_wrapper(*, filename):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.ReadAllScalarsOn()
    reader.Update()

    return dsa.WrapDataObject(reader.GetOutput())

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return '%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)

def main():
    parser = argparse.ArgumentParser(description='Compare two .vtk files')
    parser.add_argument('reference_filename', metavar='f1', type=str,
                        help='reference file for comparison')
    parser.add_argument('test_filename', metavar='f2', type=str,
                        help='file to be tested for equivalence')

    args = vars(parser.parse_args())
    compare_vtk(**args)

if __name__=="__main__":
    main()