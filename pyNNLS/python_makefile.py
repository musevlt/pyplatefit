
fortran_source = '''

'''


def compile_fortran(source, module_name, extra_args=''):
    """
        Jan 15, 2019, Madusha Gunawardhana (gunawardhana@strw.leidenuniv.nl).
        Python routine to compile the fortran NNLS routines included in IDL PLATEFIT package.
    """

    import os
    import tempfile
    import sys
    import numpy.f2py  # just to check if it is presents
    from numpy.distutils.exec_command import exec_command

    args = ' -c -m {} {} {} {} {}'.format(module_name,
                                'mnbrak.f90', 'nnls_burst.f90', 'dbrent.f90', 'fit_cont_nnls.f90')
    # args = ' -c -m {} {}'.format(module_name,
    #                              'average.f90')
    command = '"{}" -c "import numpy.f2py as f2py;f2py.main()" {}'.format(sys.executable, args)
    status, output = exec_command(command)

    return status, output, command



status, output, command = compile_fortran(fortran_source, 'nnls_burst_python0',
                                          extra_args=" ")
# status, output, command = compile_fortran(fortran_source, 'average22',
#                                           extra_args="--f90flags='-fopenmp' -lgomp")

print(status)
if status == 0:
    print('The *.so appears to be compiled correctly')
else:
    print('Something went wrong with the compilation')
