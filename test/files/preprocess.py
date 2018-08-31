import glob
import os
import re
import sys
import numpy as np

my_path = os.path.dirname(os.path.realpath(__file__))
cppy_files = glob.glob(my_path + '/cppy_source/*.cppy')

dtype_map = {
    'bool': 'bool',
    'float32': 'float',
    'float64': 'double',
    'int32': 'int',
    'int64': 'long',
    'uint32': 'unsigned int',
    'uint64': 'unsigned long',
    'complex64': 'std::complex<float>',
    'complex128': 'std::complex<double>',
}

def get_xtype(arr, xtensor=False):
    if isinstance(arr, (int)):
        return "int"
    elif isinstance(arr, float):
        return "double"

    if xtensor:
        s = "xtensor<" + dtype_map[str(arr.dtype)] + ", " + str(arr.ndim) + ">"
    else:
        s = "xarray<" + dtype_map[str(arr.dtype)] + ">"
    return s

def get_cpp_initlist(arr, name):
    if isinstance(arr, (int, str, float)):
        return get_xtype(arr) + " " + name + " = " + str(arr) + ';'
    name = get_xtype(arr) + " " + name
    s = np.array2string(arr, separator=',', precision=16)
    s = s.replace('[', '{')
    s = s.replace(']', '}')
    s = s.replace('j', 'i')
    s += ';'
    s = s.replace("\n", "\n" + " " * (len(name) + 3))
    s = name + " = " + s
    return s



def translate_file(contents, f):
	current_vars = {}

	matches = re.findall(r"\/\*py.*?\*\/", contents, re.MULTILINE | re.DOTALL)

	def exec_comment(txt, upper_level=False):
		lines = txt.split('\n')
		if upper_level:
			txt = '\n'.join(["import numpy as np"] + [x.strip() for x in lines[1:-1]])
		locals_before = list(locals().keys())
		exec(txt, globals(), current_vars)
		current_vars.update(
			{x: val for x, val in locals().items() if x not in locals_before}
		)

	result_file = ""

	idx = 0
	lidx = 0
	for line in contents.split('\n'):
		if lidx == 8:
			f = os.path.split(f)[1]
			result_file += "// This file is generated from test/files/cppy_source/{} by preprocess.py!".format(f) + '\n\n'
		lstrip = line.lstrip()
		if lstrip.startswith("/*py"):
			exec_comment(matches[idx], True)
			idx += 1
		if lstrip.startswith("// py_"):
			indent_n = len(line) - len(lstrip)
			if '=' in lstrip:
				exec_comment(lstrip[6:])
				var = line.strip()[6:lstrip.index('=')].strip()
			else:
				var = line.strip()[6:]
			indent = line[:indent_n]
			init_list = get_cpp_initlist(current_vars[var], 'py_' + var)
			init_list = '\n'.join([indent + x for x in init_list.split('\n')])
			result_file += line + '\n'
			result_file += init_list + '\n'
		else:
			result_file += line + '\n'
		lidx += 1
	return result_file

print("::: PREPROCESSING :::\n")

for f in cppy_files:
	print(" - PROCESSING {}".format(f))

	global current_vars
	current_vars = {}  # reset

	with open(f) as fi:
		contents = fi.read()

	# reset global seed
	np.random.seed(42)
	result = translate_file(contents, f)
	f_result = os.path.split(f)[1]
	with open(my_path + "/../" + f_result[:-1], 'w+') as fo:
		fo.write(result)
	print("::: DONE :::")
	# print(result)
