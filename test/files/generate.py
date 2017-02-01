#!/usr/bin/env python3

import glob

fs = glob.glob("xio_expected_results/*.txt")

include_file = "#include <string>\n\n"

for f in fs:
	with open(f) as ff:
		ctn = ff.read()
		n = f.split("/")[1]
		include_file += "static std::string {} = R\"xio({})xio\";\n\n\n".format(n[:-4], ctn)

with open("xio_expected_results.hpp", "w+") as fo:
	fo.write(include_file)