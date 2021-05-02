import argparse
import ast
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np

# Directory where XGBoost source code is located
PROJECT_BASE = '../XGBoost'
# Directory to write compiled object files to - make sure this directory exists
COMPILE_LOCATION = './compiled'
# Path to write the executable to - make sure all directories on the path exist
EXEC_PATH = './benchmark'


configs = {
	'config1': ["10", "10", "2", "100000", "1000000", "10000"],
	'config2': ["10", "10", "2", "100", "1000", "100"],
	'config3': ["10", "10", "2", "1000", "10000", "1000"],
	'config4': ["10", "10", "2", "100000", "1000000", "100000"]

}


def execute(executable_path, args, mode='tree'):
    """
	Executes the program with the given args and mode.
	returns an array containing the results
	"""

    print("2. Executing program...")
    result = []

    command_params = [executable_path, mode] + args
    print("Executing {}".format(' '.join(command_params)))
    myout = subprocess.Popen(command_params, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = myout.communicate()
    output = stdout.decode('ascii')

    for line in output.splitlines():
        stripped_line = line.rstrip()
        datapoint = ast.literal_eval(stripped_line)
        if datapoint is not None:
        	if mode=="all":
        		result.append({
	                'rows': int(datapoint['rows']),
	                'cols': int(datapoint['columns']),
	                'cycles_splitfinding': float(datapoint['cycles_splitfinding']),
	                'cycles_split':float(datapoint['cycles_split']),
	                'cycles_tree':float(datapoint['cycles_tree'])
	            })
        	else:

	            result.append({
	                'rows': int(datapoint['rows']),
	                'cols': int(datapoint['columns']),
	                'cycles': float(datapoint['cycles'])
	            })

    return result


def get_source_files(base_folder):
	"""
	Fetches all files we need to compile for the benchmark from CMakeList.txt
	"""
	file = os.path.join(base_folder, 'CMakeLists.txt')
	f = open(file, "r")
	line = [line.rstrip() for line in f.readlines() if line.startswith("add_executable(bench")][0]
	return re.search('^add_executable\(bench (.*)\)$', line).group(1).split(' ')

	re.search('^(\d+) rows, (\d+) columns: (\d+\.\d+) cycles$', s)


def compile(source_files, params):
	"""
	Compiles the program into an executable
	"""
	print("1. Starting compilation...")

	source_files = [file for file in source_files if file.endswith('.c')]
	object_files = []

	# Create object files
	for file in source_files:
		obj_filename = os.path.basename(file.rstrip('.c')) + '.o'
		command = ["gcc"] + params + ["-c", os.path.join(PROJECT_BASE, file)] + ["-o", os.path.join(COMPILE_LOCATION, obj_filename)]
		print("Compiling object file with {}".format(" ".join(command)))
		subprocess.run(command)
		object_files.append(obj_filename)

	# Create executable
	object_files_with_path = [os.path.join(COMPILE_LOCATION, file) for file in object_files]
	command = ["gcc"] + object_files_with_path + params + ["-o", EXEC_PATH]
	print("Linking object files with {}".format(" ".join(command)))
	subprocess.run(command)

	print()

def check_paths():
    """ Verifies all path constants exist """
    ok = True

    cwd = os.getcwd()
    if not cwd.endswith('performance'):
        print('ERROR: You are not in the working directory of python script!')
        ok = False

    if not os.path.isdir(PROJECT_BASE):
        print("directory PROJECT_BASE={} does not exist".format(PROJECT_BASE))
        ok = False

    if not os.path.isdir(COMPILE_LOCATION):
        print("directory COMPILE_LOCATION={} does not exist".format(COMPILE_LOCATION))
        ok = False

    return ok


def calculate_performance(cycles, flops):
	return (flops/cycles)

def getFlops(mode, sort_type, input_sizes):
	if (sort_type == 'm'):
		flops = [ 2*inp['n'] + inp['m']*17*inp['n'] + inp['m']*inp['n']*np.log2(inp['n']) for inp in input_sizes ] #mergesort
	elif sort_type == 'b':
		flops = [ 2*inp['n'] + inp['m']*17*inp['n'] + inp['m']*inp['n'] for inp in input_sizes ] #bucketsort
	elif sort_type == 'p':
		if mode=="all":
			flops = [2*inp['n'] + inp['m']*14*inp['n'] + inp['n']*np.log2(inp['n']) for inp in input_sizes]
		else:
			flops = [2*inp['n'] + inp['m']*14*inp['n'] for inp in input_sizes]
	return flops

def plot(x, perf,plot_perf,inputs_opt=None, perf_opt=None,labels_opt=None):

	from matplotlib.cm import get_cmap

	print("3. Generating plots...")

	if (inputs_opt is not None) & (labels_opt is None):
		labels_opt = []
		for i in range(len(inputs_opt)):
			labels_opt.append("Opt "+str(i+1))

	width = 500
	cmap = get_cmap("Set2")

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.set_prop_cycle(color=cmap.colors)
	ax.plot(x, perf, '-.', label='Baseline')

	if inputs_opt is not None:
		for i in range(len(inputs_opt)):
			ax.plot(inputs_opt[i],perf_opt[i],'-.',label=labels_opt[i])
	plt.xlabel('Number of rows')

	if plot_perf:
		plt.ylabel('P(n) [flops/cycle]', rotation=0, horizontalalignment='left')
		ax.yaxis.set_label_coords(-0.05, 1.02)
		plt.title("Performance of Split Finding Algorithm",loc='center',pad=20)

	else:
		ax.set_ylabel('Run time [cycles]', rotation=90, horizontalalignment='left')
		plt.title("Runtime of Split Finding Algorithm",loc='center')

	ax.legend()
	return plt


def main(mode, run=False, plot_perf=False, save_baseline=False, config_name=None):
	ok = check_paths()
	if ok is False:
		return 1

	fn = config_name+"_"+mode
	r = re.compile(fn+"(_\d)?"+"\.npz")
	existing_files = list(filter(r.match,os.listdir()))

	if run:
		if (len(existing_files)==0) or save_baseline:
			new_fn = fn+".npz"
		else:
			numbers = re.compile(fn+"_\d"+"\.npz")
			numbered_files = list(filter(numbers.match,existing_files))
			new_fn = fn+"_"+str(len(numbered_files)+1)+".npz"
		params = ["-O3", "-fno-tree-vectorize"]
		compile(get_source_files(PROJECT_BASE), params)
		results = execute(EXEC_PATH, configs[config_name], mode)
		input_sizes = np.array([{'n': obj['rows'] , 'm': obj['cols']} for obj in results])
		if mode=="all":
			cycles_split = np.array([obj['cycles_split'] for obj in results])
			cycles_splitfinding = np.array([obj['cycles_splitfinding'] for obj in results])
			cycles_tree = np.array([obj['cycles_tree'] for obj in results])
			#cycles = splitfinding_cycles+split_cycles+tree_cycles
			np.savez_compressed(new_fn,input_sizes,cycles_split,cycles_splitfinding,cycles_tree)
		else:
			cycles = np.array([obj['cycles'] for obj in results])
			np.savez_compressed(new_fn,input_sizes,cycles)
		existing_files.append(new_fn)

	input_rows=[]
	cycles=[]
	perf = []
	input_rows_opt=[]
	cycles_opt=[]
	labels_opt=[] #"Bucket Sort","Column Major","Presorted Data","Loop Unrolling","Eliminate Pointer Chasing"]
	perfs_opt=[]
	sorts = ['m','b','m']+['p']*(len(existing_files)-3)
	append_labels = (labels_opt==[])
	for file in sorted(existing_files):
		if file==(fn+".npz"):
			pkl = np.load(file,allow_pickle=True)
			input_sizes = pkl['arr_0']
			cycles = pkl['arr_1']
			if mode=="all":
				cycles = pkl['arr_1']+pkl['arr_2']+pkl['arr_3']

			input_rows = [obj['n'] for obj in input_sizes]
			flops = getFlops(mode,sorts[0],input_sizes)
			perf = calculate_performance(cycles,flops)
		else:
			pkl = np.load(file,allow_pickle=True)
			cycles_tmp = pkl['arr_1']

			if mode=="all":
				cycles_tmp = cycles_tmp+pkl['arr_2']+pkl['arr_3']

			input_sizes_tmp = pkl['arr_0']
			input_rows_opt.append([obj['n'] for obj in input_sizes_tmp])
			cycles_opt.append(cycles_tmp)
			opt_num = file.split("_")[-1].split(".")[0]

			if append_labels:
				labels_opt.append("Optimization "+opt_num)

			perfs_opt.append(calculate_performance(cycles_tmp,getFlops(mode,sorts[int(opt_num)],input_sizes_tmp)))

	if plot_perf:
		plt = plot(input_rows, perf,plot_perf,input_rows_opt,perfs_opt,labels_opt)
		plt.savefig("Performance.png")
	else:
		plt = plot(input_rows, cycles,plot_perf,input_rows_opt,cycles_opt,labels_opt)
		plt.savefig("Runtime.png")
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', help='Run XGBoost?', action='store_true')
	parser.add_argument('-p', help='Plot performance?', action='store_true')
	parser.add_argument('-b', help='Save as baseline?', action='store_true')
	parser.add_argument('-mode', type=str, default='splitfind')
	arg = parser.parse_args()
	assert(arg.mode in ['splitfind', 'split', 'tree', 'all'])
	main(arg.mode, arg.r, arg.p, arg.b, 'config4')











