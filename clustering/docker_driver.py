


import sys
import glob 
import os
import subprocess

if __name__ == '__main__':
    ## read in tc_dir as first argument
    tc_dir = sys.argv[1]
    timeout = sys.argv[2]
    verbose = sys.argv[3] if len(sys.argv) > 3 else False
    
    # then get all input.*.txt files in tc_dir
    input_files = glob.glob(os.path.join(tc_dir, 'input.*.txt'))
    
    # for each input_file, run os.path.join(tc_dir, soln.py) < input_file, use subprocess and feed in 
    for i, input_file in enumerate(input_files):
        output_file = input_file.replace('input', 'output')
        soln_file = os.path.join(tc_dir, 'soln.py')
        # run soln.py < input_file > output_file
        # if syntax error or runtime error, write to output.*.txt
        # else write to output.*.txt
        if verbose:
            print(f'Running {soln_file} < {input_file} > {output_file}')
        p = subprocess.run(['python', soln_file], stdin=open(input_file, 'r'), stdout=open(output_file, 'w'), stderr=subprocess.PIPE, timeout=timeout)
        if p.returncode != 0:
            err = p.stderr.decode('utf-8')
            if verbose:
                print(f'Error running {soln_file} < {input_file} > {output_file}')
                print(err)
            if "SyntaxError" in err:
                with open(output_file, 'w') as f:
                    f.write("Syntax Error")
            else:
                with open(output_file, 'w') as f:
                    f.write("Runtime Error")
        else:
            if verbose:
                print(f'Finished running {soln_file} < {input_file} > {output_file}')
                
    # for each input_file, run os.path.join(tc_dir, soln.py) < input_file, use subprocess and feed in
        
        
    # capture each output and write to output.*.txt
    # if syntax error or runtime error, write to output.*.txt
    
    