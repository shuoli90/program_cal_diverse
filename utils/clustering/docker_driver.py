


import sys
import glob 
import os
import subprocess
import traceback

if __name__ == '__main__':
    ## read in tc_dir as first argument
    tc_dir = sys.argv[1]
    timeout = int(sys.argv[2])
    verbose = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    n_test_cases = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    open_ended = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else False
    if verbose:
        print(f"tc_dir: {tc_dir}, timeout: {timeout}, verbose: {verbose}, n_test_cases: {n_test_cases}, open_ended: {open_ended}")
    
    # then get all input.*.txt files in tc_dir
    input_files = glob.glob(os.path.join(tc_dir, 'input.*.txt'))
    # order 
    input_files = sorted(input_files)
    soln_printed = False
    already_errored = False
    
    # for each input_file, run os.path.join(tc_dir, soln.py) < input_file, use subprocess and feed in 
    for i, input_file in enumerate(input_files):
        output_file = input_file.replace('input', 'output') # input.56.txt -> output.56.txt
        soln_file = os.path.join(tc_dir, 'soln.py')
        # run soln.py < input_file > output_file
        # if syntax error or runtime error, write to output.*.txt
        # else write to output.*.txt
        # if verbose: 
        #     if open_ended:
        #         print(f'Running {soln_file} {input_file} > {output_file}')
        #     else: 
        #         print(f'Running {soln_file} < {input_file} > {output_file}')
        
        if already_errored: 
            with open(output_file, 'w') as f:
                # f.write("Timeout")
                f.write("Error")
            continue
        
        try: 
            if open_ended:
                p = subprocess.run(['python', soln_file, input_file], stdout=open(output_file, 'w'), stderr=subprocess.PIPE, timeout=timeout)
            else: 
                p = subprocess.run(['python', soln_file], stdin=open(input_file, 'r'), stdout=open(output_file, 'w'), stderr=subprocess.PIPE, timeout=timeout)
            if p.returncode != 0:
                err = p.stderr.decode('utf-8')
                if verbose:
                    print(f'Error running {soln_file} < {input_file} > {output_file}')
                    print(err)
                    if not soln_printed:
                        with open(soln_file, 'r') as f:
                            print(f.read())
                            soln_printed = True
                if "SyntaxError" in err: # maybe we need to catch import error, or other errors
                    already_errored = True
                    with open(output_file, 'w') as f:
                        f.write("Syntax Error")
                else:
                    already_errored = True
                    with open(output_file, 'w') as f:
                        f.write("Runtime Error")
                        # f.write(err)
                        
        except subprocess.TimeoutExpired:
            if verbose:
                print(f'Timeout running {soln_file} < {input_file} > {output_file}')
            with open(output_file, 'w') as f:
                f.write("Timeout")
            already_errored = True
                
        except Exception as e:
            if verbose:
                print(f'Error running {soln_file} < {input_file} > {output_file}')
            traceback.print_exc()
            raise e
        
        else:
            if verbose:
                print(f'Finished running {soln_file} < {input_file} > {output_file}')
                
    # for each input_file, run os.path.join(tc_dir, soln.py) < input_file, use subprocess and feed in
        
        
    # capture each output and write to output.*.txt
    # if syntax error or runtime error, write to output.*.txt
    
    