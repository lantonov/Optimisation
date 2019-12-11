from subprocess import Popen, PIPE
import sys


# Path to the cutechess-cli executable.
# On Windows this should point to cutechess-cli.exe
cutechess_cli_path = 'C:\\Cutechess\\cutechess-cli'

# The engine whose parameters will be optimized
engine = 'name=test cmd=C:\\msys2\\home\\lanto\\nullmove-tune\\stockfish.exe'

# Format for the commands that are sent to the engine to
# set the parameter values. When the command is sent,
# {name} will be replaced with the parameter name and {value}
# with the parameter value.
engine_param_cmd = 'option.{name}={value}'

# A pool of opponents for the engine. The opponent will be
# chosen based on the seed sent by CLOP.
opponent = 'name=base cmd=C:\\msys2\\home\\lanto\\nullmove-tune\\stockfish.exe'

# Additional cutechess-cli options, eg. time control and opening book
options = '-each proto=uci tc=10+0.1 option.Hash=16 -rounds 20 -repeat -openings file=C:\\Cutechess\\2moves_v1.pgn format=pgn order=random -draw movenumber=34 movecount=8 score=20 -resign movecount=3 score=400 -pgnout results.pgn -concurrency 2'

def main(variables, trial=None):
    fcp = engine
    scp = opponent
    # Parse the parameters that should be optimized
    for name in variables:
        # Make sure the parameter value is numeric
        try:
            float(variables[name])
        except ValueError:
            sys.stderr.write('invalid value for parameter %s: %s\n' % (argv[i], argv[i + 1]))
            return 2

        initstr = engine_param_cmd.format(name = name, value = variables[name])
        fcp += ' "%s"' % initstr

    cutechess_args = '-engine %s -engine %s %s' % (fcp, scp, options)
    command = '%s %s' % (cutechess_cli_path, cutechess_args)

    # Run cutechess-cli and wait for it to finish
    process = Popen(command, shell = True, stdout = PIPE)
    output = process.communicate()[0]
    if process.returncode != 0:
        sys.stderr.write('failed to execute command: %s\n' % command)
        return 2

    # Convert Cutechess-cli's result into W/L/D
    score = []
    for line in output.decode("utf-8").splitlines():
        if line.startswith('Finished game'):
            if line.find(": 1-0") != -1:
              if line.find("test vs base") != -1:
                score.append('w')
              if line.find("base vs test") != -1:
                score.append('l')
            elif line.find(": 0-1") != -1:
              if line.find("test vs base") != -1:
                score.append('l')
              if line.find("base vs test") != -1:
                score.append('w')
            elif line.find(": 1/2-1/2") != -1:
              score.append('d')
            else:
                sys.stderr.write('the game did not terminate properly\n')
                return 2
#    sys.stdout.write(str(score))
    return score

if __name__ == "__main__":
    sys.exit(main(variables))
