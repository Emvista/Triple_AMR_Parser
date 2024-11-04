# built-in imports
import datetime
import contextlib
import os
import logging
# third party imports
import wikipedia
from pathlib import Path

logger = logging.getLogger()
logfmt = '%(asctime)s - %(levelname)s - \t%(message)s'
logging.basicConfig(format=logfmt, datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)


@contextlib.contextmanager
def new_cd(x):
    d = os.getcwd()

    # This could raise an exception, but it's probably
    # best to let it propagate and let the caller
    # deal with it, since they requested x
    os.chdir(x)

    try:
        yield

    finally:
        # This could also raise an exception, but you *really*
        # aren't equipped to figure out what went wrong if the
        # old working directory can't be restored.
        os.chdir(d)


def fill_empty_line(input_file: Path) -> Path:
    # Define the character to fill empty lines
    fill_char = '#'
    output_file = Path(input_file.as_posix() + '.no_empty_lines')
    # Open the original file in read mode and a new file in write mode
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Read the file line by line
        for line in infile:
            # Check if the line is empty (after stripping any whitespace)
            if line.strip() == '':
                # If empty, replace with the fill character and add a newline
                outfile.write(fill_char + '\n')
            else:
                # If not empty, write the line as is
                outfile.write(line)
    print(f"File processing complete. Check {output_file} for results.")
    return output_file


def get_latest_checkpoint(output_dir):

    # Find all subdirectories starting with 'checkpoint_'
    checkpoint_dirs = [d for d in output_dir.glob('checkpoint-*') if d.is_dir()]

    # Sort directories by creation time, newest first
    checkpoint_dirs.sort(key=lambda d: d.stat().st_ctime, reverse=True)

    # Get the latest created directory
    latest_checkpoint_dir = checkpoint_dirs[0] if checkpoint_dirs else None

    # Print the latest checkpoint directory
    print(f"The latest checkpoint directory is: {latest_checkpoint_dir}")
    return latest_checkpoint_dir

def wiki_connection_ok():
    wikipedia.set_rate_limiting(rate_limit=True, min_wait=datetime.timedelta(seconds=10))
    try:
        wikipedia.random(pages=1)
        print("Connection to wiki API is OK")
        return True
    except Exception as e:
        print(f"Failed to connect to wiki API: {e}")
        return False


if __name__ == '__main__':
    wiki_connection_ok()