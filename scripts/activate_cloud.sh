# Source this file from the repository root in the cloud machine.
# Ansys client libraries and plotting need writable user data/cache directories.
export XDG_DATA_HOME="$PWD/.runtime/data"
export XDG_CACHE_HOME="$PWD/.runtime/cache"
export MPLCONFIGDIR="$PWD/.runtime/matplotlib"
mkdir -p "$XDG_DATA_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
export PATH="$PWD/.venv/bin:$PATH"
