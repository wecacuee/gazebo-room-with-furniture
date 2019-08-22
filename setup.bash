scriptname="${BASH_SOURCE[0]}"
scriptdir="$(dirname "$scriptname")"
setupdir="$(cd "$scriptdir" && pwd)"
export GAZEBO_MODEL_PATH="$setupdir:$GAZEBO_MODEL_PATH"
