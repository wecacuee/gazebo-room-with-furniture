function setup_env() {
    local scriptname="${BASH_SOURCE[0]}"
    local scriptdir="$(dirname "$scriptname")"
    local setupdir="$(cd "$scriptdir" && pwd)"

    {
        [ -f /usr/share/gazebo-9/setup.sh ] && source /usr/share/gazebo-9/setup.sh
        [ -f $setupdir/gazebo_models/setup.bash ] && source $setupdir/gazebo_models/setup.bash
        unset GAZEBO_MODEL_URI
    }

    export GAZEBO_MODEL_PATH="$setupdir/models:$GAZEBO_MODEL_PATH"
    [ -f $setupdir/envrc ] && source $setupdir/envrc
    if [ -z "$IKEA_MODELS_DATASET_DIR" ]; then
        echo "Please download set IKEA_MODELS_DATASET_DIR from http://ikea.csail.mit.edu/zip/IKEA_models.zip"
    fi
    export GAZEBO_RESOURCE_PATH=$IKEA_MODELS_DATASET_DIR:$GAZEBO_RESOURCE_PATH
}
setup_env
