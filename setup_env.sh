THIS_DIR="$(cd "$(dirname "$0")"; pwd)"
LIB_DIR=$THIS_DIR/lib

PATH_EXTENSION_="$THIS_DIR:$LIB_DIR"
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$PATH_EXTENSION_"
else
    export PYTHONPATH="$PATH_EXTENSION_:$PYTHONPATH"
fi
