THIS_DIR="$(cd "$(dirname "$0")"; pwd)"

if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$THIS_DIR"
else
    export PYTHONPATH="$THIS_DIR:$PYTHONPATH"
fi

