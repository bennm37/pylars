for f in ./scripts/*.py
do
    echo "running $f"
    python "$f"
done
for f in ./scripts/*/*.py
do
    echo "running $f"
    python "$f"
done