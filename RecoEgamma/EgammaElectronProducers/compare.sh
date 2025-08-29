
other="/afs/hep.wisc.edu/home/astrel/work/Debug/CMSSW_14_1_0_pre7/src/RecoEgamma/EgammaElectronProducers/python"

find . -maxdepth 1 -type f -name '*.py' -print0 |
while IFS= read -r -d '' f; do
  base=$(basename "$f")
  target="$other/$base"
  if [[ -e "$target" ]]; then
    # Show diffs (unified). Remove -u if you want plain diff.
    diff -u -- "$f" "$target"
  else
    echo "Missing in develop: $base"
  fi
done





