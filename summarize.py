"""Print all benchmark metrics side by side, one column per concurrency level."""
import re

files = ["results/bench_c1.txt", "results/bench_c5.txt",
         "results/bench_c20.txt", "results/bench_c40.txt"]
rows, order = {}, []
for f in files:
    for line in open(f):
        m = re.match(r"^([A-Za-z][^:]+):\s+([\d.]+)\s*$", line)
        if m:
            label = m.group(1).strip()
            if label not in rows:
                rows[label] = {}
                order.append(label)
            rows[label][f] = m.group(2)

w = max(len(l) for l in order)
names = [f.split("bench_")[1][:-4] for f in files]
print("Metric".ljust(w), *[n.rjust(10) for n in names])
for label in order:
    print(label.ljust(w), *[rows[label].get(f, "").rjust(10) for f in files])
