
import sys

def main():
    # Read in generated labels
    in_path = sys.argv[1]
    with open(in_path, "r") as fd:
        labels = [l.split(" ") for l in fd.read().split("\n") if l != '']
        labels = [(float(l[0]), float(l[1])) for l in labels]
    
    # Postprocess labels here
    out = [0, 0]
    out[0] = sum([i[0] for i in labels]) / len(labels)
    out[1] = sum([i[1] for i in labels]) / len(labels)
    labels = [out for _ in labels]

    # Write out to output file
    out_path = sys.argv[2]
    with open(out_path, "w") as fd:
        for l in labels:
            fd.write(str(l[0]) + " " + str(l[1]) + "\n")


if __name__ == "__main__":
    main()