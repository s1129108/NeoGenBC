import argparse
import numpy as np
import os
import pickle
import h5py

# ---------------------------
# Load embeddings based on file type
# ---------------------------
def loadData(path):
    if path.endswith(".npy"):
        Data = np.load(path)
        Data = np.squeeze(Data, axis=0)
        print("Loaded .npy:", Data.shape)

    elif path.endswith(".hdf5"):
        hdf5_file = h5py.File(path, "r")
        key = list(hdf5_file.keys())[0]
        Data = np.array(hdf5_file[key])
        print("Loaded .hdf5:", Data.shape)

    elif path.endswith(".esm2"):
        # ESM2 saved from pickle
        with open(path, "rb") as f:
            Data = pickle.load(f)
        print("Loaded .esm2 (pickle):", Data.shape)

    else:
        Data = np.loadtxt(path)
        print("Loaded .txt:", Data.shape)

    return Data


# ---------------------------
# Pad or truncate sequence
# ---------------------------
def get_series_feature(org_data, maxseq, length):
    out = np.zeros((maxseq, length), dtype=np.float32)

    seq_len = len(org_data)

    if seq_len < maxseq:
        out[:seq_len] = org_data
    else:
        out[:] = org_data[:maxseq]

    out = out.reshape(1, 1, maxseq, length)
    return out


# ---------------------------
# Save final dataset
# ---------------------------
def saveData(path, data):
    print("Saving:", data.shape)
    np.save(path, data)


# ---------------------------
# Main driver
# ---------------------------
def main(path_input, path_output, data_type, maxseq, length):

    files = os.listdir(path_input)
    results = []

    # Create epitope ID file
    epitope_id_path = os.path.join(os.path.dirname(path_output), "epitope_ID.txt")
    epitope_id_file = open(epitope_id_path, "w")

    for fname in files:
        if fname.endswith(data_type):
            fpath = os.path.join(path_input, fname)
            print("Processing:", fpath)

            # Extract ID (filename without extension)
            epitope_id = os.path.splitext(fname)[0]
            epitope_id_file.write(epitope_id + "\n")

            emb = loadData(fpath)
            feat = get_series_feature(emb, maxseq, length)
            results.append(feat)

    epitope_id_file.close()

    final = np.concatenate(results, axis=0)
    saveData(path_output, final)

    print(f"Epitope IDs saved to: {epitope_id_path}")


# ---------------------------
# CLI Entry
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--path_input", type=str, required=True)
    parser.add_argument("-out", "--path_output", type=str, required=True)
    parser.add_argument("-dt", "--data_type", type=str, required=True)
    parser.add_argument("-maxseq", "--max_sequence", type=int, default=0)
    args = parser.parse_args()

    if args.data_type == ".prottrans":
        length = 1024
    elif args.data_type == ".esm":
        length = 1280
    elif args.data_type == ".npy":
        length = 768
    else:
        length = 20

    main(args.path_input, args.path_output, args.data_type, args.max_sequence, length)
