import pickle

RESULTS = "./pickles/results/results_txt2img"  # Path to the pickle file
TXT = "./pickles/results/readable_results.txt"  # Path to the output text file

# Load the pickle file
with open(RESULTS, 'rb') as f:
    data = pickle.load(f)
i = 0
# Write all the data to a text file
with open(TXT, 'w') as f:
    for key, value in data.items():
        f.write(f"{i}. {key}: {value}\n\n")
        i += 1
        if i > 100:
            break
