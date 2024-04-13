import json
import pickle as pkl

JSON_PATH = "/ghome/group01/MCV-C5-G1/Week5/new_captions.json"
TRIPLETS_PATH = "/ghome/group01/MCV-C5-G1/Week5/pickles/triplets/"

def append_new_triplets():
	with open(TRIPLETS_PATH + "triplets.pkl", "rb") as f:
		triplets = pkl.load(f)
	
	with open(JSON_PATH, "r") as f:
		new_captions = json.load(f)

	for caption in new_captions:
		triplets.append((caption["caption"], "positives/train_" + str(caption["id"]), "negatives/train_" + str(caption["id"])))

	with open(TRIPLETS_PATH + "triplets_final.pkl", "wb") as f:
		pkl.dump(triplets, f)
  
if __name__ == "__main__":
	append_new_triplets()
