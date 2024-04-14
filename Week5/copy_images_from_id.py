import os
import shutil
import numpy as np
import cv2

OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week5/retrieval"
res = { 'A commercial airplane flies through the sky to its destination. ': [111830, 14151, 403096, 506458, 356293], 'The front view of the stove built within the counter. ': [506458, 403096, 309022, 34826, 459921], 'A lot of spectators  watch a motorcade on Washington D.C. ': [208844, 141671, 560620, 254161, 329139], 'A person holding upright an older motorcycle that looks to be retrofitted.': [450762, 506458, 403096, 556083, 208844], 'A cat standing on the toilet bowl seat': [274657, 403096, 506458, 479384, 434958], 'A bathroom with dark wooden fixtures  and shelving': [506458, 459921, 201120, 309022, 302026], 'a bathroom with a sink and a toilet near a scale': [309022, 506458, 521495, 184359, 403096], 'A commercial plane on the strip to take off.': [111830, 506458, 403096, 14151, 318778], 'The view of the shower, sink, and towel rack.': [356293, 318778, 375324, 293125, 293244], 'Several pieces of art and a painting on display': [403096, 506458, 150558, 309022, 247764], 'a gay parade with some lesbians on motorcycles': [208844, 403096, 377113, 556083, 382111], 'A fluffy cat sleeping on top of a red car.': [274657, 401037, 248395, 567603, 93893], 'a dog standing near a street sign on a dirt road': [403096, 506458, 474272, 565239, 496411], 'A group of women are describing a new setup for a building plan': [506458, 403096, 309022, 508962, 195645], 'Two people riding horses along  a trail': [506458, 403096, 76431, 450762, 308974], 'A counter top in a laundry room with a tile back splash': [506458, 403096, 309022, 293125, 34826], 'A man sitting on a toilet in front of the computer.': [506458, 274657, 403096, 354307, 365542]}
# Ensure to remove all the images and files from the output folder
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH)

TRAIN_PATH = "/ghome/group01/mcv/datasets/C5/COCO/val2014"
ids = [356293, 318778, 375324, 293125, 293244]

# Collect and resize images
max_height = 0
for key, value in res.items():
    ids = value
    images = []
    for id in ids:
        padded_id = str(id).zfill(6)  # Pad the id with zeros to ensure it has 6 digits
        filename = "COCO_val2014_000000" + padded_id + ".jpg"
        source_path = os.path.join(TRAIN_PATH, filename)
        image = cv2.imread(source_path)
        height, width = image.shape[:2]
        max_height = max(max_height, height)
        images.append(image)

    # Resize images to have the same height
    resized_images = [cv2.resize(img, (int(width * max_height / height), max_height)) for img in images]

    # Concatenate images horizontally
    concatenated_image = np.hstack(resized_images)

    # Save the concatenated image
    output_filename = key+".jpg"
    output_path = os.path.join(OUTPUT_PATH, output_filename)
    cv2.imwrite(output_path, concatenated_image)
