# Author: Randa Yoga Saputra
# Version: 1
# Purpose: Used for manipulating files, folders, image data.
# Running:
# Augmenting herpes images - py -3.8 .\augment.py --class_dir "herpes" --tt_dir "herpes_v_scc"

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import argparse 

# 0) Globals
# Create an argument parser
argParser = argparse.ArgumentParser(
	description="Augment.py - Data augmenter and other file/ folder manipulations."
)
argParser.add_argument('--count', 
    type=int,
    default=10,
    help='[optional] Specifies the number of augmented files to derive from each image.  Default used is 10.')
argParser.add_argument('--equalize', 
    nargs=3, # dirName1, dirName2, L
    type=str,
    default=None,
    help='[optional] Name of files in new augmented directory.  Default used is class (label) names.')
argParser.add_argument('--fname', 
    type=str,
    default="",
    help='[optional] Name of files in new augmented directory.  Default used is class (label) names.')
argParser.add_argument('--saveto', 
    type=str,
    default="",
    help='[optional] Name of directory to save augmented image files to.  Default used is class (label) names + DEFAULT_SAVETO_SUFFIX.')  
argParser.add_argument('--tt_dir', 
    type=str,
    help='[herpes/ squamous cell carcinoma]: herpes_v_scc, [lesion/ nonlesion]: lesion_v_nonlesion, [lip/ tongue]: lip_v_tongue') 
argParser.add_argument('--class_dir', 
    type=str,
    help='[herpes]: herpes, [tongue]: tongue, [lips]: lip, [squamous cell carcinoma]: squamouscCellCarcinoma, [lesion]: lesion, [non lesion]: nonlesion') 
args = argParser.parse_args()
DEFAULT_SAVETO_SUFFIX = "_aug" # this will be appended to savetodirectory names if saveto is left unspecified.

# 1) Define an ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def rename_files(sub_dirs: list = ["Herpes_Simplex_Test"], 
	replace_term: str = "_x_x", replace_val: str = "", suffix: str = "", file_ext: str = ".png"):
	"""Rename all files in a directory based on several criteria.
	"""
	# subdirs: ['Apthous Ulcer', 'Herpes Simplex', 'Herpes Simplex_normalized', 'Squamous Cell Carcinoma']
	for dir in sub_dirs:
		for img in os.listdir(os.path.join(path, dir)):
			img_path = os.path.join(path, dir, img)
			os.rename(img_path, img_path.replace(replace_term, replace_val).replace(file_ext,"") + suffix + file_ext)

def augment(dirName: str, fname: str, save_to_dir: str, count: int = 20):
    """
    """
    if not os.path.exists(save_to_dir): # make save_to_directory if it doesn't exist
        os.mkdir(save_to_dir)
    for imgName in os.listdir(dirName):
        img = load_img(os.path.join(dirName, imgName))  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir=save_to_dir, save_prefix=fname, save_format='png'):
            i += 1
            if i > count:
                break  # otherwise the generator would loop indefinitely

def equalize(x1: int, x2: int, L: int, depth: int = 20, TOL = 0):
    """Make dataset cardinality equal.
    Algorithm to find integer solutions to the following:
    a * x1 = b * x2 | a, b > L
    """
    for k in range(10):
        for a in range(L, L + depth):
            if (a * (x1 + k)) % x2 <= TOL:
                print("b: ",(a * (x1+k)) / x2, f"{x1} -> {x1+k}, a: {a}")
                break
        else:
            continue
        break

def equalize2(x1: int, x2: int, L: int, a_max: int = 20, b_max: int = 20):
    """Make dataset cardinality equal.
    Algorithm to find integer solutions to the following:
    a * x1 = b * x2 | a, b > L
    """
    min_diff = float("inf")
    a_star, b_star = float("inf"), float("inf")

    for a in range(L, L + a_max):
        V1 = a * x1
        b =  (a * x1) // x2
        V2 = x2 * b
        if abs(V2 - V1) < min_diff and b <= b_max:
            min_diff = abs(V2 - V1)
            a_star, b_star = a, b 
            # print("a:",a,"b:",b,"V1:",V1,"V2:",V2,"diff:",min_diff)

    return (a_star, b_star)

def main():
    #dirName = "../data/traintest/traintest_lesion_v_nonlesion/lesion/"
    #fname = "herpes"
    #save_to_dir = "augmented_herpes"

    dirName = f"../data/traintest/traintest_{args.tt_dir}/{args.class_dir}/"
    fname = args.fname if args.fname != "" else args.class_dir
    save_to_dir = args.saveto if args.saveto != "" else args.class_dir + DEFAULT_SAVETO_SUFFIX
    
    print(f"dir: {dirName}")
    print(f"fname: {fname}")
    print(f"saveto: {save_to_dir}")
    print(f"count: {args.count}")

    if args.equalize:
        try:
            print("Initiating equalization")
            dir1, dir2, L = args.equalize
            a_star, b_star = equalize2(len(os.listdir(dir1)), len(os.listdir(dir2)), int(L), a_max = 30, b_max = 30)

            tt_dir1, class_dir1 = [i for i in dir1.split("/") if i.strip() != ""][-2:]
            tt_dir2, class_dir2 = [i for i in dir2.split("/") if i.strip() != ""][-2:]

            # using the optimal a_star/ b_star to augment datasets and make the cardinality as close as possible.
            augment(dir1, class_dir1, class_dir1 + DEFAULT_SAVETO_SUFFIX, count = a_star)
            augment(dir2, class_dir2, class_dir2 + DEFAULT_SAVETO_SUFFIX, count = b_star)
        except:
            print(
                "1.Check that you specified enough arguments, 2. they are in correct order, 3. paths are valid."
                )
    else:
        print("args.equalize left unspecified.")
        augment(dirName, fname, save_to_dir, count=args.count)

if True:
    if __name__ == "__main__": main()

