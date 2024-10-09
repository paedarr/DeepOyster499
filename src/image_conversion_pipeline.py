import PIL
from PIL import Image, UnidentifiedImageError
import numpy as np
import os, sys
import csv
from pathlib import Path
from sys import platform # for os check
import pillow_heif
import pillow_heif.HeifImagePlugin
from pillow_heif import register_heif_opener
# from metadata_processing import add_metadata <- separate file with metadata stuff (add later)
# pip install pillow-heif , !please refer to the pillow-heif documentation for installation help:
# https://pillow-heif.readthedocs.io/en/latest/installation.html

def main():

    register_heif_opener() # to use the pillow-heif plugin


    data_array = []
    photo_file_array = []
    # userFilename = input("Please enter filename: ")
    with open('mud1.csv', mode ='r')as file:  # ADD PATHLIB FUNCTIONALITY HERE?
        csvFile = csv.reader(file)
        firstPass = True
        for lines in csvFile:
            if (firstPass):
                placeholder = lines
                firstPass = False
                continue
            startText = lines[0]
            newText = startText.replace(" ","") # base csv file may have spaces, remove these for file processing
            data_array.append(lines[1])
            
    print("\n")
    print("*The Image Folder must be in same directory as this code*\n")
    subfolder = input("Please enter image folder name: ")
    subfolder = Path(__file__).with_name(subfolder) # Python 3.4 pathlib module to allow for reliably open file in same directory


    folder_size = len(os.listdir(subfolder))
    for photo in os.listdir(subfolder):
        photo_file_array.append(photo)

    working_directory = os.getcwd()
    print(working_directory)
    print(subfolder)

    files = [f for f in os.listdir(subfolder) if f.endswith('.HEIC') or f.endswith('.HEIF') or f.endswith('') and f != '.DS_Store' and f != 'output']

    if not os.path.exists(os.path.join(subfolder, 'output')): # checks if output folder exists, if not, make it
        os.makedirs(os.path.join(subfolder, 'output'))

    index = 0
    output_path = ""
    for filename in files:
        image = Image
        if platform == "linux" or platform == "linux2":
            # linux - may not work, what filepath format does it use?
            subfolder = str(subfolder) 
            path_str = subfolder + "/" + filename
            print("Grabbing Image At > " + path_str)
            imgfile_orig = pillow_heif.read_heif(path_str)
            image = imgfile_orig.to_pillow()
            output_path = str(subfolder) + "/" + "output/"
        elif platform == "darwin":
            # mac os
            subfolder = str(subfolder) 
            path_str = subfolder + "/" + filename
            print("Grabbing Image At > " + path_str)
            imgfile_orig = pillow_heif.read_heif(path_str)
            image = imgfile_orig.to_pillow()
            output_path = str(subfolder) + "/" + "output/"
            # image = Image.open(Path(pathstr)) <- keep this for original PIL functionality (if desired)
        else:
            # assume windows (win32)
            subfolder = str(subfolder)
            path_str = subfolder.replace("/", "\\")
            imgfile_orig = pillow_heif.read_heif(path_str)
            image = imgfile_orig.to_pillow()
            output_path = str(subfolder) + "\\" + "output\\"
        image.convert('RGB').save(os.path.join(subfolder, 'output', os.path.splitext(filename)[0] + '.jpg'))

    
    # 0-------- adding metadata to all images in 'output' --------0
    # output_files = [f for f in os.listdir(output_path) if f.endswith('.jpg') and f != '.DS_Store' and f != 'output']
    # i = 0
    # for items in output_files:
    #     path_str = output_path + str(items)
    #     print(path_str)
    #     add_metadata(path_str, data_array[index])
    #     i += 0
   
if __name__ == "__main__":
    main()