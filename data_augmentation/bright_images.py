import pandas as pd 
import numpy as np 
import glob
import cv2
import sys

# This python script will use the above libraries to iterate through each image of each letter(class) to create brightened grey and colour copies. 
# OPTIONAL: Each image (64x64=4,096 array) will be saved as a row in dataframes (1. df for brightened grey images, 2. df for bright colour, 3. df for original images) which will be saved to .csv files. 

# sys.argv[1] will be the directory in ...alphabet_asl/asl_list/ that the lists were saved to in 'split_only_asl.sh' (variable $1)
# sys.argv[2] will be the filename of the text file that the list of classes was saved to in 'split_only_asl.sh' (variable $3)
## SEE split_only_asl.sh ##
# sys.argv[3] will be the csv file that the df of brightened grey images will be saved to e.g. asl_grey.csv
# sys.argv[4] will be the csv file that the df of brightened color arrays will be saved to e.g. asl_colours.csv
# sys.argv[5] will be the csv file that the df of original colour arrays will be saved to e.g. asl_original.csv

# To start creating a function to brighten the images (and normalizes the pixel values) 
def brightening(image, gamma=1.0):
    '''
    This will return the image (np.array values) with 
    altered brightness (first the value will be bound between 0 and 1 and then raised to the inverse of gamma) 
    
    INPUTS
    ----
    image: the image to alter the brightness of 
    gamma: the denominator of the exponent that each pixel of the image (np.array value)
    
    OUTPUTS
    -----
    image with adusted np.array values (adjusted brightness)
    '''
    
    assert isinstance(gamma, (float, int)), " 'gamma' must be numeric"
    
    return (image/255)**(1/gamma)

# Getting the list of classes created in 'split_only_ash.sh $3' to iterate through 
classes = np.genfromtxt(f'../asl_list/{sys.argv[1]}/{sys.argv[2]}', delimiter='\n', dtype='str')

# # Creating the csv files that will be appended to with each image iteration 
# # Creating the column names and saving to 'pixels'
# pixels = list(range(0,4096))
# pixels.append('Class')
# # Creating the dataframes to be written to csv's
# to_add_grey = pd.DataFrame(columns=pixels)
# to_add_colour = pd.DataFrame(columns=pixels)
# to_add_original = pd.DataFrame(columns=pixels)
# # Writing to csv
# to_add_grey.to_csv(f'../asl_list/{sys.argv[1]}/{sys.argv[3]}')
# to_add_colour.to_csv(f'../asl_list/{sys.argv[1]}/{sys.argv[4]}')
# to_add_original.to_csv(f'../asl_list/{sys.argv[1]}/{sys.argv[5]}')

# iterate through each class (each folder in ...asl_alphabet_train/training_set/)
for letter in classes:

    # A list of the paths for each image in that class
    image_paths = glob.glob(f'../asl_alphabet_train/training_set/{letter}/*')

    # Creating an empty list that image names will be appended to (e.g. C101 )
    image_names = []
    
    # iterate through each photo path to isolate the image file name
    for path in image_paths:

        # Remove path from string
        path_removed = path.replace(f'../asl_alphabet_train/training_set/{letter}\\','')
        # Remove filetype from string
        ftype_removed = path_removed.replace('.jpg','')
        # Append the remaining string to the list 'image_names'
        image_names.append(ftype_removed)

    # iterate through each image to create brightened grey and colour versions & append flattened, normalized image arrays to dataframes
    for image in image_names:

        # # Creating a dataframe to append the grey image arrays to 
        # df_grey_images = pd.DataFrame()
        # # Creating a dataframe to append brighter colour image arrays to 
        # df_bright_images = pd.DataFrame()
        # # Creating a dataframe to append original colour image arrays to
        # df_original_images = pd.DataFrame()

        # Read in the image as grayscale (1 channel)
        grey_img = cv2.imread(f'../asl_alphabet_train/training_set/{letter}\\{image}.jpg', cv2.IMREAD_GRAYSCALE)
        # Read in the same image unchanged (in colour with 3 channels (RGB))
        colour_img = cv2.imread(f'../asl_alphabet_train/training_set/{letter}\\{image}.jpg', cv2.IMREAD_UNCHANGED)

        # brighten and normalize the images 
        bright_grey_img = brightening(grey_img, gamma=1.8)
        bright_colour_img = brightening(colour_img, gamma=1.8)

        # # Resizing brightened images for dataframes 
        # bright_grey_img_rz = cv2.resize(bright_grey_img, (64, 64))
        # bright_colour_img_rz = cv2.resize(bright_colour_img, (64, 64))
        # colour_img_rz = cv2.resize(colour_img, (64, 64))

        # # Flattening the grey image to append to dataframe: df_grey_images 
        # flat_grey = pd.DataFrame(bright_grey_img_rz.reshape(1,-1))

        # # Flattening the RGB channels of the brighter colour image and appending each channel to df_bright_images as separate rows
        # flat_red = pd.DataFrame(bright_colour_img_rz[:,:,0].reshape(1,-1))
        # flat_green = pd.DataFrame(bright_colour_img_rz[:,:,1].reshape(1,-1))
        # flat_blue = pd.DataFrame(bright_colour_img_rz[:,:,2].reshape(1,-1))

        # # Flattening the RGB channels of the original image and appending each channel to df_original_images as separate rows
        # flat_red_og = pd.DataFrame(colour_img_rz[:,:,0].reshape(1,-1))
        # flat_green_og = pd.DataFrame(colour_img_rz[:,:,1].reshape(1,-1))
        # flat_blue_og = pd.DataFrame(colour_img_rz[:,:,2].reshape(1,-1))

        # # Adding a column with the class 
        # # Create the class column
        # add_class = pd.DataFrame(pd.Series(f'{letter}'), columns=['Class'])
        # # Add the class column to each flattened image channel
        # df_grey = pd.concat([flat_grey, add_class], axis=1)
        # df_red = pd.concat([flat_red, add_class], axis=1)
        # df_green = pd.concat([flat_green, add_class], axis=1)
        # df_blue = pd.concat([flat_blue, add_class], axis=1)
        # df_red_og = pd.concat([flat_red_og, add_class], axis=1)
        # df_green_og = pd.concat([flat_green_og, add_class], axis=1)
        # df_blue_og = pd.concat([flat_blue_og, add_class], axis=1)

        # # Append the images+class to the respective dataframes
        # df_grey_images = df_grey_images.append(df_grey, ignore_index=True)
        # df_bright_images = df_bright_images.append([df_red, df_green, df_blue], ignore_index=True)
        # df_original_images = df_original_images.append([df_red_og, df_green_og, df_blue_og], ignore_index=True)

        # # Appending the dataframe to the csv file every image (to not exceed memory limits and see where the script fails [if it does])
        # df_grey_images.to_csv(f'../asl_list/{sys.argv[1]}/{sys.argv[3]}', 
        # mode='a', header=False)
        # df_bright_images.to_csv(f'../asl_list/{sys.argv[1]}/{sys.argv[4]}', 
        # mode='a', header=False)
        # df_original_images.to_csv(f'../asl_list/{sys.argv[1]}/{sys.argv[5]}', 
        # mode='a', header=False)

        # Saving the transformed images (must reverse normalization by multiplying each pixel by 255)
        bool_status = cv2.imwrite(f'../bright_grey/training_set/{letter}/grey_{image}.jpg', 255*bright_grey_img)
        # Sanity verbose
        print(f'"grey_{image}.jpg" has been saved: {bool_status}')

        colour_status = cv2.imwrite(f'../brighten_only/training_set/{letter}/bright_{image}.jpg', 255*bright_colour_img)
        # Sanity verbose 
        print(f'"bright_{image}.jpg" has been saved: {colour_status}')
    
    # Sanity verbose
    print(f'Class {letter} completed.')

