#!/bin/bash

# This bash script: 
#   - Creates a directory for text files (e.g. an audit file, lists of images) to be saved to
#   - Create directories for brightened grey and brightened colour versions of the images
#   - Create test directories for the original, brightened grey and brightened colour datasets
#   - Make lists of images for each letter (randomly selecting 20% of the iamges) that will be moved to the test directories 
#   - Run a python script that will create brightened grey and colour versions of the images and save the dataframes of the image arrays to .csv files 
#   - Move the images in the test lists to their respective test directories 

# The variables:
# $1 = name of a directory in ...alphabet_asl/asl_list/ to store all the lists for this attempt e.g. 17mar2020
# $2 = name of an audit txt file  e.g. 17mar2020_audit.txt
# $3 = the textfile name to save the classes to  e.g. asl_classes.txt

# Create a directory to hold future lists that will be iterated through
mkdir -p "C:/Users/main/Desktop/alphabet_asl/asl_list/$1"
# Sanity verbose
echo "A new directory 'Desktop/alphabet_asl/asl_list/$1' was created."
echo "A new directory 'Desktop/alphabet_asl/asl_list/$1' was created." >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

# Creating a list of the class filenames to iterate through
ls "C:/Users/main/Desktop/alphabet_asl/asl_alphabet_train/training_set" > "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$3"
# Sanity verbose
echo "A list of classes was saved to 'asl_list/$1/$3'."
echo "A list of classes was saved to 'asl_list/$1/$3'." >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

# Setting deliminator
IFS="\n"

# Iterate through every class listed to create directories and make a list of random images to be moved to the test directories
while read letter
do 
    # Create a class directory in both the train and test directories
    mkdir -p C:/Users/main/Desktop/alphabet_asl/asl_alphabet_train/{training_set/"$letter",test_set/"$letter"}
    mkdir -p C:/Users/main/Desktop/alphabet_asl/brighten_only/{training_set/"$letter",test_set/"$letter"}
    mkdir -p C:/Users/main/Desktop/alphabet_asl/bright_grey/{training_set/"$letter",test_set/"$letter"}
    # Sanity verbose
    echo "Directory '$letter/' was created in 'training_set/' and 'test_set/'."
    echo "Directory '$letter/' was created in 'training_set/' and 'test_set/'." >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

    # A count of training images in that class
    num_images=$(ls -1q "C:/Users/main/Desktop/alphabet_asl/asl_alphabet_train/training_set/$letter" | wc -l)
    # Calculate 20% of num_images
    test_size=$(expr $num_images / 5)
  
    # Saving the list of random images to a text file
    ls "C:/Users/main/Desktop/alphabet_asl/asl_alphabet_train/training_set/$letter" | sort -R | tail -$test_size > "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/test_$letter.txt"
    # Sanity verbose 
    echo "The 'asl_list/$1/test_$letter.txt' was created."
    echo "The 'asl_list/$1/test_$letter.txt' was created." >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

    # Also saving the test images with prefices 'bright_' and 'grey_' that will later be iterated through in the shell
    sed -e 's/^/bright_/' "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/test_$letter.txt" > "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/bright_test_$letter.txt"
    sed -e 's/^/grey_/' "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/test_$letter.txt" > "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/grey_test_$letter.txt"
    # Sanity verbose 
    echo "'asl_list/$1/bright_test_$letter.txt grey_test_$letter.txt' were created."
    echo "'asl_list/$1/bright_test_$letter.txt grey_test_$letter.txt' were created." >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

done < "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$3"

# Update
echo "Moving on to brightening the images..."
echo "Moving on to brightening the images..." >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

# Execute python script to created brightened images (grey and colour versions) and dataframes for future MLClassifiers
python "C:/Users/main/Desktop/alphabet_asl/bright_images.py" "$1" "$3" asl_grey.csv asl_colour.csv asl_original.csv

# Update 
echo "Brightened Grey and Colour Images and their dataframes (in .../asl_list/$1/) have been created."
echo "Brightened Grey and Colour Images and their dataframes (in .../asl_list/$1/) have been created." >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

# Iterate through each class to make test sets 
while read letter
do
    # Move original training images in 'test_{letter}.txt' to asl_alphabet_train/test_set/
    while read image 
    do
        # Move the images listed in testlist to test_set in designated class
        mv "C:/Users/main/Desktop/alphabet_asl/asl_alphabet_train/training_set/$letter/$image" "C:/Users/main/Desktop/alphabet_asl/asl_alphabet_train/test_set/$letter/"

        # Sanity verbose
        echo "$image was moved to test_set/$letter"
        echo "$image was moved to test_set/$letter" >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

    done < "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/test_$letter.txt"

    # Move brightened, grey images in 'grey_test_{letter}.txt' to bright_grey/test_set/
    while read grey_img 
    do
        # Move the images listed in testlist to test_set in designated class
        mv "C:/Users/main/Desktop/alphabet_asl/bright_grey/training_set/$letter/$grey_img" "C:/Users/main/Desktop/alphabet_asl/bright_grey/test_set/$letter/"

        # Sanity verbose
        echo "$grey_img was moved to test_set/$letter"
        echo "$grey_img was moved to test_set/$letter" >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

    done < "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/grey_test_$letter.txt"

    # Move brightened, colour images in 'bright_test_{letter}.txt' to brighten_only/test_set/ 
    while read colour_img 
    do
        # Move the images listed in testlist to test_set in designated class
        mv "C:/Users/main/Desktop/alphabet_asl/brighten_only/training_set/$letter/$colour_img" "C:/Users/main/Desktop/alphabet_asl/brighten_only/test_set/$letter/"

        # Sanity verbose
        echo "$colour_img was moved to test_set/$letter"
        echo "$colour_img was moved to test_set/$letter" >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

    done < "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/bright_test_$letter.txt"

    # Update
    echo "Class {$letter} test sets made for original, brightened grey and brightened colour images."
    echo "Class {$letter} test sets made for original, brightened grey and brightened colour images." >> "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$2"

done < "C:/Users/main/Desktop/alphabet_asl/asl_list/$1/$3"