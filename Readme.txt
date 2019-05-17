Major Project Submission

Title - 'A Sliding Window Approach for Malaria Detection in Thin Blood Film Images'
Name - Siddhanth Pillay
Roll No - 15IT129

---------------------------------------------------------------------------

Installations for the environment:

1) PyTorch: Install PyTorch-GPU for CUDA 9 using the following command:
pip3 install torch torchvision

2) OpenCV: Install OpenCV using the following command:
pip3 install opencv

3) Staintools: Install Staintools using the following commands:
- pip3 install spams
- pip3 install staintools

4) Scikit-Image: Install Scikit-Image using the following command:
pip3 install scikit-image

---------------------------------------------------------------------------

Steps to Run the Code:

1) To normalize the images using the various staing normalization methods, use the script 
'stain_normalizer.py'. By varying the name of the folder location where all the images are stored (modify line 15 in
code to set folder location where all images are present), one can obtain the stain normalized images for three techniques - 
Macenko, Reinhard, Vahadane

2) Store the stain normalized images in a folder called 'normalized images'. Use the file called 'patch_maker.py' to run the script
to create patches for training

3) Run the training script for either of the stain normalized technique - LeNet_thin_film_x (x = macenko/reinhard/vahadane)
This will store the model and it's weights in files called 'thin_film_LeNet.pt' and 'thin_film_LeNet_dict.pt', respectively

4) To generate the images for the results, first generate the pickle files using scripts LeNet_test_x (x = macenko/reinhard/vahadane).

5) Then, generate the heatmaps using the file heatmap_mine.png