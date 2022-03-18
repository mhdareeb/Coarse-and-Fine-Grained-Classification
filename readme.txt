##############################################################################################################
--------------------------------------------------------------------------------------------------------------
############################################## READ ME.txt ###################################################
--------------------------------------------------------------------------------------------------------------
##############################################################################################################


1. zip file contanins 7 python files - coarse.py, fine_aircrafts.py, fine_dogs.py
				       fine_cars.py, fine_flowers.py, fine_birds.py, and output.py

   zip file contains an output.txt which has the final coarse and fine classes for each image in test set
   zip file contains a project report named CS783_Assignment2.pdf
   zip file contains 2 .png files for the 2 models used - InceptionV3(coarse.png) and MobileNetV2(fine.png) generated using Keras.utils.plot_model
   zip file contains output.txt which has final predictions
   zip file contains a model_hash.txt conatining all model hashes
   zip file contains a output_hash.txt conatining output hash
   zip file contains a pdf model of entire architecture as architecture.pdf

2. coarse.py - used for predicting coarse classes
3. fine_aircrafts.py - used for saving model for predicting aircrafts
4. fine_birds.py - used for saving model for predicting birds
5. fine_cars.py - used for saving model for predicting cars
6. fine_dogs.py - used for saving model for predicting dogs
7. fine_flowers.py - used for saving model for predicting flowers
8. output.py - used for creating final output file

------------------------------------------------------------------------------------------------------------

A) Basic operation for all code -->
   -------------------------------

1.Run files coarse.py, fine_aircrafts.py, fine_dogs.py, fine_cars.py, fine_flowers.py, fine_birds.py each once 
inside working directory.

2.This will save models - Inception_coarse_final, 
			  MobileNet_only_aircrafts_final, MobileNet_only_birds_final,
			  MobileNet_only_cars_final,MobileNet_only_dogs_final and
			  MobileNet_only_flowers_final which are used in output.py

3.Run output.py - this will save final output.txt 
									   
------------------------------------------------------------------------------------------------------------------
##################################################################################################################
------------------------------------------------------------------------------------------------------------------
