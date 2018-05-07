# Object Detection in Visually Degraded Environments
Hello! This is a large amount of the code which was used while working on my project of "Object Detection in Visually Degraded Environments," done working with the [Autonomous Robotics Lab](http://www.autonomousrobotslab.com/). The main idea for this was to create a binary classifier to classify sections of our images as "cars" or "non-cars," and then use a sliding window or some more advanced region proposal algorithm in conjunction with this classifier to create an object detector. In this repository, there is a number of jupyter notebooks, and python programs which performed the main responsibilities of

1. Merging Two Datasets
2. Performing pre-processing on these two datasets, and
3. Performing hyperparameter search to find the optimal architecture for a binary classifier
   on this data.  


## The Data
The data we used combined two datasets. The first dataset was RGB-colored vehicle data taken at night time from
[Dr. Long Chen from Sun Yat-Sen University in Guangzhou](http://www.carlib.net/?page_id=35) (Thank you, very
helpful!). The second dataset is a gray-scale dataset which we collected and labeled ourselves
(Shouts out to Brenda Penn and Dusty Barnes) which is pending release. To get the data in a proper form for to train a classifier on it,
we had to do a few things.

1. Convert the SYSU data to grayscale
2. Migrate the SYSU and ARL data into a single directory, and merge their groundtruth files
3. Split the merged datasets into `training`, `validation`, and `test` sets (80-10-10)
4. Cut the vehicle-containing subsets of the images out and write them to directories with same splits.
5. Stochastically select and cut out non-vehicle containing subsets of images (only ones with non-static backgrounds)


All of the `ARL_data`,  `SYSU_data,` and `merged_dataset` directories were for this, though they only contain .gitkeep files and the ground truths for practical purposes.

This preprocessing took place in the `All_Grey_Everything.ipynb`, `Crop_cars_SYSU-GRAY.ipynb`, `Crop_data.ipynb`,
`split_dataset.py`, and `crop_training_Data.py` notebooks and programs.


## The Hyperparameter search
Our model used transfer learning on top of the well-known [VGG-19 image classifier](https://arxiv.org/pdf/1409.1556.pdf). We used [Keras](https://github.com/keras-team/keras/) to build our model, which was fantastic to use. We used [sacred](https://github.com/IDSIA/sacred), [sacredboard](https://github.com/chovanecm/sacredboard) and [labwatch](https://github.com/automl/labwatch) in order to automate and organize our hyperparameter search experiments, which were all fantastic to use as well.

We have a number of actual experiments `VGG19_transfer_frozen_hps.py`, `VGG19_transfer_frozen[m1]_hps.py`,
`VGG19_transfer_frozen[m2]_hps.py` where we searched different hyperparameter configurations for the
fully-connected layers and the optimizer with the more "categorical" hyperparameters of how many convolutional
filter sets we unfroze for the training.

Then we actually had a set of shell scripts which we would use to run these experiments many times in a row, and log the results to a mongodb database thanks to sacred, and labwatch.

Once we found some reasonably well-performing hyperparameters, we would actually run an "experiment" script with those hyperparameters hard-coded in, as can be seen in `VGG_transfer_frozen_exp.py` and `VGG_transfer_frozen[m1]_exp.py`, where we would train on these hyperparameter configurations for many more epochs until we reached some validation accuracy convergence.

### Misc
We have a couple other notebooks where we ran some preliminary experiments to investigate transfer learning, and how Keras imageDataGenerator classes would treat the grayscale images. Anything not mentioned above falls into this category.
