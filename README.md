# 5.2.-Flower-identification
CNN for Flowers Classifier

5. Machine learning
    5.2. Flower identification

    This script adopt CNN algorithm.

    ### step 1: Unzip the digits.zip file
    ### step 2: Re-arrange images into train+val+test
    ### step 3: Load train, test, and validation data sets
    ### step 4: Prepare data for CNN
    ### step 5: Build CNN
      Due to RAM limitation, only simplified model is build. Another complicated model topology is commented off.
    ### step 6: Testing
    
    ## Performance:
    Estimated Running Time:
    
    Test Accuracy:

    ## Future works:
    1. DNN has been supported by universal approximation theorem that any continuous function can be approximated by a neural network with finite neurons. Fine tuning is needed to find the suitable hyper-paramters.
    
    2. Due to the difficulty of training a complicated CNN, deep transfer learning can be used, which is also proved
    by academia, to quicken the training time and boost the accuracy.

    Note:
      a. Run this script require large RAM, recommend to execute it in a cloud platform.
        
         For simplified CNN model, free platforms Kaggle/Colab are able to run.
         For complicated CNN model, need to go to AWS/Azure cloud.
            
      b. To reduce RAM space, do not read images before train_test_split, please make directories as follow:
        
            flowers_data
            │
            └───train
            |    │
            |    └───Daisy
            |    │
            |    └───Dandelion
            |    |
            |    └───Rose
            |    │
            |    └───Sunflower
            |    |
            |    └───Tulip
            └───valid
            |    │
            |    └───Daisy
            |    │
            |    └───Dandelion
            |    |
            |    └───Rose
            |    │
            |    └───Sunflower
            |    |
            |    └───Tulip
            └───test
                 │
                 └───Daisy
                 │
                 └───Dandelion
                 |
                 └───Rose
                 │
                 └───Sunflower
                 |
                 └───Tulip
