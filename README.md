# Introduction
- data
    - The data after dividing into different patchs.
- lib
    - Shared basic functions: dataset, data processing and model training 
- model
    - the code of net
- output
    - save model and logs
- src
    - main function and other auxiliary functions
- test
    - test a single image, which is used for deployment 
- config.py
    - parameter configuration


# User guide
- patch_based_cnn
    - Firstly, run generate.py to divide the living and spoofing img into different patches and save them as the test set and training set in the data folder. 
    - Modify the configuration file: patch_based_cnn.py
    - Then train and test.

    ## Run
    
    - Divide img to different patchs(64 is used in our paper)
        ```
        python3 data_generate.py
        ```
    
    - Train and test
        ```
        cd src
        python3 patch_cnn_main.py
        ```
    - Single image test. In this section,you can test a single image,which is used for deployment.
        ```
        cd test 
        python patch_cnn_test.py
        ```
        
- depth_based_cnn
    - Modify the configuration file: patch_based_cnn.py
    - Then train and test.

    ## Run
    
    - Train and test
        ```
        cd src
        python3 depth_cnn_main.py
        ```
    - Single image test. In this section,you can test a single image,which is used for deployment.
        ```
        cd test 
        python depth_cnn_test.py
        ```
