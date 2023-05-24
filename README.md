# Dataset: Multi-Class Spoofing Image Pair Dataset

# Overview

Welcome to the Multi-Class Spoofing Image Pair Dataset! 
This comprehensive collection consists of 9000+ pairs of images representing various 
classes, including people, animals, vehicles, plants, and food. 
This dataset serves as a resource for researchers, computer vision enthusiasts, and
AI practitioners interested in exploring the challenging field of multi-class image spoofing detection.

# Introduction
The Multi-Class Spoofing Image Pair Dataset aims to address the complex task of detecting 
spoofing activities across diverse classes, extending beyond human subjects. 
With these carefully curated pairs of images, this dataset presents a unique opportunity to develop 
and evaluate machine learning models capable of detecting image spoofing across different domains, 
ultimately contributing to enhanced security measures and maintaining the integrity of digital media.

#Dataset Structure
The dataset consists of 2000 pairs of images, with each pair belonging to one of the five classes: 
people, animals, vehicles, plants, and food. The images have been meticulously selected to encompass 
a wide range of spoofing scenarios, techniques, and variations within each class. They are stored in 
the widely supported JPEG format, and have a range of different resolutions and sizes, resulting from  
different devices and monitors. 

The mobile devices used in this dataset creation (via crowdsourcing) are:
- iPhone 6
- iPhone 7
- iPhone 8
- iPhone X
- iPhone 13
- iPhone 13s
- iPhone 14
- iPhone pro 
- iPhone SE
- Motorola 
- Samsung Galaxy S22
- Google pixel 2
- One plus 10 
- One plus 10 pro

The different spoofing mediums (monitors) used were:
- Dell UltraSharp U2415 Monitor
- ASUS ProArt PA248Q Monitor
- LG 27UK850-W Monitor
- HP Z27 Monitor
- Samsung Odyssey G9 Monitor
- ViewSonic VP2771 Monitor
- BenQ PD2700U Monitor
- Acer Predator X27 Monitor
- Apple iPad Pro (12.9-inch, 5th generation)
- Apple iPad Air (4th generation)
- Samsung Galaxy Tab S7+
- Microsoft Surface Pro 7
- Lenovo Tab P11 Pro
- Huawei MatePad Pro
- Google Pixel Slate
- Amazon Fire HD 10 Tablet

# Potential Applications

The NSpoof Dataset has extensive applications across diverse domains, including but not limited to:

Security Systems: Enhancing the robustness and accuracy of multi-class image spoofing detection systems used in access control, biometric authentication, and surveillance systems.

Forensics: Assisting forensic investigators in analyzing manipulated images across various classes to determine their authenticity and uncover digital tampering.

Social Media and Content Moderation: Supporting social media platforms and content moderation systems in identifying and mitigating the spread of manipulated content, fake accounts, and misinformation campaigns across multiple classes.

E-commerce and Advertising: Ensuring the integrity of product images and advertising campaigns by detecting and preventing spoofing attempts across different product categories.


# Code specifics
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
    
    - Divide img to different patchs
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
