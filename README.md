# Detection of Edited Images

The goal of this project is to create a web application that can distinguish between real and edited images. For the edited images it needs to further differentiate between the image being a deepfake or it being manipulated using facetune.

## Problem

As generative AI becomes more powerful, it can lead to misuse that goes undetected. In the near past we have seen multiple instances of "fun" videos being made using faceswap or similar deepfake technologies. While recreationally it opens up a new avenue for entertainment, these deepfakes have been used for various nefarious purposes.
Take the incident of the persona of Oliver Taylor , where long story short, a seemingly normal human persona was used to share terrorist propaganda. Or if we take the incident of Female Activists in the arab countries who lost credibility after altered images were circulated in the media by their opposition.
There has also been an instance of a political figure shifting the blame of one of their comments by claiming that the comments had been altered by deepfake technologies. In this case eventhough the deepfake technology gets the short end of the stick, an app that can verify this claim will help hold people accountable for their actions.

Added to this beautifying face manipulation technologies have become lighter, less complicated and more freely available. People no longer have to pay any sum as they would have to do with s/w like photoshop. With the current influencer culture, it becomes important to understand if an image influencing you is actually unaltered. The story of young kids being conscious of their bodies and chasing unhealthy standards, is old but still a very valid concern.

That is why I think this app would be very helpful atleast as a starting step.
Moreover clearly classifying the s/w involved in the manipulation is essential because based on the software the general riskiness of the data consumption changes.

## Approach

Flow Diagram:
![image](https://user-images.githubusercontent.com/110474064/234205320-2a44c4e6-e8f9-46c9-9a6f-9750d39a1527.png)

Here we have 2 separate models that are trained using 2 separate datasets. The first model is trained using a dataset with real and altered images (altered images are a combination of deepfake images and factuned images). The second model is trained using a dataset of deepfake and factune images as the 2 classes.

The original datasets were randomly sampled to get an equal number of real and altered images to pass through the first model. If the image is edited only then will it get passed to the 2nd model which determines the technology used to create it.

## Data Sourcing and Manipulations:

Deepfakes dataset sourced from - https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images?resource=download

Facetune Dataset sourced from - https://github.com/JordanMLee/Identifying-Human-Edited-Images

For the real vs edited dataset, equal number of images were sampled from both of these inidividual datasets for both of the classes.
For the deepfake vs facetune dataset, equal number of images were sampled from these same datasets.

In both cases the files were randomly split into train, test and val sets. Any files that threw "UnidentifiedImage" Exception were deleted as these were likely corrupted and could not be read by the PIL library.

##
Edited image detection
https://duke.box.com/s/ca2usi4foz9ama3pkp2rc5sm9ue9lv9q 
