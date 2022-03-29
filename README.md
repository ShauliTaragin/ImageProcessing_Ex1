# ImageProcessing Ex1
#### The first Task in image processing course



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Content</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#Version">Version and Information</a></li>
    <li><a href="#Submission">Submission files</a></li>
    <li><a href="#Functions">Functions</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project



### ***Author : Shauli Taragin***

### ID: 209337161
### Professor : Dr Gil Ben-Artzi

---------

<!-- Version -->
### Version and Information

Python Version - Python 3.9.2

Platform - Pycharm

---------

<!-- Submission -->
### Submission Files 

* ex1_main.py - The driver code. 
<br>In here we have the main script for running the code I wrote. That is testing that each function that was required , was implemented by me successfully.

  
* ex1_utils.py- The primary implementation class.
<br> This file contains the code I executed for each function(besides gamma) in our task.


* gamma.py- Implementation of Question 4.6


* testImg1.jpg - 
      
    A beautiful picture of Antigua, Guatemala. I chose this picture since it has an abundance of diverse colors from all types of shades.


* testImg2.jpg - A classic Image of Downtown New York, USA. This image was great for testing how the histogram and quantization would work with an image that has many details on the on hand, And a lot of the same shade of sky on the other.


* README.md - My readme file which you are currently reading :smiley:


---------

<!-- Functions -->
### Functions I've Written

* def imReadAndConvert(filename: str, representation: int) -> np.ndarray:

    Reads an image, and returns in converted as requested


* def imDisplay(filename: str, representation: int):

    Reads an image as RGB or GRAY_SCALE and displays it
    

* def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:

    Converts an RGB image to YIQ color space


* def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
      
    Converts an YIQ image to RGB color space

  
* def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
     
    Equalizes the histogram of an image and creates a new image fixed according to the histogram Equalization

    
* def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
      
    Converts an Image to a quantize color scheme. Returns A list of errors which we made each iteration and a list of images.