# HTR

The purpose of this Repository is to train a neural net for handwritten text recognition (HTR) using TensorFlow 2 and then converting it to TensorFlow Lite. It is the supposed to be used in a prototype Android App. This project was part of the University course "Mobile Applications" at the University of Applied Sciences Munich in the summer semester 2021.

## Training

For training the [IAM dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) (words and lines) is used. Therefore, a directory with the images and one with the descriptions (labels) must be added to the root directory of the repository. They must be named `images` and `descriptions`.

## References

- https://github.com/Tony607/keras-image-ocr
- https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/
- https://github.com/githubharald/SimpleHTR
