# Anime Music Video Maker

## Project Overview

The aim of this project is to be able to duplicate videos in the Poisonpop Candy youtube channel here: https://www.youtube.com/channel/UC8qGpw8GUxcQZftXQeFavXw

The AMVs on Poisonpop or the Alya youtube channels typically have a static or animated picture of an anime scene of a woman with the waveform visualizer on the bottom.  Different playlists can have a circular visualizer or something different.  The bar graph visualizers aren't solid colors, but are composed of transparent colored boxes stacked on top of each other, adding boxes when the amplitude goes up and removing boxes when the amplitude drops.  Below is an example:

![alt text](image.png)

In the image above, there is another layer of animation, with petal blossoms flowing around the image.  The bottom image isn't animated, but it could be.

## Tech Stack

The tech stack I propose to use are Python, Python Arcade, the virtual python environment and others that may be useful.

* Python 3.12 (or latest version)
* Python Arcade
* VENV python virtual environment
* Python moviepy - editing and exports
* ffmpeg-python / FFMPEG for video playback
* PIL / PILLOW for sprite sheet cropping

## Expected Inputs

There are several inputs that this project will need that are for the user to provide.

* Static Image of an anime scene featuring an anime girl
* Animated Image of an anime scene featuring an anime girl
* Kawaii Phonk Toxic Candypop Eletric Jump music track of the type featured on the Poisonpop Candy youtube channel. 
* Other sources include the Toxic Sugar Music Channel
* Description or choice of the type of waveform visualizer or classification of at least three different types
  * bar graph visualizer
  * oscilloscope / oscillogram / waveform plot
  * circular / radial visualizers
  * particle / object based visualizers

## Expected Outputs

The output should be a video file, either an WMV or MP4 format, and plays at a reasonably good resolution at 1280x800.

