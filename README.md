## Orchesture

A gesture-controlled orchestra conducting system

Orchesture is an experimental music interface that allows users to create and control orchestral music using only hand gestures captured through a webcam. Instead of playing a traditional instrument, users conduct an orchestra in real time using their right and left hands to control melody and harmony.

This project was created to explore how music-making can become more accessible to people who:

Cannot play traditional instruments due to physical or mental limitations

Have musical ideas but lack formal training

Want an intuitive and visual way to learn harmony and chords

Are interested in alternative musical interfaces for education and creativity

Orchesture functions like a virtual piano, but in a broader and more expressive way—allowing users to play full chords and musical structures directly through motion.

## Features
<img width="527" height="573" alt="gesture" src="https://github.com/user-attachments/assets/4b0f71c0-8d6a-4d99-b607-262952794204" />


Two-hand control system

--> Right hand: controls the main solo violin melody

--> Left hand: controls orchestral accompaniment (chords)

Screen divided into 7 pitch regions - Notes mapped from C to B

Wrist position determines pitch

Gesture-based modifiers
- fist : stop playing note
- okay sign : # for right hand, 7th for left hand
- point : +1 octave for right hand, minor for left hand

## Custom gesture training system

Captured gestures using webcam

Gestures are classified using the K-Nearest Neighbors (KNN) algorithm

## Music generation

Gesture data is converted into MIDI-style musical input

Music is produced using Logic Pro as the sound engine

## How It Works

The webcam captures live video input.

Hand landmarks are detected using MediaPipe and computer vision techniques.

A trained KNN model classifies gestures in real time.

The screen is divided into seven regions (C–B), and hand position selects pitch.

Recognized gestures modify pitch (sharp/octave) or harmony (minor/7th).

The resulting musical data is sent to Logic Pro to generate orchestral sound.

The user effectively becomes a conductor, shaping melody and harmony through movement.

## Technologies Used

Python

OpenCV (cv2) – real-time video capture and processing

MediaPipe – hand landmark detection

KNN (machine learning) – gesture classification

Logic Pro – music and sound generation

## Project Motivation

The goal of Orchesture is to make music creation more inclusive and intuitive.

Traditional instruments require years of training and physical dexterity. Orchesture lowers that barrier by allowing users to:

- Play chords directly instead of one note at a time

- Use natural hand gestures instead of complex finger techniques

- Visually understand musical structure (notes and harmonies mapped to space)

This makes it suitable for:

- People with limited mobility

- Beginners and children learning music concepts

- Artists experimenting with new musical interfaces

## Project Status

This project is currently unfinished and under development.
Planned improvements include:

Adding screenshots and demo videos

Improving gesture recognition accuracy

Expanding chord types and musical modes

Improving UI and visual feedback

Making setup and installation easier


## Future Ideas

Multiple instrument support

Rhythm and tempo control using gestures

Custom gesture mapping

Standalone sound engine (without Logic Pro dependency)

Educational mode for teaching chords and harmony

## License

This project is for educational and experimental purposes.
