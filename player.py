import os
import glob
import pygame

pygame.mixer.init()

current_key = None
current_channel = None


def play_chord(region, notes, is_minor, stop):

    global current_key, current_channel

    if region is None or stop:
        if current_channel:
            current_channel.stop()
        current_key = None
        return

    root = notes[region]

    folder = "Minor" if is_minor else "Major"
    chord = f"{root}m" if is_minor else f"{root}M"

    path = f"recordedmusic/{folder}/{chord}.wav"
    new_key = path

    if new_key == current_key:
        return

    if current_channel:
        current_channel.stop()

    sound = pygame.mixer.Sound(path)
    current_channel = sound.play(loops=-1)

    current_key = new_key