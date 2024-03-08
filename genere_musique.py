from midiutil import MIDIFile
import numpy as np
import pandas as pd
from collections import Counter
import sys
#np.random.seed(42)



# Define the mapping of chord symbols to MIDI notes (you may need to adjust this)
chord_mapping = {
    'C': [60, 64, 67],        # C Major (C-E-G)
    'D': [62, 66, 69],        # D Major (D-F#-A)
    'Dm7': [62, 65, 69, 72],  # D minor 7th (D-F-A-C)
    'A': [57, 64, 69],        # A Major (A-C#-E)
    'Bm': [59, 64, 68],       # B minor (B-D-F#)
    'Em': [64, 68, 71],       # E minor (E-G-B)
    'B7': [59, 63, 66, 70],   # B dominant 7th (B-D#-F#-A)
    'Em7': [64, 67, 71, 74],  # E minor 7th (E-G-B-D)
    'G': [55, 59, 62],        # G Major (G-B-D)
    'F': [53, 57, 60],        # F Major (F-A-C)
    'Bb': [58, 62, 65],       # Bb Major (Bb-D-F)
    'C7': [60, 64, 67, 70],   # C dominant 7th (C-E-G-Bb)
    'Gm': [55, 58, 62],       # G minor (G-Bb-D)
    'G#7': [56, 60, 63, 67],  # G# dominant 7th (G#-C-D#-F#)
    'G#': [56, 60, 64],       # G# Major (G#-C-D#)
    'C#': [61, 65, 68],       # C# Major (C#-E#-G#)
    'D7': [62, 66, 69, 73],   # D dominant 7th (D-F#-A-C)
    'G7': [55, 59, 62, 65],   # G dominant 7th (G-B-D-F)
    'Eb': [63, 67, 70],       # Eb Major (Eb-G-Bb)
    'F7': [53, 57, 60, 64],   # F dominant 7th (F-A-C-Eb)
    'F#': [54, 58, 61],       # F# Major (F#-A#-C#)
    'F#m': [54, 58, 61],      # F# minor (F#-A-C#)
    'Gm7': [55, 58, 62, 65],  # G minor 7th (G-Bb-D-F)
    'Ebm': [63, 66, 70],      # Eb minor (Eb-Gb-Bb)
    'Eb4': [63, 66, 70, 74],  # Eb Major 7th (Eb-G-Bb-D)
    'Bb7': [58, 62, 65, 69],  # Bb dominant 7th (Bb-D-F-A)
    'Bbm7': [58, 61, 65, 68], # Bb minor 7th (Bb-Db-F-Ab)
    'Bb6': [58, 62, 65, 69],  # Bb6 (Bb-D-F-G)
    'Bb9': [58, 62, 65, 69],  # Bb9 (Bb-D-F-Ab)
    'Bb13': [58, 62, 65, 69], # Bb13 (Bb-D-F-A)
    'A7sus4': [69, 72, 76, 79],# A7sus4 (A-D-E-G)
    'A7': [69, 72, 76, 79],    # A dominant 7th (A-C#-E-G)
    'A7#9': [69, 72, 76, 79],  # A7#9 (A-C#-E-G#-B)
    'A7b9': [69, 72, 76, 79],  # A7b9 (A-C#-E-G-Bb)
    'A9': [69, 72, 76, 79],    # A9 (A-C#-E-G-B)
    'A13': [69, 72, 76, 79],   # A13 (A-C#-E-G-B-D)
    'Bm7': [59, 62, 66, 69],   # B minor 7th (B-D-F#-A)
    'Bm7b5': [59, 62, 66, 69], # Bm7b5 (B-D-F-A)
    'Bm9': [59, 62, 66, 69],   # Bm9 (B-D-F#-A)
    'Bm11': [59, 62, 66, 69],  # Bm11 (B-D-F#-A)
    'C#m7': [61, 64, 68, 71],  # C# minor 7th (C#-E-G#-B)
    'C#m9': [61, 64, 68, 71],  # C#m9 (C#-E-G#-B-D#)
    'C#m11': [61, 64, 68, 71], # C#m11 (C#-E-G#-B-D#)
    'D6': [62, 66, 69, 74],    # D6 (D-F#-A-B)
    'D9': [62, 66, 69, 73],    # D9 (D-F#-A-C-E)
    'D13': [62, 66, 69, 73],   # D13 (D-F#-A-C-E-B)
    'E7': [64, 68, 71, 74],    # E dominant 7th (E-G#-B-D)
    'E7#9': [64, 68, 71, 74],  # E7#9 (E-G#-B-D-F#)
    'E7b9': [64, 68, 71, 74],  # E7b9 (E-G#-B-D-F)
    'E9': [64, 68, 71, 74],    # E9 (E-G#-B-D-F#)
    'E13': [64, 68, 71, 74],   # E13 (E-G#-B-D-F#-A)
    'F#7': [66, 70, 73, 76],   # F# dominant 7th (F#-A#-C#-E)
    'F#9': [66, 70, 73, 76],   # F#9 (F#-A#-C#-E-G#)
    'F#13': [66, 70, 73, 76],  # F#13 (F#-A#-C#-E-G#-B)
    'G6': [55, 59, 62, 67],    # G6 (G-B-D-E)
    'G9': [55, 59, 62, 65],    # G9 (G-B-D-F-A)
    'G13': [55, 59, 62, 65],   # G13 (G-B-D-F-A-E)
    'Gm6': [67, 70, 74, 79],
    'Fsus4': [65, 68, 72]
}



data_2 = [
    ['D', 'G', 'D', 'A', 'Bm', 'A', 'Em', 'B7', 'Em', 'B7', 'Em', 'A', 'D', 'G', 'D', 'A', 'Bm', 'A', 'B7', 'Em', 'B7', 'Em', 'B7', 'Em', 'B', 'Em', 'Em7', 'A', 'Em7', 'A', 'F', 'Bb', 'F', 'C', 'Dm', 'C', 'D7', 'Gm', 'D7', 'Gm', 'D7', 'Gm', 'C', 'G#7', 'C#', 'Bb7', 'Ebm', 'Eb4', 'C#', 'G#7', 'F7', 'Bbm', 'F', 'F#', 'F', 'Bbm7', 'Ebm', 'G#', 'A7'], #Michel Sardou, Lac du Codemara
    ['F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F7', 'Bb7', 'G7', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'F7', 'Gm', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'F7', 'Gm', 'C7', 'F', 'F7', 'G7', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F7', 'Bb7', 'G7', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'F7', 'Gm', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'F7', 'Gm', 'C7', 'F', 'F7', 'G7', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F', 'Eb(dim)', 'Bb6', 'C7', 'F'], #Charles Trenet, Boum
    ['Bm', 'Bm', 'D5', 'E7', 'G', 'A', 'E/F#', 'A', 'Bm', 'D5', 'E7', 'E7', 'G', 'A', 'A', 'E', 'F#m', 'F#m', 'A', 'Bm', 'D5', 'E7', 'E7', 'G', 'A', 'A', 'E', 'F#m', 'F#m', 'G', 'E', 'F#m', 'G', 'A/D5/E7', 'A/D5/E7', 'E', 'F#m', 'G', 'E', 'F#m', 'G', 'A', 'Bm', 'D5', 'E7', 'G', 'E', 'G', 'A', 'Bm', 'D5', 'E7', 'G', 'E', 'G', 'A', 'Bm', 'D5', 'E7', 'G', 'A', 'E/F#', 'A', 'Bm', 'D5', 'E7', 'E7', 'G', 'A', 'A', 'E', 'F#m', 'F#m', 'A', 'Bm', 'D5', 'E7', 'E7', 'G', 'A', 'A', 'E', 'F#m', 'F#', 'G', 'E', 'F#m', 'G', 'A/D5/E7', 'A/D5/E7', 'E', 'F#m', 'G', 'E', 'F#m', 'G', 'A', 'Bm', 'D5', 'E7', 'G', 'E', 'G', 'A', 'Bm', 'D5', 'E7', 'G', 'E', 'G', 'A', 'Bm', 'D5', 'E7', 'G', 'E', 'G', 'A', 'Bm', 'D5', 'E7', 'G', 'E', 'G', 'A', 'Bm', 'D5', 'E7', 'G', 'A', 'E', 'F#m', 'G', 'E', 'G', 'A'], # Jhonny Haliday , Allumer le feu
    ['Gm', 'Dm7', 'F', 'Bb', 'F', 'F', 'C', 'Dm', 'Dm', 'Bb', 'F'], #Yves Simon, Diabolo Menthe
    ['Bm', 'Em', 'A7', 'D', 'Bm', 'Em', 'A7', 'D', 'G', 'F#7', 'Bm', 'G', 'F#7', 'Bm', 'A', 'G', 'Bm', 'Em', 'A', 'G', 'Bm', 'Em', 'F#7', 'Bm', 'Bm', 'A', 'G', 'F#7', 'Bm', 'Bm', 'Em', 'F#7', 'Bm'], #Renaud, Mistral gagnant
    ['G6', 'E7', 'Am7', 'D7', 'G6', 'G7', 'C', 'Am7', 'Bb7(dim)', 'G6', 'Em7', 'Am7', 'D7', 'G6', 'G6', 'D7', 'G6', 'G6', 'D7', 'G6', 'Cm6', 'G6', 'Cm6', 'D7', 'G6', 'G6', 'G6', 'G7', 'C', 'Bb7(dim)', 'G6', 'Em7', 'Am7', 'D7', 'G6', 'Eb', 'G6'] #Anie Cordy, Tata yoyo

]

n = 2
bigrams = []
for chanson in data_2:
    ngrams = zip(*[chanson[i:] for i in range(n)])
    temp = [" ".join(ngram) for ngram in ngrams]
    bigrams.extend(temp)



# data = pd.read_csv('Liverpool_band_chord_sequence.csv')

# print(data)

# chords = data['chords'].values
# ngrams = zip(*[chords[i:] for i in range(n)])
# bigrams = [" ".join(ngram) for ngram in ngrams]
# print(ngrams)
# print(bigrams)


def predict_next_state(chord:str, data:list=bigrams):
    """Predict next chord based on current state."""
    #print(chord)
    # create list of bigrams which stats with current chord
    bigrams_with_current_chord = [bigram for bigram in bigrams if bigram.split(' ')[0]==chord]
    #print(bigrams_with_current_chord)
    # count appearance of each bigram
    count_appearance = dict(Counter(bigrams_with_current_chord))
    #print(count_appearance)
    # convert apperance into probabilities
    for ngram in count_appearance.keys():
        count_appearance[ngram] = count_appearance[ngram]/len(bigrams_with_current_chord)
    # create list of possible options for the next chord
    options = [key.split(' ')[1] for key in count_appearance.keys()]
    #print(options)
    # create  list of probability distribution
    probabilities = list(count_appearance.values())
    #print(probabilities)
    # return random prediction
    return np.random.choice(options, p=probabilities)

def generate_sequence(chord:str=None, data:list=bigrams, length:int=100):
    """Generate sequence of defined length."""
    # create list to store future chords
    chords = []
    for n in range(length):
        # append next chord for the list
        chords.append(predict_next_state(chord, bigrams))
        # use last chord in sequence to predict next chord
        chord = chords[-1]
    return chords



def chords_to_midi(chord_sequence, output_file="output.mid"):
    # Create a MIDIFile Object
    midi = MIDIFile(1)  # One track

    # Add track name and tempo
    track = 0
    time = 0
    midi.addTrackName(track, time, "Chord Progression")
    midi.addTempo(track, time, 120)  # Tempo (bpm)

    # Define some constants for note durations
    quarter_note_duration = 1  # in beats


    # Convert chord sequence to MIDI notes and add to MIDIFile
    for chord_name in chord_sequence:
        # print(chord_name)
        if chord_name in chord_mapping:
            for note in chord_mapping[chord_name]:
                # print(note)
                midi.addNote(track, 0, note, time, quarter_note_duration, 100)  # Add note to track
            time += quarter_note_duration

    # Write the MIDI file to disk
    with open(output_file, "wb") as midi_file:
        midi.writeFile(midi_file)

    print("MIDI file generated successfully:", output_file)

# Your chord sequence
if len(sys.argv) > 1:
    chord_sequence = generate_sequence(sys.argv[1])
else:
    print(bigrams)
    chord_sequence = generate_sequence(input("Donner une la première note parmis celles disponnibles : "))
# print(chord_sequence)
# Convert the chord sequence to MIDI
chords_to_midi(chord_sequence, "output.mid")
