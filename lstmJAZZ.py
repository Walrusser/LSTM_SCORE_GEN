import time
import datetime

import keras
import numpy
import h5py

from keras.layers import LSTM, Activation, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils

from music21 import chord, converter, instrument, note

SEQ_OFFSET = 100
EPOCHS = 200
B_SIZE = 64
FINAL_MIDI_LENGTH = 500

notes = []

midi = converter.parse("midi/midi_noGuitar.mid")
parsedNotes = midi.flat.notes

for n in parsedNotes:
    if isinstance(n, note.Note):
        notes.append(str(n.pitch))
    elif isinstance(n, chord.Chord):
        notes.append(".".join(str(i) for i in n.normalOrder))

pitchnames = sorted(set(item for item in notes))

#Output Data
Y = numpy.zeros(shape=(len(notes),len(pitchnames)))

#Input Data
X = numpy.zeros(shape=(len(notes), SEQ_OFFSET, 1))

"""for index in range(0, len(notes) - SEQ_OFFSET):
    last100 = notes[index:index + SEQ_OFFSET] #The last 100 notes in an array
    result = notes[index + SEQ_OFFSET] #The note after the last 100
    
    for n in range(0, len(last100)): #For each note/chord in the last 100 array find the correct pitch index and make the sequence a new array
        X[index, n] = pitchnames.index(last100[n])
    
    print(index)
    Y[index, pitchnames.index(result)] = 1 #Find the correct pitch index and put as the result
    
X = X / len(set(notes)) #Normalize so the highest pitch index becomes 1

X = X[:len(X)-SEQ_OFFSET-1]
Y = Y[:len(Y)-SEQ_OFFSET-1]"""

#Create the model
model = Sequential()

model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(512))

model.add(Dense(256))
model.add(Dropout(0.3))

model.add(Dense(len(set(notes))))#Final layer with the note array size as size

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(X, Y, epochs=EPOCHS, batch_size=B_SIZE)

model.save('midi_LSTM_{}.h5'.format(datetime.datetime.now))

#Generate Notes
initNoteIndex = numpy.random.randint(0, len(pitchnames) - 1)
Y_Predction = []

pattern = X[initNoteIndex]

for i in range(FINAL_MIDI_LENGTH):
    pattern = pattern / len(set(notes))
    pattern.reshape(100,1)

    prediction = model.predict(pattern)

    resultIndex = numpy.argmax(prediction)
    result = pitchnames[resultIndex]
    Y_Predction.append(result)

    pattern = numpy.append(pattern, resultIndex)
