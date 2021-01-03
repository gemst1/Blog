import numpy as np
from tensorflow import keras

lstm = keras.layers.LSTM(3, return_sequences=True, return_state=True)
input = np.ones((1,5,7))
whole_seq, final_output, final_cell_state =lstm(input)

for x in lstm.weights:
    print(x.name, '--->', x.shape)

print(whole_seq.shape)
print(final_output.shape)
print(final_cell_state.shape)

print(whole_seq.numpy())
print(final_output.numpy())
print(final_cell_state.numpy())