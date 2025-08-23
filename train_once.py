import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("data.csv")
X = df[['Feature1', 'Feature2']].values
y = df[['Output']].values

# Build model
model = Sequential()
model.add(Dense(32, input_dim=2, activation='linear'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=100, verbose=1)

# Save trained model
model.save("model.h5")
print("✅ Model trained and saved as model.h5")
