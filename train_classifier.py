import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Ensure all data are of consistent length
# If they are lists or sequences, transform them into a consistent shape.
data = data_dict['data']
if isinstance(data, list):
    # Example: Pad sequences to the maximum length in the dataset
    max_length = max(len(item) if hasattr(item, '__len__') else 0 for item in data)
    uniform_data = []
    for item in data:
        if isinstance(item, (list, np.ndarray)):
            # Pad or truncate as needed
            padded_item = item[:max_length] + [0] * (max_length - len(item))
            uniform_data.append(padded_item)
        else:
            raise ValueError(f"Inconsistent data type detected in 'data': {type(item)}")
    data = np.asarray(uniform_data)
else:
    raise ValueError("'data' is not a list or iterable!")

# Convert labels to numpy array
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the RandomForest model
model = RandomForestClassifier()

model.fit(x_train, y_train)

# Predict and calculate the accuracy score
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
