# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load your dataset
# Replace 'your_dataset.csv' with the actual path to your dataset file
data = pd.read_csv('/Users/namangupta/Desktop/IDS/System/KDDTrain.csv')

# Remove single quotes from column names
data.columns = data.columns.str.strip("'")

# Encode categorical variables
le = LabelEncoder()
data['protocol_type'] = le.fit_transform(data['protocol_type'])
data['flag'] = le.fit_transform(data['flag'])
data['class'] = le.fit_transform(data['class'])

# Separate features and labels
X = data.drop('class', axis=1)
y = data['class']

# Split the dataset into training and testing sets with 40% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for 1D CNN (assuming sequences of length X_train.shape[1])
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Create a simple 1D CNN model
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
predictions = (model.predict(X_test_reshaped) > 0.5).astype(int)  # Convert probabilities to binary predictions

def get_accuracy():
    return accuracy_score(y_test, predictions)

# You can also print other metrics like classification report
print(classification_report(y_test, predictions))

def preprocess_input(user_input):
    user_input_df = pd.DataFrame([user_input])

    features_used = ['protocol_type', 'flag', 'src_bytes', 'dst_bytes', 'hot','count', 'srv_count',
                     'same_srv_rate', 'dst_host_count', 'dst_host_srv_count',
                     'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                     'dst_host_same_src_port_rate', 'dst_host_rerror_rate']

    # Keep only the features used during training
    user_input_df = user_input_df[features_used]

    # Transform categorical features using the same label encoder used during training
    for column in ['protocol_type', 'flag']:
        if column in le.classes_:
            user_input_df[column] = le.transform(user_input_df[column])
        else:
            # Handle unseen labels by assigning a default value or using a specific strategy
            user_input_df[column] = -1  # Replace with a suitable default value or strategy

    # Other numerical features
    numerical_features = ['src_bytes', 'dst_bytes', 'hot', 'count', 'srv_count',
                          'same_srv_rate', 'dst_host_count', 'dst_host_srv_count',
                          'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                          'dst_host_same_src_port_rate', 'dst_host_rerror_rate']

    for feature in numerical_features:
        user_input_df[feature] = user_input[feature]

    user_input_scaled = scaler.transform(user_input_df)
    user_input_reshaped = user_input_scaled.reshape(user_input_scaled.shape[0], user_input_scaled.shape[1], 1)

    return user_input_reshaped

def predict_user_input(user_input):
    preprocessed_input = preprocess_input(user_input)
    prediction = model.predict(preprocessed_input)
    predicted_class = (prediction > 0.5).astype(int)
    print(prediction)
    return predicted_class[0]

