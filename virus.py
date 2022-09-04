import numpy as np
import tensorflow as tf
import pandas as pd
import csv


from sklearn.model_selection import train_test_split

# Hyperparameters
EPOCHS = 15 # Number of times the algorithm cycles through all the training data
NUM_CATEGORIES = 1 # Number of outputs - this is binary classification so we just have 1
TEST_SIZE = 0.2 # How we are splitting the data - 20% of the data becomes a validation set


def main():

    # Get image arrays and labels for all image files
    symptoms, labels = load_data("CS51_Assignment_Diagnosis_Data.csv") #banknotes.csv, test.csv

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        symptoms, labels, test_size=TEST_SIZE
    )

    # Get a compiled neural network using tensorflow
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)


def load_data(data_dir):
    """
    Load image data from directory, split into symptoms 
    and labels, and convert all data into standard units.
    """
    
    # open the file
    with open(data_dir, newline='') as csvfile:
        symptom = csv.reader(csvfile)

        # skip the headers
        next(symptom)

        # seperate data into symptoms and lables as dicts
        data = []
        for row in symptom:
            data.append({
                "symptom": [float(cell) for cell in row[1:]],
                "label": 1 if row[0] == "1" else 0
            })

    # seperate dicts into lists
    symptoms = [row["symptom"] for row in data]
    labels = [row["label"] for row in data]

    # convert to dataframe to access columns. Standard units is based on standard deviations which is based
    # on all the data in the set, so we need to access all of it at once
    df = pd.DataFrame(symptoms)

    # convert each column to standard units (each column is a symptom)
    for i in range(4):
        df[i] = standard_unit(df[i])

    # convert back to a list
    symptoms = df.values.tolist()

    # return a list of symptoms and a list of labels
    return symptoms, labels


def standard_unit(data):
    """
    Takes a list as input and converts each item in the list into
    standard units. Standard Units calculated by:
    ((point - mean) / standard deviation)
    """
    # calculate mean and standard deviation
    mean = np.mean(data)
    sd = np. std(data)

    # list to store converted values
    new = []
    
    # apply the formula
    for i in data:
        new.append((i-mean)/sd)
    
    return new


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Define the structure of the model
    model = tf.keras.models.Sequential([

        # Hidden layer with 16 nodes and 4 inputs (4 variables)
        tf.keras.layers.Dense(16, input_shape=(4,), activation="relu"),

        # Hidden layers with 16 nodes
        tf.keras.layers.Dense(16, activation="relu"),

        # Add dropout - each node has a 50% chance of being ignored every learning cycle
        tf.keras.layers.Dropout(0.5),

        # output layer with NUM_CATEGORIES nodes
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid")

    ])
    
    # A form of annealing. The learning rate decreases each epoch, letting the algorithm jump around 
    # lots at first to try out different strategies, but eventally have to settle into a position
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=10,
        decay_rate=0.9)

    #define optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # compile model
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()
