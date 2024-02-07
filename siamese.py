# Autoencoder model
input_img = Input(shape=(128, 128, sequence_length))
# Encoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(sequence_length, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

#Define Base Network
#This function defines the base CNN to be used in both branches of the Siamese network.
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    return Model(input, x)

input_shape = (128, 10, 1)  # Adjust this based on your reshaped data

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

base_network = create_base_network(input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
output = Dense(1, activation='sigmoid')(distance)

siamese_net = Model([input_a, input_b], output)
siamese_net.compile(loss='binary_crossentropy', optimizer='adam')

def pair_generator(images, labels, batch_size):
    while True:
        batch_pairs = []
        batch_labels = []

        while len(batch_pairs) < batch_size:
            idx1, idx2 = np.random.choice(len(images), 2, replace=False)

            img1 = images[idx1]
            img2 = images[idx2]

            label = int(labels[idx1] == labels[idx2])

            batch_pairs.append([img1, img2])
            batch_labels.append(label)

        yield ([np.array([pair[0] for pair in batch_pairs]),
                np.array([pair[1] for pair in batch_pairs])],
                np.array(batch_labels))
        
def reshape_images(images, other_dim):
    reshaped_images = []
    for img in images:
        reshaped_img = img.reshape((128, other_dim, 1))
        reshaped_images.append(reshaped_img)
    return np.array(reshaped_images)

other_dimension = 1280 // 128  # Adjust based on your data
X_train_reshaped = reshape_images(X_train, other_dimension)

batch_size = 32
epochs = 10

siamese_net.fit(pair_generator(X_train_reshaped, y_train, batch_size),
                steps_per_epoch=len(X_train_reshaped) // batch_size,
                epochs=epochs)


# Testing the Siamese Network with New Images
def test_siamese_model(model, directory_path):
    sequences, labels = load_and_preprocess_images_in_sequence(directory_path)

    test_pairs, test_pair_labels = create_pairs(sequences, labels)

    similarity_scores = model.predict([test_pairs[:, 0], test_pairs[:, 1]])

    return similarity_scores

# Example usage:
new_images_directory = "/path/to/new_images_directory"
similarity_scores = test_siamese_model(siamese_net, new_images_directory)