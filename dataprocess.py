def attention_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoding path
    c1 = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(32, (3, 3), padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(64, (3, 3), padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(128, (3, 3), padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(256, (3, 3), padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, (3, 3), padding='same')(c4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(512, (3, 3), padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)

    # Decoding path
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)

    # Attention mechanism
    attention = layers.Conv2D(256, (1, 1), activation='sigmoid')(c4)
    attention = layers.multiply([c6, attention])

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(attention)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)

    # Attention mechanism
    attention = layers.Conv2D(128, (1, 1), activation='sigmoid')(c3)
    attention = layers.multiply([c7, attention])

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(attention)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)

    # Attention mechanism
    attention = layers.Conv2D(64, (1, 1), activation='sigmoid')(c2)
    attention = layers.multiply([c8, attention])

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(attention)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation('relu')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
