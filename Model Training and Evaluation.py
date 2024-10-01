# Initialize models
input_shape = (256, 256, 1)  # Adjust shape as necessary
nested_unet_model = nested_unet(input_shape)
attention_unet_model = attention_unet(input_shape)

# Compile models
nested_unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
attention_unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
model_checkpoint_callback = callbacks.ModelCheckpoint(
    'best_model.h5', save_best_only=True)

# Train models
nested_unet_history = nested_unet_model.fit(train_images, train_masks, 
    validation_data=(test_images, test_masks), 
    epochs=50, batch_size=16, 
    callbacks=[model_checkpoint_callback])

attention_unet_history = attention_unet_model.fit(train_images, train_masks, 
    validation_data=(test_images, test_masks), 
    epochs=50, batch_size=16, 
    callbacks=[model_checkpoint_callback])

# Evaluate models using DICE score
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6  # to avoid division by zero
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Predictions
nested_unet_predictions = nested_unet_model.predict(test_images)
attention_unet_predictions = attention_unet_model.predict(test_images)

# DICE score calculation
nested_unet_dice = dice_coefficient(test_masks, nested_unet_predictions)
attention_unet_dice = dice_coefficient(test_masks, attention_unet_predictions)

print(f'Nested U-Net DICE Score: {nested_unet_dice.numpy()}')
print(f'Attention U-Net DICE Score: {attention_unet_dice.numpy()}')

# Save models
nested_unet_model.save('nested_unet_model.h5')
attention_unet_model.save('attention_unet_model.h5')
