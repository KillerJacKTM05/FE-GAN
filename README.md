# Ver. 0.1
DK: I have converted my previous Noise-to-Real Face GAN project to get both input and noise as an input. Also, i have created a brand-new data loader class that only loads chosen images from dataset to buffer. It also organizes images and make necessary normalizations.
In my attempt #1 (before my last changes) i used input and noise in a concatenation layer as shown below:
  ## Use your residual blocks
    x = residual_block(x, 128)
    x = layers.UpSampling2D()(x)
    
    x = Conv2D(64, kernel_size=1, padding='same', kernel_regularizer=l2(0.01))(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.UpSampling2D()(x)
    
    x = Conv2D(32, kernel_size=1, padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.Concatenate()([x, input_real_image])
    x = layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')(x)
    
    model = Model([input_noise, input_real_image, input_label], x)
    return model

Test results of Attempt#1 can be found in it's folder. However, it leaded to create realistic but identical images for emotions even if they are different. Each epoch has different input and created different images on epoch's basis, but identical inside of themselves. Therefore, i changed the generator structure to this git-push form. Yet, haven't tested.