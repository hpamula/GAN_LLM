## Changes made to the originial code proposed by Francois Chollet in the 1st edition od Deep learning book
```py
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
```
becomes
```py
discriminator_optimizer = keras.optimizers.RMSprop(learning_rate=0.0008, clipvalue=1.0)
```

---

```py
gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
```
becomes
```py
gan_optimizer = keras.optimizers.RMSprop(learning_rate=0.0004, clipvalue=1.0)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
discriminator.trainable = True
# gan.load_weights('gan.weights.h5')
```

---

```py
x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

iterations = 10000
batch_size = 20
save_dir = '/home/ubuntu/gan_images/'
```
becomes
```py
x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 127.5 - 1

iterations = 10000
batch_size = 20
save_dir = 'gan_images'
import shutil
shutil.rmtree(save_dir, ignore_errors=True)
os.makedirs(save_dir, exist_ok=True)
d_losses = []
a_losses = []
steps_plotted = []
```

---

```py
generated_images = generator.predict(random_latent_vectors)
```
becomes
```py
generated_images = generator.predict(random_latent_vectors, verbose=0)
```

---

```py
a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
```
becomes
```py
a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
d_losses.append(d_loss)
a_losses.append(a_loss)
steps_plotted.append(step)
```

---

```py
gan.save_weights('gan.h5')

# Print metrics
print('discriminator loss at step %s: %s' % (step, d_loss))
print('adversarial loss at step %s: %s' % (step, a_loss))
```
becomes
```py
gan.save_weights(f'gan_step_{step}.weights.h5')
previous_file = f'gan_step_{step - 100}.weights.h5'
if os.path.exists(previous_file):
    os.remove(previous_file)
print(f'step {step}, discriminator loss: {d_loss:.4f}, adversarial loss: {a_loss:.4f}')
```

```py
img = image.array_to_img((generated_images[0] + 1) * 127.5, scale=False)
img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))

# Save one real image, for comparison
img = image.array_to_img(real_images[0] * 255., scale=False)
```
becomes
```py
img = image.array_to_img((generated_images[0] + 1) * 127.5, scale=False)
img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))

# Save one real image, for comparison
img = image.array_to_img((real_images[0] + 1) * 127.5, scale=False)
```

---

---

```py
img = image.array_to_img(generated_images[i] * 255., scale=False)
```
becomes
```py
img = image.array_to_img((generated_images[i] + 1) * 127.5, scale=False)
```

---

added
```py
import matplotlib.pyplot as plt
plt.style.use('dark_background')

plt.figure(figsize=(10, 6))
plt.plot(steps_plotted, d_losses, label='Discriminator Loss')
plt.plot(steps_plotted, a_losses, label='Adversarial Loss')
plt.title('GAN Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.ylim(0.6, 1)
plt.grid(True)
plt.savefig('training_loss.png')
plt.show()
```

---

additional measurements
```py
print(f"x_train average: {np.average(x_train):.4f}, median: {np.median(x_train):.4f}")
...
print(f"generated_images average: {np.average(generated_images):.4f}, median: {np.median(generated_images):.4f}")
```