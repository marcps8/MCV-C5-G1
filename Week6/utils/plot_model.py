import matplotlib.pyplot as plt

models = ['Multiplicative Fusion \n (Whispers)', 'Additive Fusion \n (Librosa)', 'Multiplicative Fusion \n (Librosa)']
train_accuracy = [57.03, 62.02, 57.28]  # Training accuracy for each model
val_accuracy = [57.65, 56.86, 55.18]    # Validation accuracy for each model

# Plotting
x = range(len(models))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x, train_accuracy, width, label='Train Accuracy')
bars2 = ax.bar([i + width for i in x], val_accuracy, width, label='Validation Accuracy')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Training and Validation Accuracy per Model')
ax.set_xticks([i + width / 2 for i in x])
ax.set_xticklabels(models)

ax.legend()
plt.ylim(0, 100)
plt.savefig('accuracies.png')
