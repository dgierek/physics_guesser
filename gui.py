# Third party imports:
import sys
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget, QPushButton, QProgressBar, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Self written library imports:
from trainer import train_autoencoder
import autoencoder
import datasets


class TrainingThread(QThread):
    update_progress = pyqtSignal(int)
    update_loss_plot = pyqtSignal(list, list)
    update_recon_images = pyqtSignal(object, object)

    def __init__(self, model, train_loader, val_loader, epochs, save_model, initial_lr, lr_decay_step):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_model = save_model
        self.initial_lr = initial_lr
        self.lr_decay_step = lr_decay_step

    def run(self):
        train_autoencoder(self.model, self.train_loader, self.val_loader, epochs=self.epochs,
                          save_model=self.save_model, initial_lr=self.initial_lr, lr_decay_step=self.lr_decay_step,
                          gui_train_thread=self)

class TrainWindow(QWidget):
    def __init__(self, model, train_loader, val_loader, epochs, save_model, initial_lr, lr_decay_step):
        # super(TrainWindow, self).__init__()
        super().__init__()

        self.setWindowTitle("Train the Network from Scratch")
        self.setGeometry(250, 150, 600, 500)

        layout = QVBoxLayout()

        # Progress Bar
        self.progress_bar = QProgressBar()
        layout.addWidget(QLabel("Training Progress:"))
        layout.addWidget(self.progress_bar)

        # Loss Plot (Initially Hidden)
        self.loss_fig, self.loss_ax = plt.subplots(figsize=(8,5))
        self.loss_canvas = FigureCanvas(self.loss_fig)
        self.loss_canvas.setVisible(False)  # Initially hidden
        layout.addWidget(QLabel("Loss Plot:"))
        layout.addWidget(self.loss_canvas)

        self.recon_fig, self.recon_ax = plt.subplots(1, 2, figsize=(10, 5))
        self.recon_canvas = FigureCanvas(self.recon_fig)
        self.recon_canvas.setVisible(False)  # Initially hidden
        layout.addWidget(QLabel("Reconstructed Image:"))
        layout.addWidget(self.recon_canvas)

        # Start Training Button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        layout.addWidget(self.start_button)

        # Adding the variable to show the plots at first plotting:
        self.loss_canvas_first_plot = True
        self.recon_canvas_first_plot = True

        self.setLayout(layout)

        # Training thread initialization
        self.training_thread = TrainingThread(model, train_loader, val_loader, epochs, save_model, initial_lr,
                                              lr_decay_step)
        self.training_thread.update_progress.connect(self.progress_bar.setValue)
        self.training_thread.update_loss_plot.connect(self.update_loss_plot)
        self.training_thread.update_recon_images.connect(self.update_reconstruction_plots)

    def start_training(self):
        self.training_thread.start()  # Start training in a separate thread

    def update_loss_plot(self, train_loss_values, val_loss_values):
        if self.loss_canvas_first_plot:
            self.loss_canvas.setVisible(True)  # Show loss plot when training starts
            self.loss_canvas_first_plot = False

        self.loss_ax.clear()
        self.loss_ax.plot(train_loss_values, label="Train Loss")
        self.loss_ax.plot(val_loss_values, label="Validation Loss")
        self.loss_ax.legend()
        self.loss_canvas.draw()

        QApplication.processEvents()

    def update_reconstruction_plots(self, original, reconstructed):
        if self.recon_canvas_first_plot:
            self.recon_canvas.setVisible(True)  # Show original image plot when training starts
            self.recon_canvas_first_plot = False

        self.recon_ax[0].clear()
        self.recon_ax[1].clear()

        # Create subplots for original and reconstructed images
        self.recon_ax[0].imshow(original[0].permute(1, 2, 0).cpu().numpy())
        self.recon_ax[0].set_title("Original Image")
        self.recon_ax[0].axis("off")

        self.recon_ax[1].imshow(reconstructed[0].permute(1, 2, 0).cpu().numpy())
        self.recon_ax[1].set_title("Reconstructed Image")
        self.recon_ax[1].axis("off")

        self.recon_fig.tight_layout()
        # Refresh the canvas
        self.recon_canvas.draw()

        QApplication.processEvents()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Instantiate the model:
    model = autoencoder.AutoencoderWithEvolution(latent_dim=10)

    # Load dataset using DataLoader:
    dataset = datasets.LoadedDataset('preloaded_gravity_dataset_with_noise_std_0.5.pt')

    # Split dataset into training and validation sets:
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    window = TrainWindow(model, train_loader, val_loader, epochs=40, save_model=True, initial_lr=1.e-2,
                         lr_decay_step=20)
    window.show()
    sys.exit(app.exec())