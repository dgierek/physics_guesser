# Third party imports
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel

class TrainWindow(QWidget):
    def __init__(self, model, train_loader, val_loader):
        super().__init__()

        self.setWindowTitle("Train the Network from Scratch")
        self.setGeometry(250, 250, 500, 400)

        layout = QVBoxLayout()

        # Progress Bar
        self.progress_bar = QProgressBar()
        layout.addWidget(QLabel("Training Progress:"))
        layout.addWidget(self.progress_bar)

        # Loss Plot
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(QLabel("Loss Plot:"))
        layout.addWidget(self.canvas)

        # Start Training Button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(lambda: self.train_autoencoder(model, train_loader, val_loader))
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def train_autoencoder(self, model, train_loader, val_loader, epochs=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        history = {"loss": [], "val_loss": []}
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                batch = batch.to(device)
                x_i = batch[:, 1]
                optimizer.zero_grad()
                reconstructed = model(x_i)
                loss = loss_fn(x_i, reconstructed)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            history["loss"].append(total_loss / len(train_loader))
            self.progress_bar.setValue((epoch + 1) / epochs * 100)

            # Update loss plot every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.ax.clear()
                self.ax.plot(history["loss"], label="Train Loss")
                self.ax.legend()
                self.canvas.draw()

            # Validate and show reconstruction images
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        x_i = batch[:, 1]
                        reconstructed = model(x_i)
                        self.show_reconstruction_example(x_i, reconstructed)
                        break
                model.train()

    def show_reconstruction_example(self, original, reconstructed):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].imshow(original[0].permute(1, 2, 0).cpu().numpy())
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(reconstructed[0].permute(1, 2, 0).cpu().numpy())
        ax[1].set_title("Reconstructed Image")
        ax[1].axis("off")

        plt.show()



class MainMenu(QMainWindow):
    """ The class creates the main window of the application """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Neural Network Trainer")
        self.setGeometry(200, 200, 400, 300)

        # Central widget
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Buttons
        self.train_button = QPushButton("Train the Network from Scratch")
        self.eval_button = QPushButton("Evaluate a Trained Network")

        layout.addWidget(self.train_button)
        layout.addWidget(self.eval_button)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect buttons to methods
        self.train_button.clicked.connect(self.train_network)
        self.eval_button.clicked.connect(self.evaluate_network)

    def train_network(self):
        self.train_window = TrainWindow()
        self.train_window.show()


    def evaluate_network(self):
        print("Evaluating trained network...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainMenu()
    window.show()
    sys.exit(app.exec())