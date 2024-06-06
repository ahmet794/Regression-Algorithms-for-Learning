# The GUI can be run from this file.
import os
import subprocess
import sys
import pkg_resources

# Change the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def install_packages():
    # Attempt to load the requirements.txt file
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.readlines()
    except FileNotFoundError:
        print('requirements.txt file not found. Make sure it is present in the same directory as this script.')
        sys.exit(1)

    # Strip whitespace and newline characters
    requirements = [line.strip() for line in requirements]

    # Get the list of already installed packages
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    # Find packages that need to be installed
    missing_packages = [pkg for pkg in requirements if pkg.split('==')[0].lower() not in installed_packages]

    # Install missing packages
    if missing_packages:
        print('Installing missing packages...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
        print('All packages are installed.')
    else:
        print('All required packages are already installed.')

# Call the install_packages function
install_packages()

import plotting as plot
import numpy as np
import model_selection
import data_utils
import pandas as pd
from ridge_regression import RidgeRegression
from lasso_regression import LassoRegression
from data_utils import StandardScaler
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QComboBox, QLineEdit, QCheckBox, QListWidget, QMessageBox, QGridLayout, QStackedWidget, QDialog, QTableWidget, QTableWidgetItem, QFileDialog, QFormLayout, QHBoxLayout
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from sklearn.model_selection import train_test_split

class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ahmet Cihan Housing Price Predictor")
        self.resize(800, 700)
        self.model = None

        # Central widget and the main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Housing Price Predictor")
        titleFont = QFont()
        titleFont.setBold(True)
        titleFont.setPointSize(26)
        title.setFont(titleFont)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Another Layout for the Statements
        statementLayout = QVBoxLayout()

        model_statement = QLabel("Please choose the model you want to train to make predictions:")
        model_statement.setAlignment(Qt.AlignCenter)
        statementLayout.addWidget(model_statement)

        # Selection box for models
        self.modelSelector = QComboBox()
        self.modelSelector.addItems(["Linear Regression", "Ridge Regression", "Lasso Regression"])
        statementLayout.addWidget(self.modelSelector, alignment=Qt.AlignCenter)

        self.modelSelector.currentTextChanged.connect(self.onModelSelected)

        # Addtional elements for Ridge
        self.ridgelambdaLabel = QLabel("Enter λ value: ")
        self.ridgelambdaInput = QLineEdit()

        # Cross-validation parameters
        self.crossValidationCheckBox = QCheckBox("Apply cross-validation for the optimal model")
        self.crossValidationCheckBox.stateChanged.connect(self.onModelSelected)

        self.kValueLabel = QLabel("Enter k value for for cross-validation:")
        self.kValueInput = QLineEdit()

        self.lambdaListLabel = QLabel("λ values:")
        self.lambdaListInput = QListWidget()

        self.lambdaValueLabel = QLabel("Enter a λ value: \n(Enter the λ values you want to apply in the cross-validation one by one)")
        self.lambdaValueLineEdit = QLineEdit()
        self.lambdaValueLineEdit.returnPressed.connect(self.addLambdaValue)

        self.addLambdaValueButton = QPushButton("Add λ Value")
        self.addLambdaValueButton.clicked.connect(self.addLambdaValue)

        self.removelambdaValueButton = QPushButton("Remove Selected λ Value")
        self.removelambdaValueButton.clicked.connect(self.removeSelectedLambdaValue)

        # Cross-validation parameters and hide the earlier lambda value entry
        self.ridgelambdaLabel.hide()
        self.ridgelambdaInput.hide()
        self.crossValidationCheckBox.hide()
        self.lambdaValueLabel.hide()
        self.lambdaValueLineEdit.hide()
        self.addLambdaValueButton.hide()
        self.removelambdaValueButton.hide()
        self.kValueLabel.hide()
        self.kValueInput.hide()
        self.lambdaListLabel.hide()
        self.lambdaListInput.hide()

        # Cross-validation parameters
        self.crossValidationLayout = QVBoxLayout()
        self.crossValidationLayout.addWidget(self.ridgelambdaLabel)
        self.crossValidationLayout.addWidget(self.ridgelambdaInput)
        self.crossValidationLayout.addWidget(self.crossValidationCheckBox)
        self.crossValidationLayout.addWidget(self.kValueLabel)
        self.crossValidationLayout.addWidget(self.kValueInput)
        self.crossValidationLayout.addWidget(self.lambdaValueLabel)
        self.crossValidationLayout.addWidget(self.lambdaValueLineEdit)
        self.crossValidationLayout.addWidget(self.addLambdaValueButton)
        self.crossValidationLayout.addWidget(self.removelambdaValueButton)
        self.crossValidationLayout.addWidget(self.lambdaListLabel)
        self.crossValidationLayout.addWidget(self.lambdaListInput)

        # Lasso parameters layout
        self.parametersLassoLayout = QGridLayout()
        self.learningRateLabel = QLabel("Learning Rate: \n(Typically a value in between 0.000001 to 0.1)")
        self.learningRateInput = QLineEdit("0.1")

        self.convergenceThresholdLabel = QLabel("Convergence Threshold: \n(Usually a value in between 0.000001 to 0.1)")
        self.convergenceThresholdInput = QLineEdit("0.1")
        
        self.lassolambdaLabel = QLabel("λ Value \n (Penalty term coefficient)\n*Do not have to change if cross-validation is applied*")
        self.lassolambdaInput = QLineEdit("1")

        self.maxIterationsLabel = QLabel('Maxiumum Iterations \n(To reach towards the global minimum)')
        self.maxIterationsInput = QLineEdit("1000000")

        # Lasso Parameters
        self.learningRateLabel.hide()
        self.learningRateInput.hide()
        self.convergenceThresholdLabel.hide()
        self.convergenceThresholdInput.hide()
        self.lassolambdaLabel.hide()
        self.lassolambdaInput.hide()
        self.maxIterationsLabel.hide()
        self.maxIterationsInput.hide()
        

        # Lasso Parameters gridlayaout
        # first row and columns
        self.parametersLassoLayout.addWidget(self.learningRateLabel, 0, 0)
        self.parametersLassoLayout.addWidget(self.learningRateInput, 0, 1)
        self.parametersLassoLayout.addWidget(self.convergenceThresholdLabel, 0, 2)
        self.parametersLassoLayout.addWidget(self.convergenceThresholdInput, 0, 3)

        # second row and columns
        self.parametersLassoLayout.addWidget(self.lassolambdaLabel, 1, 0)
        self.parametersLassoLayout.addWidget(self.lassolambdaInput, 1, 1)
        self.parametersLassoLayout.addWidget(self.maxIterationsLabel, 1, 2)
        self.parametersLassoLayout.addWidget(self.maxIterationsInput, 1, 3)

        main_layout.addLayout(self.crossValidationLayout)
        main_layout.addLayout(self.parametersLassoLayout)

        data_statement = QLabel("Please choose the state you would like to train the model on:")
        data_statement.setAlignment(Qt.AlignCenter)
        statementLayout.addWidget(data_statement)

        # Selection box for data
        self.dataSelector = QComboBox()
        self.dataSelector.addItems(["Boston", "Extended Boston", "Washington", "California", "New York"])
        statementLayout.addWidget(self.dataSelector, alignment=Qt.AlignCenter)

        # Import data
        or_statement = QLabel("Or Import Data: \n(Imported data should have features and the target variables together,\nthe last column of the data should be the target variables)")
        or_statement.setAlignment(Qt.AlignCenter)
        statementLayout.addWidget(or_statement)

        self.importDataButton = QPushButton("+ Import")
        self.importDataButton.setMaximumWidth(200)
        self.importDataButton.clicked.connect(self.importData)
        statementLayout.addWidget(self.importDataButton, alignment=Qt.AlignCenter)

        # Create models with chosen model and data
        self.createModelButton = QPushButton('Create Model')
        self.createModelButton.setMaximumWidth(200)
        self.createModelButton.clicked.connect(self.createModel)
        statementLayout.addWidget(self.createModelButton, alignment=Qt.AlignCenter)

        # Make Predictions button
        self.buttonMakePredictions = QPushButton("Make Predictions")
        self.buttonMakePredictions.clicked.connect(self.makePredictions)
        self.buttonMakePredictions.hide()
        statementLayout.addWidget(self.buttonMakePredictions, alignment=Qt.AlignCenter)

        # Further Analysis button
        self.buttonFurtherAnalysis = QPushButton("Further Analysis")
        self.buttonFurtherAnalysis.clicked.connect(self.viewFurtherAnalysis)
        statementLayout.addWidget(self.buttonFurtherAnalysis, alignment=Qt.AlignCenter)

        main_layout.addLayout(statementLayout)
        central_widget.setLayout(main_layout)

    def createModel(self):
        """
        Create the model that is chosen with chosen data, chosen model and given parameters.
        """
        scaler = StandardScaler()
        selectedModel = self.modelSelector.currentText()

        if hasattr(self, 'imported_data') and self.imported_data is not None:
            data = self.imported_data
            # Split the imported data into train and test sets
            X = data[:, :-1]
            y = data[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            train_data = np.concatenate((X_train, y_train), axis=1)
            test_data = np.concatenate((X_test, y_test), axis=1)
            self.imported_data = None
        else:
            selectedData = self.dataSelector.currentText()
            if selectedData == "Boston":
                train_data, test_data = data_utils.load_boston(17)
                data = np.concatenate((train_data, test_data), axis=0)

            elif selectedData == "California":
                train_data, test_data = data_utils.load_california(17)
                data = np.concatenate((train_data, test_data), axis=0)
            
            elif selectedData == "Washington":
                train_data, test_data = data_utils.load_washington(17)
                data = np.concatenate((train_data, test_data), axis=0)
            
            elif selectedData == "Extended Boston":
                train_data, test_data = data_utils.load_extended_boston(17)
                data = np.concatenate((train_data, test_data), axis=0)

            elif selectedData == "New York":
                train_data, test_data = data_utils.load_newyork(17)
                data = np.concatenate((train_data, test_data), axis=0)
            

        if selectedModel == "Linear Regression":
            self.model = RidgeRegression(0)
            self.showFurtherButtons()

        
        elif selectedModel == "Ridge Regression":
            if self.crossValidationCheckBox.checkState() == Qt.Checked:
                k_str = self.kValueInput.text()
                k = int(k_str)
                lambda_values = self.getLambdaValues()
                self.model = model_selection.lambda_ridge(data, k, lambda_values)
                self.showFurtherButtons()
                self.kValueInput.clear()
                self.lambdaListInput.clear()
            else:
                alpha_str = self.ridgelambdaInput.text()
                try:
                    alpha = float(alpha_str)
                    self.ridgelambdaInput.clear()
                except ValueError:
                    self.ridgelambdaInput.clear()
                    error_dialog = QMessageBox()
                    error_dialog.setWindowTitle("Input Error")
                    error_dialog.setText("Please enter a valid λ value.")
                    error_dialog.setIcon(QMessageBox.Warning)
                    error_dialog.setStandardButtons(QMessageBox.Ok)
                    error_dialog.exec_()
                self.model = RidgeRegression(alpha)
                self.showFurtherButtons()
        
        elif selectedModel == "Lasso Regression":
            alpha_str = self.lassolambdaInput.text()
            learning_rate_str = self.learningRateInput.text()
            convergence_threshold_str = self.convergenceThresholdInput.text()
            max_iterations_str = self.maxIterationsInput.text()
            try:
                alpha = float(alpha_str)
                learning_rate = float(learning_rate_str)
                convergence_threshold = float(convergence_threshold_str)
                max_iterations = int(max_iterations_str)
            except ValueError:
                error_dialog = QMessageBox()
                error_dialog.setWindowTitle("Input Error")
                error_dialog.setText("Please enter a valid values.")
                error_dialog.setIcon(QMessageBox.Warning)
                error_dialog.setStandardButtons(QMessageBox.Ok)
                error_dialog.exec_()
            if self.crossValidationCheckBox.checkState() == Qt.Checked:
                lasso_model = LassoRegression(learning_rate, convergence_threshold, alpha, max_iterations)
                k_str = self.kValueInput.text()
                k = int(k_str)
                lambda_values = self.getLambdaValues()
                self.model = model_selection.lambda_lasso(data, k, lambda_values, lasso_model)
                self.showFurtherButtons()
                self.kValueInput.clear()
                self.lambdaListInput.clear()
            else:
                self.model = LassoRegression(learning_rate, convergence_threshold, alpha, max_iterations)
                self.showFurtherButtons()

            X_train, Y_train = scaler.fit_transform(train_data[:, :-1]), train_data[:, -1] 
            X_cal, Y_cal = scaler.transform(test_data[:, :-1]), test_data[:, -1]

            # Recombine feature and target data
            train_data = np.hstack((X_train, Y_train.reshape(-1, 1)))
            test_data = np.hstack((X_cal, Y_cal.reshape(-1, 1)))


    def importData(self):
         file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "CSV Files (*.csv);;Excel Files (*.xls *.xlsx);;All Files (*)")

         if file_path:
            try:
                df = pd.read_csv(file_path)
                data = df.to_numpy()
                self.imported_data = data

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import file:\n{e}")
                self.imported_data = None


    def onModelSelected(self):
        """
        Function to show and hide parameter entries according to the chosen model.
        """
        selectedModel = self.modelSelector.currentText()

        if selectedModel == "Ridge Regression":
            self.hideFurtherButtons()
            self.ridgelambdaLabel.show()
            self.ridgelambdaInput.show()
            self.crossValidationCheckBox.show()
            self.hideCrossValidationParameters()
            self.hideLassoParameters()
            if self.crossValidationCheckBox.checkState() == Qt.Checked:
                self.ridgelambdaLabel.hide()
                self.ridgelambdaInput.hide()
                self.showCrossValidationParameters()
            else:
                self.hideCrossValidationParameters()
        elif selectedModel == "Lasso Regression":      
            self.hideFurtherButtons()      
            self.ridgelambdaLabel.hide()
            self.ridgelambdaInput.hide()
            self.crossValidationCheckBox.show()
            self.showLassoParameters()
            if self.crossValidationCheckBox.checkState() == Qt.Checked:
                self.showCrossValidationParameters()
            else:
                self.hideCrossValidationParameters()
    
        else:
            self.hideFurtherButtons()
            self.ridgelambdaLabel.hide()
            self.ridgelambdaInput.hide()
            self.crossValidationCheckBox.hide()
            self.hideCrossValidationParameters()
            self.hideLassoParameters()

    def getData(self):

        if hasattr(self, 'imported_data') and self.imported_data is not None:
            selectedData = "Imported Data"
            data = self.imported_data
            # Split the imported data into train and test sets
            X = data[:, :-1]
            y = data[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            train_data = np.concatenate((X_train, y_train), axis=1)
            test_data = np.concatenate((X_test, y_test), axis=1)
            self.imported_data = None
        else:
            selectedData = self.dataSelector.currentText()
            if selectedData == "Boston":
                train_data, test_data = data_utils.load_boston(17)
                data = np.concatenate((train_data, test_data), axis=0)

            elif selectedData == "California":
                train_data, test_data = data_utils.load_california(17)
                data = np.concatenate((train_data, test_data), axis=0)
            
            elif selectedData == "Washington":
                train_data, test_data = data_utils.load_washington(17)
                data = np.concatenate((train_data, test_data), axis=0)
            
            elif selectedData == "Extended Boston":
                train_data, test_data = data_utils.load_extended_boston(17)
                data = np.concatenate((train_data, test_data), axis=0)
            
            elif selectedData == "New York":
                train_data, test_data = data_utils.load_newyork(17)
                data = np.concatenate((train_data, test_data), axis=0)

        return train_data, test_data, data, selectedData


    def addLambdaValue(self):
        """
        Adds the given values to the lambda values list.
        """
        lambda_value = self.lambdaValueLineEdit.text()
        try:
            lambda_value = float(lambda_value)
            self.lambdaListInput.addItem(str(lambda_value))
            self.lambdaValueLineEdit.clear()
        except ValueError:
            self.lambdaValueLineEdit.clear()
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Input Error")
            error_dialog.setText("Please enter a valid λ value.")
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()

    def removeSelectedLambdaValue(self):
        """
        Removes the selected items from the lambda values list.
        """
        selected_items = self.lambdaListInput.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            self.lambdaListInput.takeItem(self.lambdaListInput.row(item))

    def getLambdaValues(self):
        """
        Obtain the lambda values for the models from the given lambda values list.

        :return: Return the entered lambda values.
        """
        lambda_values = []
        for i in range(self.lambdaListInput.count()):
            item = self.lambdaListInput.item(i)
            lambda_values.append(float(item.text()))
        return lambda_values
    

    # show hide functions for visibility in the GUI
    def showFurtherButtons(self):
        self.buttonMakePredictions.show()

    def hideFurtherButtons(self):
        self.buttonMakePredictions.hide()

    def hideCrossValidationParameters(self):
        self.kValueLabel.hide()
        self.kValueInput.hide()
        self.lambdaValueLabel.hide()
        self.lambdaValueLineEdit.hide()
        self.addLambdaValueButton.hide()
        self.removelambdaValueButton.hide()
        self.lambdaListLabel.hide()
        self.lambdaListInput.hide()

    def showCrossValidationParameters(self):
        self.kValueLabel.show()
        self.kValueInput.show()
        self.lambdaValueLabel.show()
        self.lambdaValueLineEdit.show()
        self.addLambdaValueButton.show()
        self.removelambdaValueButton.show()
        self.lambdaListLabel.show()
        self.lambdaListInput.show()

    def showLassoParameters(self):
        self.learningRateLabel.show()
        self.learningRateInput.show()
        self.convergenceThresholdLabel.show()
        self.convergenceThresholdInput.show()
        self.maxIterationsLabel.show()
        self.maxIterationsInput.show()
        self.lassolambdaLabel.show()
        self.lassolambdaInput.show()

    def hideLassoParameters(self):
        self.learningRateLabel.hide()
        self.learningRateInput.hide()
        self.convergenceThresholdLabel.hide()
        self.convergenceThresholdInput.hide()
        self.maxIterationsLabel.hide()
        self.maxIterationsInput.hide()
        self.lassolambdaLabel.hide()
        self.lassolambdaInput.hide()


    def viewFurtherAnalysis(self):
        train_data, test_data, data, selectedData = self.getData()
        dialog = AnalysisDialog(train_data, test_data, data, selectedData)
        dialog.exec()

    def makePredictions(self):
        train_data, test_data, data, selectedData = self.getData()
        if selectedData == "Extended Boston":
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Extended Boston Dataset Error")
            error_dialog.setText(f"Boston Datasets are not available for house price predictions because of ethical reasons. Choose another data for predictions or make further analysis with {selectedData} dataset.")
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()
            self.model = None
            self.buttonMakePredictions.hide()
        elif selectedData == "Boston":
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Boston Dataset Error")
            error_dialog.setText(f"Boston Datasets are not available for house price predictions because of ethical reasons. Choose another data for predictions or make further analysis with {selectedData} dataset.")
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()
            self.model = None
            self.buttonMakePredictions.hide()
        else:
            dialog = MakePredictions(data, selectedData, self.model)
            dialog.exec()
            self.buttonMakePredictions.hide()

    
class MakePredictions(QDialog):
    scaler = StandardScaler()
    def __init__(self, data, selectedData, model):
        super().__init__()
        self.setWindowTitle(f"Making Predictions Using {selectedData} Data")
        self.setLayout(QVBoxLayout())
        self.resize(600, 500)

        self.data = data
        self.selectedData = selectedData
        self.model = model

        self.formLayout = QFormLayout()
        self.inputFields = []

        feature_names = self.featureNames()
        for name in feature_names:
            lineEdit = QLineEdit(self)
            self.formLayout.addRow(f"{name}:", lineEdit)
            self.inputFields.append(lineEdit)

        self.layout().addLayout(self.formLayout)

        # Prediction button
        self.predictButton = QPushButton("Predict", self)
        self.predictButton.clicked.connect(self.makePrediction)
        self.layout().addWidget(self.predictButton)

        # Label to display prediction
        self.predictionLabel = QLabel("", self)
        self.layout().addWidget(self.predictionLabel)

        # Label to display conformal prediction
        self.conformalPredictionLabel = QLabel("", self)
        self.layout().addWidget(self.conformalPredictionLabel)


    def featureNames(self):
        if self.selectedData == "Washington":
            feature_names = ["Number of Bedrooms", "Number of Bathrooms", "Living room square feet are", "The square feet of the lot", "Number of floors", "Waterfront", "Rating of the view", "Condition of the House", "The Square feet of the house above ground floor", "The Square feet of the basement", "Year it was built", "Year it was renovated", "Significance Level (CP)"]
        elif self.selectedData == "California":
            feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitute", "Longitude", "Significance Level(CP)"]
        elif self.selectedData == "New York":
            feature_names = ["Area", "Bedrooms ", "Bathrooms", "Stories", "Mainroad", "Guestroom", "Basement", "Hot Water Heater", "AC", "Parking", "Prefarea", "Furnishing Status", "Significance Level(CP)"]
        return feature_names
    
    def makePrediction(self):
        values = [float(field.text()) for field in self.inputFields]
        input_values = values[:-1]
        significance_level = values[-1]

        # Reshape inputs to match what the model expects
        inputs_reshaped = np.array(input_values).reshape(1, -1)

        # Create calibration data for conformal prediction
        new_train_data , calibration_data = train_test_split(self.data, test_size=0.25, random_state=17)

        X_train, Y_train = self.scaler.fit_transform(new_train_data[:, :-1]), new_train_data[:, -1] 
        X_test = self.scaler.transform(inputs_reshaped)
        # Recombine feature and target data
        train_data = np.hstack((X_train, Y_train.reshape(-1, 1)))

        # Train the data
        self.model.train(train_data)
        # Make prediction
        prediction = self.model.predictGUI(X_test)

        # Transform the calibration data
        X_cal, Y_cal = self.scaler.transform(calibration_data[:, :-1]), calibration_data[:, -1]
        calibration_data = np.hstack((X_cal, Y_cal.reshape(-1, 1)))

        # Apply conformal prediction
        cp = self.model.conformal_predictionGUI(prediction, calibration_data, significance_level)

        # Display prediction
        self.predictionLabel.setText(f"Predicted Value: {prediction[0]}")
        cp_lower = cp[0][0]
        cp_upper = cp[0][1]
        self.conformalPredictionLabel.setText(f"Conformal Prediction Interval: {cp_lower}, {cp_upper}")

class AnalysisDialog(QDialog):
    def __init__(self, train_data, test_data, data, selectedData):
        super().__init__()
        self.setWindowTitle(f"Further Analysis on {selectedData}")
        self.layout = QVBoxLayout(self)

        self.train_data = train_data
        self.test_data = test_data
        self.data = data
        self.selectedData = selectedData

        self.linear = None
        self.ridge = None
        self.lasso = None

        self.ridgekValueLabel = QLabel("Enter k value for for cross-validation:")
        self.ridgekValueInput = QLineEdit()

        self.ridgelambdaListLabel = QLabel("λ values:")
        self.ridgelambdaListInput = QListWidget()

        self.ridgelambdaValueLabel = QLabel("Enter a λ value: \n(Enter the λ values you want to apply in the cross-validation one by one)")
        self.ridgelambdaValueLineEdit = QLineEdit()
        self.ridgelambdaValueLineEdit.returnPressed.connect(self.addridgelambdaValue)

        self.ridgeaddLambdaValueButton = QPushButton("Add λ Value")
        self.ridgeaddLambdaValueButton.clicked.connect(self.addridgelambdaValue)

        self.ridgeremovelambdaValueButton = QPushButton("Remove Selected λ Value")
        self.ridgeremovelambdaValueButton.clicked.connect(self.removeridgeSelectedLambdaValue)

        self.kValueLabel = QLabel("Enter k value for for cross-validation:")
        self.kValueInput = QLineEdit()

        self.lambdaListLabel = QLabel("λ values:")
        self.lambdaListInput = QListWidget()

        self.lambdaValueLabel = QLabel("Enter a λ value: \n(Enter the λ values you want to apply in the cross-validation one by one)")
        self.lambdaValueLineEdit = QLineEdit()
        self.lambdaValueLineEdit.returnPressed.connect(self.addlambdaValue)

        self.addLambdaValueButton = QPushButton("Add λ Value")
        self.addLambdaValueButton.clicked.connect(self.addlambdaValue)

        self.removelambdaValueButton = QPushButton("Remove Selected λ Value")
        self.removelambdaValueButton.clicked.connect(self.removeSelectedLambdaValue)

        self.hideRidgeCrossValidationParameters()
        self.hideCrossValidationParameters()
        
        # Stackable widget to hold pages
        self.pages = QStackedWidget()
        self.layout.addWidget(self.pages)
        
        # Create pages
        self.createIntroPage()
        self.createRidgePage()
        self.createLassoPage()
        
        # Set initial page
        self.pages.setCurrentIndex(0)


    def createIntroPage(self):
        self.linear = RidgeRegression(0)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel(f"For further analysis, with {self.selectedData} data, all available models will be created. \nLinear Regression model is now created, click next to create the Ridge Regression Model"))
        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda: self.pages.setCurrentIndex(1))
        layout.addWidget(next_button)
        self.pages.addWidget(page)

    def createRidgePage(self):
        def crossValidationChecked():
            if self.ridgecrossValidationCheckBox.checkState() == Qt.Checked:
                self.ridgelambdaLabel.hide()
                self.ridgelambdaInput.hide()
                self.showRidgeCrossValidationParameters()

            else:
                self.ridgelambdaLabel.show()
                self.ridgelambdaInput.show()
                self.hideRidgeCrossValidationParameters()
    

        # Implement similar to the intro page but include parameter inputs and a model creation button
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("To create the Ridge Regression Model enter a λ or choose cross-validation and enter parameters"))
        # Addtional elements for Ridge
        self.ridgelambdaLabel = QLabel("Enter λ value: ")
        self.ridgelambdaInput = QLineEdit()

        # Cross-validation parameters
        self.ridgecrossValidationCheckBox = QCheckBox("Apply cross-validation for the optimal model")
        self.ridgecrossValidationCheckBox.stateChanged.connect(crossValidationChecked)

        layout.addWidget(self.ridgelambdaLabel)
        layout.addWidget(self.ridgelambdaInput)
        layout.addWidget(self.ridgecrossValidationCheckBox)
        layout.addWidget(self.ridgekValueLabel)
        layout.addWidget(self.ridgekValueInput)
        layout.addWidget(self.ridgelambdaValueLabel)
        layout.addWidget(self.ridgelambdaValueLineEdit)
        layout.addWidget(self.ridgeaddLambdaValueButton)
        layout.addWidget(self.ridgeremovelambdaValueButton)
        layout.addWidget(self.ridgelambdaListLabel)
        layout.addWidget(self.ridgelambdaListInput)

        def runRidge():
            if self.ridgecrossValidationCheckBox.checkState() == Qt.Checked:
                k_str = self.ridgekValueInput.text()
                k = int(k_str)
                lambda_values = self.getridgelambdaValues()
                model = model_selection.lambda_ridge(self.data, k, lambda_values)
                self.ridgekValueInput.clear()
                self.ridgelambdaListInput.clear()
            else:
                alpha_str = self.ridgelambdaInput.text()
                alpha = float(alpha_str)
                self.ridgelambdaInput.clear()
                model = RidgeRegression(alpha)
            self.ridge = model
            layout.addWidget(QLabel("Model crated, please press next"))

        createModel_button = QPushButton("Create Model")
        createModel_button.clicked.connect(runRidge)
        layout.addWidget(createModel_button)

        next_button = QPushButton("Next")
        next_button.clicked.connect(self.checkRidgeModelCreation)
        layout.addWidget(next_button)
        self.pages.addWidget(page)
    

    def createLassoPage(self):
        def lassoCrossValidationChecked():
            if self.lassoCrossValidationCheckBox.checkState() == Qt.Checked:
                self.showCrossValidationParameters() 
            else:
                self.hideCrossValidationParameters() 

        # Page layout and widgets
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("To create the Lasso Regression Model enter model parameters or choose cross-validation and enter parameters"))

        # Adjustments to reflect it's for Lasso
        self.lassoCrossValidationCheckBox = QCheckBox("Apply cross-validation for the optimal model")
        self.lassoCrossValidationCheckBox.stateChanged.connect(lassoCrossValidationChecked)

        # Common cross-validation widgets reused
        # Parameters specific to Lasso
        self.learningRateLabel = QLabel("Learning Rate: \n(Typically a value between 0.000001 to 0.1)")
        self.learningRateInput = QLineEdit("0.1")
        self.convergenceThresholdLabel = QLabel("Convergence Threshold: \n(Usually a value between 0.000001 to 0.1)")
        self.convergenceThresholdInput = QLineEdit("0.1")
        self.lassolambdaLabel = QLabel("λ Value: \n(Penalty term, coefficient)\n*Not needed if cross-validation is applied*")
        self.lassolambdaInput = QLineEdit("1")
        self.maxIterationsLabel = QLabel('Maximum Iterations: \n(To reach towards the global minimum)')
        self.maxIterationsInput = QLineEdit("100000")

        # Layouts
        layout.addWidget(self.lassoCrossValidationCheckBox)
        layout.addWidget(self.kValueLabel)
        layout.addWidget(self.kValueInput)
        layout.addWidget(self.lambdaValueLabel)
        layout.addWidget(self.lambdaValueLineEdit)
        layout.addWidget(self.addLambdaValueButton)
        layout.addWidget(self.removelambdaValueButton)
        layout.addWidget(self.lambdaListLabel)
        layout.addWidget(self.lambdaListInput)

        # Adding Lasso specific parameters to the layout
        self.parametersLassoLayout = QGridLayout()
        self.parametersLassoLayout.addWidget(self.learningRateLabel, 0, 0)
        self.parametersLassoLayout.addWidget(self.learningRateInput, 0, 1)
        self.parametersLassoLayout.addWidget(self.convergenceThresholdLabel, 0, 2)
        self.parametersLassoLayout.addWidget(self.convergenceThresholdInput, 0, 3)
        self.parametersLassoLayout.addWidget(self.lassolambdaLabel, 1, 0)
        self.parametersLassoLayout.addWidget(self.lassolambdaInput, 1, 1)
        self.parametersLassoLayout.addWidget(self.maxIterationsLabel, 1, 2)
        self.parametersLassoLayout.addWidget(self.maxIterationsInput, 1, 3)
        layout.addLayout(self.parametersLassoLayout)

        def runLasso():
            alpha_str = self.lassolambdaInput.text()
            learning_rate_str = self.learningRateInput.text()
            convergence_threshold_str = self.convergenceThresholdInput.text()
            max_iterations_str = self.maxIterationsInput.text()

            alpha = float(alpha_str)
            learning_rate = float(learning_rate_str)
            convergence_threshold = float(convergence_threshold_str)
            max_iterations = int(max_iterations_str)


            if self.lassoCrossValidationCheckBox.checkState() == Qt.Checked:
                lasso_model = LassoRegression(learning_rate, convergence_threshold, alpha, max_iterations)
                k_str = self.kValueInput.text()
                k = int(k_str)
                lambda_values = self.getlambdaValues()
                model = model_selection.lambda_lasso(self.data, k, lambda_values, lasso_model)
                self.kValueInput.clear()
                self.lambdaListInput.clear()
            else:
                model = LassoRegression(learning_rate, convergence_threshold, alpha, max_iterations)

            self.lasso = model
            layout.addWidget(QLabel("Model crated, please press next for further analysis"))


        createModel_button = QPushButton("Create Model")
        createModel_button.clicked.connect(runLasso)
        layout.addWidget(createModel_button)

        next_button = QPushButton("Next")
        next_button.clicked.connect(self.showResults)
        layout.addWidget(next_button)
        self.pages.addWidget(page)
        

    def createResultPage(self):
        scaler = StandardScaler()

        X_train, Y_train = scaler.fit_transform(self.train_data[:, :-1]), self.train_data[:, -1] 
        X_cal, Y_cal = scaler.transform(self.test_data[:, :-1]), self.test_data[:, -1]

        # Recombine feature and target data
        train_data = np.hstack((X_train, Y_train.reshape(-1, 1)))
        test_data = np.hstack((X_cal, Y_cal.reshape(-1, 1)))

        # Linear Regression Results
        self.linear.train(train_data)
        linear_r_score_train = self.linear.r_score()
        linear_mse_train = self.linear.mse_()
        self.linear.test(test_data)
        linear_r_score_test = self.linear.r_score()
        linear_mse_test = self.linear.mse_()
        linear_var = self.linear.var()

        # Linear Regression Results
        self.ridge.train(train_data)
        ridge_r_score_train = self.ridge.r_score()
        ridge_mse_train = self.ridge.mse_()
        self.ridge.test(test_data)
        ridge_r_score_test = self.ridge.r_score()
        ridge_mse_test = self.ridge.mse_()
        ridge_var = self.ridge.var()

        # Lasso Regression Results
        self.lasso.train(train_data)
        lasso_r_score_train = self.lasso.r_score()
        lasso_mse_train = self.lasso.mse_()
        self.lasso.test(test_data)
        lasso_r_score_test = self.lasso.r_score()
        lasso_mse_test = self.lasso.mse_()
        lasso_var = self.lasso.var()

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page.setLayout(page_layout)
        resultTable = QTableWidget(3, 3)
        resultTable.setHorizontalHeaderLabels(['Linear Regression', 'Ridge Regression', 'Lasso Regression'])
        resultTable.setVerticalHeaderLabels(['R² Score', 'MSE', 'Variance'])

        # Populate the table using the class attributes
        resultTable.setItem(0, 0, QTableWidgetItem(f"Train: {linear_r_score_train}\nTest: {linear_r_score_test}"))
        resultTable.setItem(1, 0, QTableWidgetItem(f"Train: {linear_mse_train}\nTest: {linear_mse_test}"))
        resultTable.setItem(2, 0, QTableWidgetItem(str(linear_var)))

        resultTable.setItem(0, 1, QTableWidgetItem(f"Train: {ridge_r_score_train}\nTest: {ridge_r_score_test}"))
        resultTable.setItem(1, 1, QTableWidgetItem(f"Train: {ridge_mse_train}\nTest: {ridge_mse_test}"))
        resultTable.setItem(2, 1, QTableWidgetItem(str(ridge_var)))

        resultTable.setItem(0, 2, QTableWidgetItem(f"Train: {lasso_r_score_train}\nTest: {lasso_r_score_test}"))
        resultTable.setItem(1, 2, QTableWidgetItem(f"Train: {lasso_mse_train}\nTest: {lasso_mse_test}"))
        resultTable.setItem(2, 2, QTableWidgetItem(str(lasso_var)))

        resultTable.resizeColumnsToContents()
        resultTable.resizeRowsToContents()

        plots_layout = QHBoxLayout()

        linearCanvas = plot.model_coefficients(self.linear.coef_(), title='Linear Regression Coefficients')
        ridgeCanvas = plot.model_coefficients(self.ridge.coef_(), title='Ridge Regression Coefficients')
        lassoCanvas = plot.model_coefficients(self.lasso.coef_(), title='Lasso Regression Coefficients')

        # Add canvases to the horizontal layout
        plots_layout.addWidget(linearCanvas)
        plots_layout.addWidget(ridgeCanvas)
        plots_layout.addWidget(lassoCanvas)

        # Add the result table to the main layout
        page_layout.addWidget(resultTable)

        # Add the horizontal layout of plots to the main layout
        page_layout.addLayout(plots_layout)

        self.resize(600, 600)

        self.pages.addWidget(page)

    def checkRidgeModelCreation(self):
        if self.ridge is None:
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Model Error")
            error_dialog.setText("Make sure to create a model before pressing next")
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()
        else:
            self.pages.setCurrentIndex(self.pages.currentIndex() + 1)
        

    def showResults(self):
        if self.lasso is None:
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Model Error")
            error_dialog.setText("Make sure to create a model before pressing next")
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()
        else:
            self.createResultPage()
            self.pages.setCurrentIndex(self.pages.currentIndex() + 1)
        

    def setData(self, train_data, test_data, data, selectedData):
        self.train_data = train_data
        self.test_data = test_data
        self.data = data
        self.selectedData = selectedData

    def crossValidationChecked(self):
        if self.crossValidationCheckBox.checkState() == Qt.Checked:
            self.ridgelambdaLabel.hide()
            self.ridgelambdaInput.hide()
            GUI.showCrossValidationParameters(self)

        else:
            self.ridgelambdaLabel.show()
            self.ridgelambdaInput.show()
            GUI.hideCrossValidationParameters(self)
    
    def addlambdaValue(self):
        lambda_value = self.lambdaValueLineEdit.text()
        try:
            lambda_value = float(lambda_value)
            self.lambdaListInput.addItem(str(lambda_value))
            self.lambdaValueLineEdit.clear()
        except ValueError:
            self.lambdaValueLineEdit.clear()
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Input Error")
            error_dialog.setText("Please enter a valid λ value.")
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()

    def addridgelambdaValue(self):
        lambda_value = self.ridgelambdaValueLineEdit.text()
        try:
            lambda_value = float(lambda_value)
            self.ridgelambdaListInput.addItem(str(lambda_value))
            self.ridgelambdaValueLineEdit.clear()
        except ValueError:
            self.ridgelambdaValueLineEdit.clear()
            error_dialog = QMessageBox()
            error_dialog.setWindowTitle("Input Error")
            error_dialog.setText("Please enter a valid λ value.")
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()

    def removeSelectedLambdaValue(self):
        selected_items = self.lambdaListInput.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            self.lambdaListInput.takeItem(self.lambdaListInput.row(item))

    def removeridgeSelectedLambdaValue(self):
        selected_items = self.ridgelambdaListInput.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            self.ridgelambdaListInput.takeItem(self.ridgelambdaListInput.row(item))

    def getlambdaValues(self):
        lambda_values = []
        for i in range(self.lambdaListInput.count()):
            item = self.lambdaListInput.item(i)
            lambda_values.append(float(item.text()))
        return lambda_values
    
    def getridgelambdaValues(self):
        lambda_values = []
        for i in range(self.ridgelambdaListInput.count()):
            item = self.ridgelambdaListInput.item(i)
            lambda_values.append(float(item.text()))
        return lambda_values
    
    def hideRidgeCrossValidationParameters(self):
        self.ridgekValueLabel.hide()
        self.ridgekValueInput.hide()
        self.ridgelambdaValueLabel.hide()
        self.ridgelambdaValueLineEdit.hide()
        self.ridgeaddLambdaValueButton.hide()
        self.ridgeremovelambdaValueButton.hide()
        self.ridgelambdaListLabel.hide()
        self.ridgelambdaListInput.hide()

    def showRidgeCrossValidationParameters(self):
        self.ridgekValueLabel.show()
        self.ridgekValueInput.show()
        self.ridgelambdaValueLabel.show()
        self.ridgelambdaValueLineEdit.show()
        self.ridgeaddLambdaValueButton.show()
        self.ridgeremovelambdaValueButton.show()
        self.ridgelambdaListLabel.show()
        self.ridgelambdaListInput.show()

    def hideCrossValidationParameters(self):
        self.kValueLabel.hide()
        self.kValueInput.hide()
        self.lambdaValueLabel.hide()
        self.lambdaValueLineEdit.hide()
        self.addLambdaValueButton.hide()
        self.removelambdaValueButton.hide()
        self.lambdaListLabel.hide()
        self.lambdaListInput.hide()

    def showCrossValidationParameters(self):
        self.kValueLabel.show()
        self.kValueInput.show()
        self.lambdaValueLabel.show()
        self.lambdaValueLineEdit.show()
        self.addLambdaValueButton.show()
        self.removelambdaValueButton.show()
        self.lambdaListLabel.show()
        self.lambdaListInput.show()
        
app = QApplication(sys.argv)
window = GUI()
window.show()
sys.exit(app.exec())