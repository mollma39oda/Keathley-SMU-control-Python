import sys
import pyvisa
import time
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QMessageBox,
    QFileDialog
)
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

class SweepApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize GPIB communication
        self.rm = pyvisa.ResourceManager()
        self.keithley = None
        self.abort_sweep = False

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # GPIB Address Input
        self.gpib_address_label = QLabel("GPIB Address")
        self.gpib_address_input = QLineEdit(self)
        self.gpib_address_input.setPlaceholderText("Enter GPIB Address (e.g., 21)")
        layout.addWidget(self.gpib_address_label)
        layout.addWidget(self.gpib_address_input)

        # Initialize Keithley Button
        self.init_button = QPushButton("Initialize Keithley 2401", self)
        self.init_button.clicked.connect(self.initialize_keithley)
        layout.addWidget(self.init_button)

        # Sweep Mode Selection
        self.mode_label = QLabel("Select Sweep Mode")
        self.voltage_sweep_radio = QRadioButton("Voltage Sweep")
        self.current_sweep_radio = QRadioButton("Current Sweep")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.voltage_sweep_radio)
        self.mode_group.addButton(self.current_sweep_radio)
        self.voltage_sweep_radio.setChecked(True)  # Set default to voltage sweep
        layout.addWidget(self.mode_label)
        layout.addWidget(self.voltage_sweep_radio)
        layout.addWidget(self.current_sweep_radio)

        # Wiring Mode Selection
        self.wiring_mode_label = QLabel("Select Wiring Mode")
        self.two_wire_radio = QRadioButton("2-Wire")
        self.four_wire_radio = QRadioButton("4-Wire")
        self.wiring_mode_group = QButtonGroup(self)
        self.wiring_mode_group.addButton(self.two_wire_radio)
        self.wiring_mode_group.addButton(self.four_wire_radio)
        self.two_wire_radio.setChecked(True)  # Set default to 2-wire
        layout.addWidget(self.wiring_mode_label)
        layout.addWidget(self.two_wire_radio)
        layout.addWidget(self.four_wire_radio)

        # Terminal Mode Selection
        self.terminal_mode_label = QLabel("Select Terminal Mode")
        self.front_terminal_radio = QRadioButton("Front")
        self.rear_terminal_radio = QRadioButton("Rear")
        self.terminal_mode_group = QButtonGroup(self)
        self.terminal_mode_group.addButton(self.front_terminal_radio)
        self.terminal_mode_group.addButton(self.rear_terminal_radio)
        self.front_terminal_radio.setChecked(True)  # Set default to front
        layout.addWidget(self.terminal_mode_label)
        layout.addWidget(self.front_terminal_radio)
        layout.addWidget(self.rear_terminal_radio)

        # Start Value Input
        self.start_label = QLabel("Start Value")
        self.start_input = QLineEdit(self)
        layout.addWidget(self.start_label)
        layout.addWidget(self.start_input)

        # Stop Value Input
        self.stop_label = QLabel("Stop Value")
        self.stop_input = QLineEdit(self)
        layout.addWidget(self.stop_label)
        layout.addWidget(self.stop_input)

        # Number of Points Input
        self.points_label = QLabel("Number of Points")
        self.points_input = QLineEdit(self)
        layout.addWidget(self.points_label)
        layout.addWidget(self.points_input)

        # Delay Input
        self.delay_label = QLabel("Delay (s)")
        self.delay_input = QLineEdit(self)
        layout.addWidget(self.delay_label)
        layout.addWidget(self.delay_input)

        # Maximum Current Input
        self.max_current_label = QLabel("Max Current (A)")
        self.max_current_input = QLineEdit(self)
        layout.addWidget(self.max_current_label)
        layout.addWidget(self.max_current_input)

        # Maximum Voltage Input
        self.max_voltage_label = QLabel("Max Voltage (V)")
        self.max_voltage_input = QLineEdit(self)
        layout.addWidget(self.max_voltage_label)
        layout.addWidget(self.max_voltage_input)

        # Buttons
        self.start_button = QPushButton("Start Sweep")
        self.start_button.clicked.connect(self.start_sweep)
        layout.addWidget(self.start_button)

        self.abort_button = QPushButton("Abort Sweep")
        self.abort_button.clicked.connect(self.abort_sweep_func)
        layout.addWidget(self.abort_button)

        # Create plot layout for live plotting
        self.plot_layout = QVBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)
        layout.addLayout(self.plot_layout)

        self.setLayout(layout)
        self.setWindowTitle('Sweep Function App')
        self.show()

    def initialize_keithley(self):
        """Initialize the Keithley 2401 SMU using the provided GPIB address."""
        gpib_address = self.gpib_address_input.text().strip()
        if not gpib_address:
            QMessageBox.warning(self, "Error", "Please enter a GPIB address.")
            return

        try:
            self.keithley = self.rm.open_resource(f"GPIB0::{gpib_address}::INSTR")
            self.keithley.write("*RST")  # Reset the instrument
            QMessageBox.information(self, "Success", "Keithley 2401 initialized successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize Keithley 2401: {str(e)}")

    def setupIV(self):
        """Configure Keithley for I-V testing."""
        if self.voltage_sweep_radio.isChecked():
            self.keithley.write(':SOUR:FUNC VOLT')  # Source voltage
            self.keithley.write(':SENS:FUNC "CURR"')  # Measure current
        else:
            self.keithley.write(':SOUR:FUNC CURR')  # Source current
            self.keithley.write(':SENS:FUNC "VOLT"')  # Measure voltage

        if self.four_wire_radio.isChecked():
            self.keithley.write(":SYST:RSEN ON")  # 4-wire mode
        else:
            self.keithley.write(":SYST:RSEN OFF")  # 2-wire mode

        if self.rear_terminal_radio.isChecked():
            self.keithley.write(":ROUT:TERM REAR")  # Rear terminals
        else:
            self.keithley.write(":ROUT:TERM FRONT")  # Front terminals

        # Set the maximum current compliance limit
        max_current = self.max_current_input.text().strip()
        if max_current:
            self.keithley.write(f":SENS:CURR:PROT {max_current}")  # Set current compliance

        # Set the maximum voltage compliance limit
        max_voltage = self.max_voltage_input.text().strip()
        if max_voltage:
            self.keithley.write(f":SENS:VOLT:PROT {max_voltage}")  # Set voltage compliance

        self.keithley.write(':OUTP ON')  # Turn output ON

    def start_sweep(self):
        self.abort_sweep = False
        start = float(self.start_input.text())
        stop = float(self.stop_input.text())
        points = int(self.points_input.text())
        delay = float(self.delay_input.text())

        self.setupIV()
        self.IVsweep(start, stop, points, delay)

    def abort_sweep_func(self):
        self.abort_sweep = True

    def IVsweep(self, start, stop, points, delay):
        """Perform I-V sweep."""
        sweep_values = np.linspace(start, stop, points)
        currents = []  # List to store measured currents
        voltages = []  # List to store measured voltages

        for value in sweep_values:
            if self.abort_sweep:
                QMessageBox.information(self, "Sweep Aborted", "The sweep was aborted.")
                break

            if self.voltage_sweep_radio.isChecked():
                self.keithley.write(f':SOUR:VOLT {value}')  # Set voltage
            else:
                self.keithley.write(f':SOUR:CURR {value}')  # Set current

            time.sleep(delay)  # Allow time for measurement

            try:
                response = self.keithley.query(':MEAS?').strip()  # Get and clean the response
                values = response.split(',')  # Split the response into separate values

                if len(values) > 1:
                    voltage = float(values[0])
                    current = float(values[1])
                else:
                    voltage = 0
                    current = 0
            except ValueError:
                voltage = 0
                current = 0

            voltages.append(voltage)
            currents.append(-current)
            print(f"Set Value: {value:.10f}, Voltage: {voltage:.6f} V, Current: {current:.10f} A")

            self.update_live_plot(voltages, currents)

        self.keithley.write(':OUTP OFF')  # Turn output off

        # Save data to CSV
        data = {
            'Voltage (V)': voltages,
            'Current (A)': currents
        }
        df = pd.DataFrame(data)
        self.save_data_to_csv(df)

        # Plot I-V curve
        self.create_static_plot(voltages, currents)

    def update_live_plot(self, voltages, currents):
        """Update the live plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(voltages, currents, marker='.', linestyle='-')
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (A)")
        self.canvas.draw()

    def save_data_to_csv(self, df):
        """Save the plot and data to a CSV file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Success", f"Data saved to {file_path}")

    def create_static_plot(self, voltages, currents):
        """Create a static plot and replace the live plot."""
        static_fig, ax = plt.subplots()
        ax.plot(voltages, currents, marker='.', linestyle='-')
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (A)")
        ax.grid(True)
        plt.title("I-V Curve")

        plt.show()

        # Save static plot as PNG
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;All Files (*)", options=options)
        if file_path:
            static_fig.savefig(file_path)
            QMessageBox.information(self, "Success", f"Plot saved to {file_path}")

    def closeEvent(self, event):
        if self.keithley:
            self.keithley.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = SweepApp()
    sys.exit(app.exec_())