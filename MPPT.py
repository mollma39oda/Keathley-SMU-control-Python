import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import traceback
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QCheckBox,
    QMessageBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QTabWidget,
    QSplitter,
    QTextEdit
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Try to import pyvisa with better error handling
try:
    import pyvisa
    VISA_AVAILABLE = True
except ImportError:
    VISA_AVAILABLE = False
    print("PyVISA not installed. Running in simulation mode.")

class MPPTSweepApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize GPIB variables
        self.GPIB_ADDRESS = 21
        self.rm = None
        self.keithley = None
        self.abort_sweep = False
        self.simulation_mode = False
        
        # Data storage
        self.voltages = []
        self.currents = []
        self.powers = []

        # Check if PyVISA is available
        if not VISA_AVAILABLE:
            self.simulation_mode = True
            QMessageBox.warning(self, "PyVISA Missing", 
                               "PyVISA module not found. Running in simulation mode.")
        else:
            # Try to create the resource manager
            try:
                self.rm = pyvisa.ResourceManager()
                # List available resources for debugging
                self.available_resources = self.rm.list_resources()
            except Exception as e:
                self.simulation_mode = True
                QMessageBox.warning(self, "VISA Error", 
                                  f"Error initializing VISA: {str(e)}\nRunning in simulation mode.")

        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()
        
        # Create tabs
        self.tabs = QTabWidget()
        self.sweep_tab = QWidget()
        self.analysis_tab = QWidget()
        self.debug_tab = QWidget()  # New debug tab
        
        self.tabs.addTab(self.sweep_tab, "Sweep Control")
        self.tabs.addTab(self.analysis_tab, "Data Analysis")
        self.tabs.addTab(self.debug_tab, "Debug Info")  # Add debug tab
        
        # Setup tabs
        self.setup_sweep_tab()
        self.setup_analysis_tab()
        self.setup_debug_tab()  # Setup debug tab
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.status_label = QLabel("Ready. GPIB Address set to 21.")
        main_layout.addWidget(self.status_label)
        
        # Update status based on initialization results
        if self.simulation_mode:
            self.status_label.setText("Running in SIMULATION MODE - no real hardware control")
        elif hasattr(self, 'available_resources') and self.available_resources:
            self.status_label.setText(f"VISA initialized. Available resources: {', '.join(self.available_resources)}")
        
        self.setLayout(main_layout)
        self.setWindowTitle('MPPT Sweep Application')
        self.resize(1000, 800)
        self.show()

    def setup_sweep_tab(self):
        sweep_layout = QHBoxLayout()
        
        # Left panel - Controls
        control_panel = QVBoxLayout()
        
        # Instrument initialization
        init_group = QGroupBox("Instrument Control")
        init_layout = QVBoxLayout()
        
        # Allow GPIB address change for troubleshooting
        gpib_layout = QHBoxLayout()
        gpib_layout.addWidget(QLabel("GPIB Address:"))
        self.gpib_address_input = QLineEdit(str(self.GPIB_ADDRESS))
        gpib_layout.addWidget(self.gpib_address_input)
        init_layout.addLayout(gpib_layout)
        
        self.init_button = QPushButton("Initialize Keithley 2401", self)
        self.init_button.clicked.connect(self.initialize_keithley)
        init_layout.addWidget(self.init_button)
        
        # Wiring mode - simplified
        self.four_wire_check = QCheckBox("Use 4-Wire Sensing (Default: 2-Wire)")
        init_layout.addWidget(self.four_wire_check)
        
        init_group.setLayout(init_layout)
        control_panel.addWidget(init_group)
        
        # Sweep Parameters
        sweep_group = QGroupBox("Sweep Parameters")
        param_layout = QGridLayout()
        
        # Start Voltage
        param_layout.addWidget(QLabel("Start Voltage (V):"), 0, 0)
        self.start_voltage = QLineEdit("0")
        param_layout.addWidget(self.start_voltage, 0, 1)
        
        # Stop Voltage
        param_layout.addWidget(QLabel("Stop Voltage (V):"), 1, 0)
        self.stop_voltage = QLineEdit("5")
        param_layout.addWidget(self.stop_voltage, 1, 1)
        
        # Points
        param_layout.addWidget(QLabel("Number of Points:"), 2, 0)
        self.points_input = QLineEdit("50")
        param_layout.addWidget(self.points_input, 2, 1)
        
        # Delay
        param_layout.addWidget(QLabel("Delay (s):"), 3, 0)
        self.delay_input = QLineEdit("0.1")
        param_layout.addWidget(self.delay_input, 3, 1)
        
        # Max Current
        param_layout.addWidget(QLabel("Max Current (A):"), 4, 0)
        self.max_current_input = QLineEdit("0.5")
        param_layout.addWidget(self.max_current_input, 4, 1)
        
        sweep_group.setLayout(param_layout)
        control_panel.addWidget(sweep_group)
        
        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Start Sweep")
        self.start_button.clicked.connect(self.start_sweep)
        action_layout.addWidget(self.start_button)
        
        self.abort_button = QPushButton("Abort Sweep")
        self.abort_button.clicked.connect(self.abort_sweep_func)
        action_layout.addWidget(self.abort_button)
        
        self.save_data_button = QPushButton("Save Data")
        self.save_data_button.clicked.connect(lambda: self.save_data_to_csv())
        action_layout.addWidget(self.save_data_button)
        
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_plot)
        action_layout.addWidget(self.save_plot_button)
        
        # Add simulation mode button
        self.sim_mode_button = QPushButton("Toggle Simulation Mode")
        self.sim_mode_button.clicked.connect(self.toggle_simulation_mode)
        action_layout.addWidget(self.sim_mode_button)
        
        action_group.setLayout(action_layout)
        control_panel.addWidget(action_group)
        
        # Add stretch to push everything to the top
        control_panel.addStretch()
        
        # Right panel - Plot
        plot_panel = QVBoxLayout()
        
        # Create the figure for plotting
        self.figure = Figure(figsize=(5, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_panel.addWidget(self.toolbar)
        plot_panel.addWidget(self.canvas)
        
        # Create initial empty plot
        self.create_initial_plot()
        
        # Assemble the sweep tab layout
        control_widget = QWidget()
        control_widget.setLayout(control_panel)
        
        plot_widget = QWidget()
        plot_widget.setLayout(plot_panel)
        
        # Use splitter for resizable layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_widget)
        splitter.addWidget(plot_widget)
        splitter.setSizes([300, 700])  # Default sizes
        
        sweep_layout.addWidget(splitter)
        self.sweep_tab.setLayout(sweep_layout)

    def setup_analysis_tab(self):
        analysis_layout = QVBoxLayout()
        
        # Controls for loading data
        load_group = QGroupBox("Load Saved Data")
        load_layout = QHBoxLayout()
        
        self.load_data_button = QPushButton("Load Data from CSV")
        self.load_data_button.clicked.connect(self.load_data_from_csv)
        load_layout.addWidget(self.load_data_button)
        
        load_group.setLayout(load_layout)
        analysis_layout.addWidget(load_group)
        
        # Plot area for analysis
        self.analysis_figure = Figure(figsize=(5, 8), dpi=100)
        self.analysis_canvas = FigureCanvas(self.analysis_figure)
        self.analysis_toolbar = NavigationToolbar(self.analysis_canvas, self)
        
        analysis_layout.addWidget(self.analysis_toolbar)
        analysis_layout.addWidget(self.analysis_canvas)
        
        # Info box for displaying parameters
        info_group = QGroupBox("Measurement Information")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("No data loaded")
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        
        analysis_layout.addWidget(info_group)
        
        self.analysis_tab.setLayout(analysis_layout)

    def setup_debug_tab(self):
        """Setup the debug tab with resources list and connection info"""
        debug_layout = QVBoxLayout()
        
        # Resource list section
        resources_group = QGroupBox("VISA Resources")
        resources_layout = QVBoxLayout()
        
        self.resources_text = QTextEdit()
        self.resources_text.setReadOnly(True)
        
        # Display available resources if any
        if hasattr(self, 'available_resources'):
            self.resources_text.setText("\n".join(self.available_resources) if self.available_resources else "No VISA resources found")
        else:
            self.resources_text.setText("VISA Resource Manager not initialized")
        
        refresh_button = QPushButton("Refresh Resources List")
        refresh_button.clicked.connect(self.refresh_resources)
        
        resources_layout.addWidget(self.resources_text)
        resources_layout.addWidget(refresh_button)
        resources_group.setLayout(resources_layout)
        debug_layout.addWidget(resources_group)
        
        # Connection log section
        log_group = QGroupBox("Connection Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Add initial log entry
        self.add_log("Application started. Ready to initialize instrument.")
        
        # Test connection button
        test_button = QPushButton("Test GPIB Connection")
        test_button.clicked.connect(self.test_gpib_connection)
        log_layout.addWidget(test_button)
        
        log_group.setLayout(log_layout)
        debug_layout.addWidget(log_group)
        
        self.debug_tab.setLayout(debug_layout)

    def add_log(self, message):
        """Add a message to the debug log with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        # Also print to console for additional debugging
        print(f"[{timestamp}] {message}")

    def refresh_resources(self):
        """Refresh the list of available VISA resources"""
        if self.rm:
            try:
                self.available_resources = self.rm.list_resources()
                if self.available_resources:
                    self.resources_text.setText("\n".join(self.available_resources))
                    self.add_log(f"Resources refreshed: {len(self.available_resources)} found")
                else:
                    self.resources_text.setText("No VISA resources found")
                    self.add_log("Resources refreshed: None found")
            except Exception as e:
                self.resources_text.setText(f"Error refreshing resources: {str(e)}")
                self.add_log(f"Error refreshing resources: {str(e)}")
        else:
            self.resources_text.setText("VISA Resource Manager not initialized")
            self.add_log("Cannot refresh resources - VISA not initialized")

    def test_gpib_connection(self):
        """Test the GPIB connection using the current address"""
        if self.simulation_mode:
            self.add_log("Cannot test connection in simulation mode")
            return
            
        if not self.rm:
            self.add_log("VISA Resource Manager not initialized")
            return
            
        gpib_address = self.gpib_address_input.text().strip()
        if not gpib_address:
            self.add_log("Please enter a GPIB address")
            return
            
        self.add_log(f"Testing connection to GPIB0::{gpib_address}::INSTR...")
        
        try:
            # Try to open the resource with a short timeout
            inst = self.rm.open_resource(f"GPIB0::{gpib_address}::INSTR")
            inst.timeout = 3000  # 3 seconds timeout for testing
            
            # Try to get IDN
            try:
                idn = inst.query("*IDN?")
                self.add_log(f"Connection successful! Device identified as: {idn.strip()}")
            except Exception as e:
                self.add_log(f"Connected to device but failed to get identification: {str(e)}")
                
            # Close the resource
            inst.close()
            
        except Exception as e:
            self.add_log(f"Connection test failed: {str(e)}")
            self.add_log("Try the following troubleshooting steps:")
            self.add_log("1. Check if the instrument is powered on")
            self.add_log("2. Verify the GPIB cable connections")
            self.add_log("3. Try a different GPIB address if known")
            self.add_log("4. Make sure no other software is using the instrument")
            self.add_log("5. Check if GPIB controller/adapter is properly installed")

    def toggle_simulation_mode(self):
        """Toggle between simulation mode and real hardware mode"""
        self.simulation_mode = not self.simulation_mode
        
        if self.simulation_mode:
            self.status_label.setText("SIMULATION MODE ON - No hardware control")
            self.add_log("Switched to simulation mode - hardware commands will be simulated")
        else:
            self.status_label.setText("SIMULATION MODE OFF - Will attempt real hardware control")
            self.add_log("Switched to hardware mode - will attempt to control real hardware")
        
        # If we were connected, close the connection when going to simulation mode
        if self.simulation_mode and self.keithley:
            try:
                self.keithley.close()
                self.keithley = None
                self.add_log("Closed existing hardware connection")
            except Exception as e:
                self.add_log(f"Error closing connection: {str(e)}")

    def create_initial_plot(self):
        """Create the initial empty plot."""
        self.figure.clear()
        
        # Create a figure with shared x-axis
        self.ax1 = self.figure.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        
        # Labels
        self.ax1.set_xlabel("Voltage (V)")
        self.ax1.set_ylabel("Current (A)", color='blue')
        self.ax2.set_ylabel("Power (W)", color='red')
        
        self.ax1.set_title("I-V and P-V Curves")
        self.figure.tight_layout()
        self.canvas.draw()

    def initialize_keithley(self):
        """Initialize the Keithley 2401 SMU using the GPIB address."""
        if self.simulation_mode:
            self.add_log("Initializing Keithley in SIMULATION mode")
            QMessageBox.information(self, "Simulation", "Keithley initialized in simulation mode.")
            self.status_label.setText("SIMULATION MODE: Keithley initialized (simulated)")
            return
            
        # Get GPIB address from input field
        gpib_address = self.gpib_address_input.text().strip()
        if not gpib_address:
            QMessageBox.warning(self, "Error", "Please enter a GPIB address.")
            return
            
        # Store the current GPIB address
        self.GPIB_ADDRESS = int(gpib_address)
        
        try:
            # Make sure we have a resource manager
            if not self.rm:
                self.rm = pyvisa.ResourceManager()
                self.add_log("Created new VISA Resource Manager")
                
            # Try to open the resource
            resource_name = f"GPIB0::{gpib_address}::INSTR"
            self.add_log(f"Attempting to open {resource_name}")
            
            self.keithley = self.rm.open_resource(resource_name)
            self.keithley.timeout = 20000  # 20 seconds timeout
            
            # Reset the instrument
            self.add_log("Sending *RST to instrument")
            self.keithley.write("*RST")
            
            # Get identification info to verify connection
            self.add_log("Querying instrument identification")
            idn = self.keithley.query("*IDN?")
            
            self.add_log(f"Instrument identified as: {idn.strip()}")
            QMessageBox.information(self, "Success", f"Keithley initialized successfully.\n{idn}")
            self.status_label.setText(f"Connected to: {idn.strip()}")
            
        except pyvisa.errors.VisaIOError as e:
            error_message = str(e)
            self.add_log(f"VISA IO Error: {error_message}")
            
            # Provide more helpful messages based on error codes
            if "VI_ERROR_RSRC_NFOUND" in error_message:
                message = (
                    f"Could not find instrument at GPIB address {gpib_address}.\n\n"
                    "Please check:\n"
                    "1. The instrument is powered on\n"
                    "2. The GPIB cable is connected properly\n"
                    "3. The correct GPIB address is set on the instrument\n"
                    "4. GPIB controller/interface is installed and functioning\n\n"
                    "You can check available resources in the Debug tab."
                )
            elif "VI_ERROR_TMO" in error_message:
                message = "Timeout error communicating with the instrument."
            else:
                message = f"VISA error: {error_message}"
                
            QMessageBox.critical(self, "Connection Error", message)
            self.status_label.setText(f"Error: {message}")
            
        except Exception as e:
            self.add_log(f"General error initializing Keithley: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to initialize Keithley 2401: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")

    def setupIV(self):
        """Configure Keithley for I-V testing in voltage sweep mode."""
        if self.simulation_mode:
            self.add_log("Simulating IV setup")
            return True
            
        try:
            # Always use voltage sweep mode for MPPT
            self.keithley.write(':SOUR:FUNC VOLT')  # Source voltage
            self.keithley.write(':SENS:FUNC "CURR"')  # Measure current
            self.add_log("Set instrument to voltage source, current measure mode")
            
            # Set sensing mode based on checkbox
            if self.four_wire_check.isChecked():
                self.keithley.write(":SYST:RSEN ON")  # 4-wire mode
                self.add_log("Enabled 4-wire sensing")
            else:
                self.keithley.write(":SYST:RSEN OFF")  # 2-wire mode
                self.add_log("Using 2-wire sensing")
            
            # Always use front terminals (removed rear option as requested)
            self.keithley.write(":ROUT:TERM FRONT")  # Front terminals
            self.add_log("Set to use front terminals")
            
            # Set the maximum current compliance limit
            max_current = self.max_current_input.text().strip()
            if max_current:
                self.keithley.write(f":SENS:CURR:PROT {max_current}")  # Set current compliance
                self.add_log(f"Set current compliance to {max_current}A")
            
            # Turn output ON
            self.keithley.write(':OUTP ON')
            self.add_log("Turned output ON")
            
            # Zero the output initially
            self.keithley.write(':SOUR:VOLT 0')
            self.add_log("Set initial voltage to 0V")
            
            return True
        except Exception as e:
            self.add_log(f"Error in setupIV: {str(e)}")
            QMessageBox.critical(self, "Setup Error", f"Failed to setup Keithley: {str(e)}")
            self.status_label.setText(f"Setup Error: {str(e)}")
            return False

    def start_sweep(self):
        """Start the voltage sweep procedure."""
        if not self.keithley and not self.simulation_mode:
            QMessageBox.warning(self, "Error", "Please initialize the Keithley first.")
            return
        
        try:
            # Reset abort flag
            self.abort_sweep = False
            
            # Get sweep parameters
            start = float(self.start_voltage.text())
            stop = float(self.stop_voltage.text())
            points = int(self.points_input.text())
            delay = float(self.delay_input.text())
            
            self.add_log(f"Starting sweep: {start}V to {stop}V, {points} points, {delay}s delay")
            
            # Setup the instrument for IV sweep
            if self.simulation_mode or self.setupIV():
                # Perform the sweep
                self.status_label.setText("Running sweep...")
                self.voltage_sweep(start, stop, points, delay)
                self.status_label.setText("Sweep completed.")
            
        except ValueError as e:
            self.add_log(f"Input error in start_sweep: {str(e)}")
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for all parameters.")
        except Exception as e:
            self.add_log(f"Error in start_sweep: {str(e)}")
            QMessageBox.critical(self, "Sweep Error", f"Error during sweep: {str(e)}")
            self.status_label.setText(f"Sweep Error: {str(e)}")

    def abort_sweep_func(self):
        """Abort the current sweep operation."""
        self.abort_sweep = True
        self.add_log("User requested sweep abort")
        self.status_label.setText("Aborting sweep...")

    def voltage_sweep(self, start, stop, points, delay):
        """Perform voltage sweep and measure current."""
        sweep_values = np.linspace(start, stop, points)
        self.voltages = []  # Clear previous data
        self.currents = []
        self.powers = []
        
        # Create progress tracking
        total_points = len(sweep_values)
        
        for idx, voltage in enumerate(sweep_values):
            if self.abort_sweep:
                self.add_log("Sweep aborted by user")
                QMessageBox.information(self, "Sweep Aborted", "The sweep was aborted.")
                break
            
            try:
                # Update status with progress
                progress_msg = f"Sweeping: {idx+1}/{total_points} points. Voltage: {voltage:.3f} V"
                self.status_label.setText(progress_msg)
                if idx % 5 == 0:  # Log every 5th point to avoid excessive logging
                    self.add_log(progress_msg)
                QApplication.processEvents()  # Keep the UI responsive
                
                if self.simulation_mode:
                    # In simulation mode, generate synthetic data
                    # Simulate a solar cell I-V curve
                    # Simple model: I = Isc * (1 - exp((V-Voc)/Vt))
                    isc = 0.5  # Short circuit current
                    voc = stop * 0.8  # Open circuit voltage
                    vt = 0.6  # Thermal voltage
                    
                    if voltage >= voc:
                        current = 0.0
                    else:
                        # Add some noise
                        noise = np.random.normal(0, 0.01)
                        current = isc * (1 - np.exp((voltage - voc) / vt)) + noise
                        if current < 0:
                            current = 0
                    
                    measured_voltage = voltage
                    measured_current = current
                    
                    # Simulate measurement delay
                    time.sleep(delay / 10)  # Faster in simulation
                else:
                    # Real hardware control
                    # Set the source voltage
                    self.keithley.write(f':SOUR:VOLT {voltage}')
                    
                    # Allow settling time
                    time.sleep(delay)
                    
                    # Measure the values
                    response = self.keithley.query(':READ?').strip()
                    values = response.split(',')
                    
                    if len(values) >= 2:
                        measured_voltage = float(values[0])
                        measured_current = float(values[1]) * -1  # Invert current as requested
                    else:
                        raise ValueError("Invalid measurement response")
                
                # Calculate power
                power = measured_voltage * measured_current
                
                # Store the data
                self.voltages.append(measured_voltage)
                self.currents.append(measured_current)
                self.powers.append(power)
                
                # Update the plot with new data point
                if idx % 3 == 0 or idx == total_points - 1:  # Update less frequently for better performance
                    self.update_live_plot()
                
            except Exception as e:
                error_msg = f"Error at voltage {voltage}: {str(e)}"
                self.add_log(error_msg)
                print(error_msg)
                # Continue with the next point instead of stopping the whole sweep
        
        # Turn off output when done
        if not self.simulation_mode:
            try:
                self.keithley.write(':SOUR:VOLT 0')  # Set voltage to zero first
                self.keithley.write(':OUTP OFF')      # Turn off output
                self.add_log("Sweep complete. Output turned OFF")
            except Exception as e:
                self.add_log(f"Error turning off output: {str(e)}")
        
        # Final plot update with all data
        if len(self.voltages) > 0:
            self.update_live_plot()
            self.find_mppt_point()
            self.add_log(f"Sweep completed with {len(self.voltages)} valid data points")
        else:
            self.add_log("Sweep completed but no valid data points were collected")
            QMessageBox.warning(self, "Sweep Error", "No valid data points were collected during the sweep.")

    def update_live_plot(self):
        """Update the plot with current data."""
        if not self.voltages or not self.currents:
            return
            
        self.figure.clear()
        
        # Create axes with shared x-axis
        ax1 = self.figure.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Plot I-V curve (blue)
        ax1.plot(self.voltages, self.currents, 'b-', marker='.', label="Current")
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("Current (A)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot P-V curve (red) on secondary y-axis
        ax2.plot(self.voltages, self.powers, 'r-', marker='.', label="Power")
        ax2.set_ylabel("Power (W)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add title and grid
        ax1.set_title("I-V and P-V Curves")
        ax1.grid(True)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        self.figure.tight_layout()
        self.canvas.draw()

    def find_mppt_point(self):
        """Find and mark the Maximum Power Point."""
        if not self.powers:
            return
            
        max_power_idx = np.argmax(self.powers)
        max_power = self.powers[max_power_idx]
        mpp_voltage = self.voltages[max_power_idx]
        mpp_current = self.currents[max_power_idx]
        
        # Log MPP data
        self.add_log(f"MPP found: Voltage={mpp_voltage:.3f}V, Current={mpp_current:.3f}A, Power={max_power:.3f}W")
        
        # Update the status label
        self.status_label.setText(f"MPP: Voltage={mpp_voltage:.3f}V, Current={mpp_current:.3f}A, Power={max_power:.3f}W")
        
        # Highlight the MPP on the plot
        self.figure.clear()
        
        # Create axes with shared x-axis
        ax1 = self.figure.add_subplot(111)
        ax2 = ax1.twinx()
                # Plot I-V curve (blue)
        ax1.plot(self.voltages, self.currents, 'b-', marker='.', label="Current")
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("Current (A)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot P-V curve (red) on secondary y-axis
        ax2.plot(self.voltages, self.powers, 'r-', marker='.', label="Power")
        ax2.set_ylabel("Power (W)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Mark the MPP
        ax1.plot(mpp_voltage, mpp_current, 'gs', markersize=10, label="MPP")
        ax2.plot(mpp_voltage, max_power, 'gs', markersize=10)
        
        # Add annotations
        ax2.annotate(f"MPP: {max_power:.3f}W @ {mpp_voltage:.3f}V",
                     xy=(mpp_voltage, max_power),
                     xytext=(mpp_voltage, max_power*0.8),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     color='green',
                     fontweight='bold')
        
        # Add title and grid
        ax1.set_title("I-V and P-V Curves with Maximum Power Point")
        ax1.grid(True)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        self.figure.tight_layout()
        self.canvas.draw()

    def save_data_to_csv(self):
        """Save the data to a CSV file."""
        if not self.voltages or not self.currents:
            QMessageBox.warning(self, "No Data", "No data available to save.")
            return
            
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Data", 
                "", 
                "CSV Files (*.csv);;All Files (*)", 
                options=options
            )
            
            if file_path:
                # Get sweep parameters to save with data
                sweep_params = {
                    'Start_Voltage': self.start_voltage.text(),
                    'Stop_Voltage': self.stop_voltage.text(),
                    'Points': self.points_input.text(),
                    'Delay': self.delay_input.text(),
                    'Max_Current': self.max_current_input.text(),
                    'Four_Wire': str(self.four_wire_check.isChecked()),
                    'Measurement_Date': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'Simulation_Mode': str(self.simulation_mode)
                }
                
                # Create DataFrame for data
                data = {
                    'Voltage (V)': self.voltages,
                    'Current (A)': self.currents,
                    'Power (W)': self.powers
                }
                df = pd.DataFrame(data)
                
                # Save DataFrame to CSV
                df.to_csv(file_path, index=False)
                
                # Save parameters as a second file
                param_file = file_path.replace('.csv', '_params.csv')
                pd.DataFrame([sweep_params]).to_csv(param_file, index=False)
                
                self.add_log(f"Data saved to {file_path}")
                self.add_log(f"Parameters saved to {param_file}")
                QMessageBox.information(self, "Success", f"Data saved to {file_path}\nParameters saved to {param_file}")
                
        except Exception as e:
            self.add_log(f"Error saving data: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"Error saving data: {str(e)}")

    def save_plot(self):
        """Save the current plot as an image file."""
        if not self.voltages or not self.currents:
            QMessageBox.warning(self, "No Data", "No plot available to save.")
            return
            
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Plot", 
                "", 
                "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)", 
                options=options
            )
            
            if file_path:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.add_log(f"Plot saved to {file_path}")
                QMessageBox.information(self, "Success", f"Plot saved to {file_path}")
                
        except Exception as e:
            self.add_log(f"Error saving plot: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"Error saving plot: {str(e)}")

    def load_data_from_csv(self):
        """Load data from a previously saved CSV file."""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Load Data", 
                "", 
                "CSV Files (*.csv);;All Files (*)", 
                options=options
            )
            
            if file_path:
                self.add_log(f"Loading data from {file_path}")
                
                # Load the data file
                df = pd.read_csv(file_path)
                
                # Check if this is the data file or parameters file
                if 'Voltage (V)' in df.columns:
                    self.voltages = df['Voltage (V)'].tolist()
                    self.currents = df['Current (A)'].tolist()
                    self.powers = df['Power (W)'].tolist()
                    
                    # Try to load the corresponding parameters file
                    param_file = file_path.replace('.csv', '_params.csv')
                    if not param_file.endswith('_params.csv'):
                        param_file = file_path.replace('.csv', '_params.csv')
                    
                    try:
                        if os.path.exists(param_file):
                            self.add_log(f"Loading parameters from {param_file}")
                            param_df = pd.read_csv(param_file)
                            param_str = "Measurement Parameters:\n"
                            for col in param_df.columns:
                                param_str += f"{col}: {param_df[col][0]}\n"
                            self.info_label.setText(param_str)
                            
                            # Update UI fields with loaded parameters if available
                            try:
                                if 'Start_Voltage' in param_df.columns:
                                    self.start_voltage.setText(str(param_df['Start_Voltage'][0]))
                                if 'Stop_Voltage' in param_df.columns:
                                    self.stop_voltage.setText(str(param_df['Stop_Voltage'][0]))
                                if 'Points' in param_df.columns:
                                    self.points_input.setText(str(param_df['Points'][0]))
                                if 'Delay' in param_df.columns:
                                    self.delay_input.setText(str(param_df['Delay'][0]))
                                if 'Max_Current' in param_df.columns:
                                    self.max_current_input.setText(str(param_df['Max_Current'][0]))
                                if 'Four_Wire' in param_df.columns:
                                    self.four_wire_check.setChecked(param_df['Four_Wire'][0].lower() == 'true')
                            except Exception as ue:
                                self.add_log(f"Non-critical error updating UI from parameters: {str(ue)}")
                                
                        else:
                            self.add_log("Parameter file not found. Only data loaded.")
                            self.info_label.setText("Parameter file not found. Only data loaded.")
                    except Exception as pe:
                        self.add_log(f"Error loading parameters: {str(pe)}")
                        self.info_label.setText(f"Error loading parameters: {str(pe)}")
                    
                    # Plot the loaded data in the analysis tab
                    self.plot_analysis_data()
                    
                    # Switch to analysis tab
                    self.tabs.setCurrentIndex(1)
                    
                    QMessageBox.information(self, "Success", f"Data loaded from {file_path}")
                else:
                    self.add_log("Invalid file format - expected 'Voltage (V)' column")
                    QMessageBox.warning(self, "Invalid File", "The selected file does not contain the expected data format.")
                
        except Exception as e:
            self.add_log(f"Error loading data: {str(e)}")
            QMessageBox.critical(self, "Load Error", f"Error loading data: {str(e)}")

    def plot_analysis_data(self):
        """Plot the loaded data in the analysis tab."""
        if not self.voltages or not self.currents:
            return
            
        self.analysis_figure.clear()
        
        # Create axes with shared x-axis
        ax1 = self.analysis_figure.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Plot I-V curve (blue)
        ax1.plot(self.voltages, self.currents, 'b-', marker='.', label="Current")
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("Current (A)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot P-V curve (red) on secondary y-axis
        ax2.plot(self.voltages, self.powers, 'r-', marker='.', label="Power")
        ax2.set_ylabel("Power (W)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Find and mark the MPP
        max_power_idx = np.argmax(self.powers)
        max_power = self.powers[max_power_idx]
        mpp_voltage = self.voltages[max_power_idx]
        mpp_current = self.currents[max_power_idx]
        
        # Mark the MPP
        ax1.plot(mpp_voltage, mpp_current, 'gs', markersize=10, label="MPP")
        ax2.plot(mpp_voltage, max_power, 'gs', markersize=10)
        
        # Add annotations
        ax2.annotate(f"MPP: {max_power:.3f}W @ {mpp_voltage:.3f}V",
                     xy=(mpp_voltage, max_power),
                     xytext=(mpp_voltage, max_power*0.8),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     color='green',
                     fontweight='bold')
        
        # Add title and grid
        ax1.set_title("Loaded I-V and P-V Curves with Maximum Power Point")
        ax1.grid(True)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()
        self.add_log("Analysis plot updated with loaded data")

    def closeEvent(self, event):
        """Handle application close event."""
        try:
            if self.keithley:
                self.add_log("Closing application - turning off output and closing connection")
                self.keithley.write(':OUTP OFF')  # Ensure output is off when closing
                self.keithley.close()
                print("Keithley connection closed")
        except Exception as e:
            self.add_log(f"Error closing Keithley connection: {str(e)}")
        event.accept()

if __name__ == "__main__":
    # Add exception catching to make the app more stable
    import os
    import traceback
    
    def exception_hook(exctype, value, tb):
        """Handle uncaught exceptions to prevent crashes."""
        error_msg = ''.join(traceback.format_exception(exctype, value, tb))
        print(error_msg)
        QMessageBox.critical(None, "Error", f"An unexpected error occurred:\n{str(value)}\n\nPlease restart the application.")
        sys.__excepthook__(exctype, value, tb)
    
    sys.excepthook = exception_hook
    
    app = QApplication(sys.argv)
    ex = MPPTSweepApp()
    sys.exit(app.exec_())
