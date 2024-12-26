import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, QMessageBox, 
                            QLabel, QSpinBox, QDoubleSpinBox, QScrollArea, 
                            QGroupBox, QDateTimeEdit, QComboBox, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt
import numpy as np
import json
import os
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path
from datetime import datetime

class GTTFlowNet(QMainWindow):
    """Main window class for GTT FlowNet - Transaction Network Visualization Tool"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GTT FlowNet")
        self.setGeometry(100, 100, 1200, 800)
        
        # Settings file path
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyzer_settings.json')
        
        # Initialize data storage
        self.df = None
        self.G = nx.DiGraph()
        self.pos = None
        self.dragging = False
        self.selected_node = None
        self.selected_nodes = set()  # For multiple node selection
        self.force_strength = 0.1
        self.repulsion = 1000
        self.edge_length = 100
        self.arrow_scale = 20
        self.focused_node = None  # For node focus feature
        self.stats_text = None  # Initialize stats_text
        self.searched_node = None  # For search highlight
        self.right_button_pressed = False  # For right-click selection
        
        # Position history for undo feature
        self.position_history = []
        self.max_history = 50  # Maximum number of positions to store
        
        # Initialize draggable stats frame
        self.stats_dragging = False
        self.stats_pos = [0.02, 0.98]  # Default position
        self.stats_offset = (0, 0)  # Add offset for dragging
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create left control panel
        control_panel = self.create_control_panel()
        control_scroll = QScrollArea()
        control_scroll.setWidget(control_panel)
        control_scroll.setWidgetResizable(True)
        control_scroll.setFixedWidth(300)
        
        # Create visualization panel
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        
        # Create matplotlib figure with larger size
        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, viz_panel)
        
        # Connect mouse events
        self.connect_events()
        
        # Connect additional mouse events for stats dragging
        self.canvas.mpl_connect('button_press_event', self.on_stats_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_stats_move)
        self.canvas.mpl_connect('button_release_event', self.on_stats_release)
        
        # Add widgets to visualization layout
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        # Add panels to main layout
        main_layout.addWidget(control_scroll)
        main_layout.addWidget(viz_panel, stretch=1)
        
        # Setup animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_layout)
        self.animation_running = False
        
        # Load saved settings
        self.load_settings()
        
    def closeEvent(self, event):
        """Save settings when closing the application"""
        self.save_settings()
        event.accept()
        
    def save_settings(self):
        """Save current settings to file"""
        settings = {
            'force_strength': self.force_strength,
            'repulsion': self.repulsion,
            'edge_length': self.edge_length,
            'arrow_scale': self.arrow_scale,
            'min_transaction': self.min_transaction_spin.value(),
            'min_aggregated': self.min_aggregated_spin.value(),
            'weight_type': self.weight_type_combo.currentText(),
            'weight_factor': self.weight_factor_spin.value(),
            'edge_width': self.edge_width_spin.value(),
            'arrow_size': self.arrow_size_spin.value(),
            'analytics': self.analytics_combo.currentText()
        }
        
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Error saving settings: {e}")
            
    def load_settings(self):
        """Load settings from file"""
        if not os.path.exists(self.settings_file):
            return
            
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                
            # Apply loaded settings
            self.force_strength = settings.get('force_strength', 0.1)
            self.repulsion = settings.get('repulsion', 1000)
            self.edge_length = settings.get('edge_length', 100)
            self.arrow_scale = settings.get('arrow_scale', 20)
            
            # Update UI controls after window is shown
            QTimer.singleShot(0, lambda: self.apply_loaded_settings(settings))
        except Exception as e:
            print(f"Error loading settings: {e}")
            
    def apply_loaded_settings(self, settings):
        """Apply loaded settings to UI controls"""
        self.min_transaction_spin.setValue(settings.get('min_transaction', 0))
        self.min_aggregated_spin.setValue(settings.get('min_aggregated', 0))
        
        weight_type = settings.get('weight_type', "Total Volume")
        index = self.weight_type_combo.findText(weight_type)
        if index >= 0:
            self.weight_type_combo.setCurrentIndex(index)
            
        self.weight_factor_spin.setValue(settings.get('weight_factor', 100))
        self.edge_width_spin.setValue(settings.get('edge_width', 2))
        self.arrow_size_spin.setValue(settings.get('arrow_size', 10))
        
        analytics = settings.get('analytics', "None")
        index = self.analytics_combo.findText(analytics)
        if index >= 0:
            self.analytics_combo.setCurrentIndex(index)
            
    def toggle_animation(self):
        """Toggle the force-directed layout animation"""
        if not self.animation_running:
            self.start_button.setText("Stop Animation")
            self.animation_timer.start(50)  # Update every 50ms
            self.animation_running = True
        else:
            self.start_button.setText("Start Animation")
            self.animation_timer.stop()
            self.animation_running = False
            # Force an immediate update
            self.draw_network()
            self.canvas.draw()
            # Reset dragging state
            self.dragging = False  # Reset dragging state
            # Ensure nodes can be selected after stopping animation
            self.selected_node = None
            
    def update_layout(self):
        """Update force-directed layout"""
        if not self.pos or not self.G:
            return

        if self.dragging:  # Skip force calculation if dragging
            return

        # Save current state before force-directed update
        self.save_position_state()

        # Calculate forces and update positions
        for node in self.G.nodes():
            # Skip selected nodes during animation to maintain their positions
            if node in self.selected_nodes:
                continue

            # Initialize force components
            fx = 0
            fy = 0

            # Repulsive force from other nodes
            for other in self.G.nodes():
                if other != node:
                    dx = self.pos[node][0] - self.pos[other][0]
                    dy = self.pos[node][1] - self.pos[other][1]
                    dist = max(0.01, np.sqrt(dx * dx + dy * dy))
                    force = self.repulsion / (dist * dist)
                    fx += force * dx / dist
                    fy += force * dy / dist

            # Spring force from edges
            for neighbor in self.G.neighbors(node):
                dx = self.pos[neighbor][0] - self.pos[node][0]
                dy = self.pos[neighbor][1] - self.pos[node][1]
                dist = max(0.01, np.sqrt(dx * dx + dy * dy))
                force = (dist - self.edge_length) * 0.1
                fx += force * dx / dist
                fy += force * dy / dist

            # Update position with damping
            self.pos[node] = (
                self.pos[node][0] + fx * self.force_strength,
                self.pos[node][1] + fy * self.force_strength
            )

        self.draw_network()
        self.canvas.draw()

    def on_mouse_press(self, event):
        """Handle mouse press events"""
        if event.inaxes != self.ax or not self.G or not self.pos:
            return

        # Get clicked node
        clicked_node = None
        min_dist = float('inf')
        for node in self.G.nodes():
            x, y = self.pos[node]
            dist = np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2)
            if dist < 0.15:  # Increased threshold for easier selection
                if dist < min_dist:
                    clicked_node = node
                    min_dist = dist

        if clicked_node is not None:
            if event.button == 1:  # Left click
                # Save current state before modifying positions
                self.save_position_state()
                
                # Clear search highlight when selecting a different node
                if self.searched_node and clicked_node != self.searched_node:
                    self.searched_node = None

                # Update selection
                modifiers = QApplication.keyboardModifiers()
                if modifiers == Qt.ControlModifier:
                    if clicked_node in self.selected_nodes:
                        self.selected_nodes.remove(clicked_node)
                    else:
                        self.selected_nodes.add(clicked_node)
                else:
                    self.selected_nodes = {clicked_node}
                
                # Update dragging state
                self.selected_node = clicked_node
                self.dragging = True
            
            elif event.button == 3:  # Right click
                self.right_button_pressed = True
                self.selected_nodes.add(clicked_node)
            
            # Force immediate update
            self.draw_network()
            self.canvas.draw()

    def on_mouse_move(self, event):
        """Handle mouse movement events"""
        if not event.inaxes:
            return

        if self.right_button_pressed:
            # Add nodes under cursor to selection
            clicked_node = None
            min_dist = float('inf')
            for node in self.G.nodes():
                x, y = self.pos[node]
                dist = np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2)
                if dist < 0.15 and dist < min_dist:
                    clicked_node = node
                    min_dist = dist
            
            if clicked_node is not None and clicked_node not in self.selected_nodes:
                self.selected_nodes.add(clicked_node)
                self.draw_network()
                self.canvas.draw()
        
        elif self.dragging and self.selected_node:
            # Update position of dragged node
            self.pos[self.selected_node] = (event.xdata, event.ydata)
            self.draw_network()
            self.canvas.draw()

    def on_mouse_release(self, event):
        """Handle mouse release events"""
        if event.button == 3:  # Right click release
            self.right_button_pressed = False
        elif event.button == 1 and self.dragging:  # Left click release
            self.dragging = False
            # Force immediate update after drag
            self.draw_network()
            self.canvas.draw()

    def connect_events(self):
        """Connect all matplotlib event handlers"""
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('button_press_event', self.on_double_click)

    def on_double_click(self, event):
        if event.button == 3 and len(self.selected_nodes) >= 2:  # Right double-click
            self.export_selected_transactions()
    
    def export_selected_transactions(self, single_node=None):
        """Export transactions for selected nodes or a single node"""
        if self.df is None:
            return
            
        if single_node:
            # Export all transactions involving the single node
            mask = (self.df['Sender'] == single_node) | (self.df['Recipient'] == single_node)
            selected_transactions = self.df[mask].copy()
            export_desc = f"transactions for {single_node}"
        else:
            # Export transactions between selected nodes
            if len(self.selected_nodes) < 2:
                QMessageBox.warning(self, "Selection Required", 
                                 "Please select at least 2 nodes to export transactions between them")
                return
                
            mask = (self.df['Sender'].isin(self.selected_nodes) & 
                   self.df['Recipient'].isin(self.selected_nodes))
            selected_transactions = self.df[mask].copy()
            export_desc = f"transactions between {len(self.selected_nodes)} selected nodes"
        
        if len(selected_transactions) > 0:
            # Reorder columns to show account numbers before account names
            columns_order = [
                'Date',
                'PrimaryAccountNumber',
                'PrimaryAccountName',
                'SenderAccountNumber',
                'Sender',
                'RecipientAccountNumber',
                'Recipient',
                'Amount',
                'TransactionID'
            ]
            
            # Ensure all required columns exist
            selected_transactions = selected_transactions[columns_order]
            
            try:
                # Get save file name
                file_name, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Transactions",
                    "",
                    "Excel files (*.xlsx)"
                )
                
                if file_name:
                    # Add .xlsx extension if not present
                    if not file_name.endswith('.xlsx'):
                        file_name += '.xlsx'
                        
                    # Create Excel writer with datetime formatting
                    with pd.ExcelWriter(file_name, engine='openpyxl', datetime_format='YYYY-MM-DD') as writer:
                        selected_transactions.to_excel(writer, index=False, sheet_name='Transactions')
                        
                        # Auto-adjust column widths
                        worksheet = writer.sheets['Transactions']
                        for idx, col in enumerate(selected_transactions.columns):
                            max_length = max(
                                selected_transactions[col].astype(str).apply(len).max(),
                                len(str(col))
                            ) + 2
                            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)
                    
                    QMessageBox.information(self, "Export Successful",
                                         f"Successfully exported {len(selected_transactions)} {export_desc}")
                    
            except Exception as e:
                QMessageBox.critical(self, "Export Error",
                                  f"Error exporting transactions: {str(e)}")
        else:
            QMessageBox.information(self, "No Transactions",
                                 f"No transactions found {export_desc}")
    
    def load_file(self):
        """Load and process the Excel file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel File",
            "",
            "Excel Files (*.xlsx *.xls)"
        )
        
        if file_name:
            try:
                self.df = pd.read_excel(file_name)
                if 'Date' in self.df.columns:
                    # Convert dates to datetime
                    self.df['Date'] = pd.to_datetime(self.df['Date'])
                    min_date = self.df['Date'].min()
                    max_date = self.df['Date'].max()
                    
                    # Set the date range in the UI
                    self.start_date.setDateTime(min_date.to_pydatetime())
                    self.end_date.setDateTime(max_date.to_pydatetime())
                
                self.status_label.setText(f"Loaded: {os.path.basename(file_name)}")
                self.process_data()
            except Exception as e:
                self.status_label.setText(f"Error loading file: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    
    def process_data(self):
        """Process the data and update the network"""
        if self.df is None:
            return
            
        try:
            # Get date range from UI
            start_date = self.start_date.dateTime().toPyDateTime()
            end_date = self.end_date.dateTime().toPyDateTime()
            
            # Filter data by date range
            mask = (self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)
            filtered_df = self.df[mask]
            
            # Filter by transaction amounts
            min_transaction = self.min_transaction_spin.value()
            min_aggregated = self.min_aggregated_spin.value()
            
            if min_transaction > 0:
                filtered_df = filtered_df[filtered_df['Amount'] >= min_transaction]
            
            # Create network
            self.G = nx.DiGraph()
            
            # Add edges with weights
            for _, row in filtered_df.iterrows():
                source = row['Sender']  
                target = row['Recipient']  
                amount = row['Amount']
                
                # Add or update edge
                if self.G.has_edge(source, target):
                    self.G[source][target]['weight'] += amount
                else:
                    self.G.add_edge(source, target, weight=amount)
            
            # Filter by aggregated amount
            if min_aggregated > 0:
                edges_to_remove = []
                for u, v, data in self.G.edges(data=True):
                    if data['weight'] < min_aggregated:
                        edges_to_remove.append((u, v))
                self.G.remove_edges_from(edges_to_remove)
            
            # Remove isolated nodes
            self.G.remove_nodes_from(list(nx.isolates(self.G)))
            
            # Initialize layout if needed
            if not self.pos or len(self.pos) != len(self.G.nodes()):
                self.pos = nx.spring_layout(self.G)
            
            self.update_visualization()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing data: {str(e)}")
    
    def calculate_node_sizes(self):
        """Calculate node sizes based on selected weighting"""
        if not self.G:
            return {}
            
        weight_type = self.weight_type_combo.currentText()
        weight_factor = self.weight_factor_spin.value()
        
        sizes = {}
        for node in self.G.nodes():
            if weight_type == "Total Volume":
                size = (sum(d['weight'] for _, _, d in self.G.in_edges(node, data=True)) +
                       sum(d['weight'] for _, _, d in self.G.out_edges(node, data=True)))
            elif weight_type == "Outgoing Volume":
                size = sum(d['weight'] for _, _, d in self.G.out_edges(node, data=True))
            else:  # Incoming Volume
                size = sum(d['weight'] for _, _, d in self.G.in_edges(node, data=True))
            sizes[node] = size
        
        # Normalize sizes
        if sizes:
            max_size = max(sizes.values())
            min_size = 500  # Minimum node size
            if max_size > 0:
                sizes = {k: min_size + (v / max_size) * weight_factor * 100 for k, v in sizes.items()}
            else:
                sizes = {k: min_size for k in sizes}
        
        return sizes
    
    def update_visualization(self):
        """Update the network visualization"""
        if self.G is None or self.G.number_of_nodes() == 0:
            return
        self.draw_network()
        self.canvas.draw()
    
    def draw_network(self):
        """Draw the network visualization"""
        if not self.G or self.G.number_of_nodes() == 0:
            return
            
        self.ax.clear()
        
        # Calculate node colors and sizes
        node_colors = []
        for node in self.G.nodes():
            if node == self.searched_node:
                node_colors.append('yellow')  # Highlight searched node
            elif node in self.selected_nodes:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightblue')
        
        node_sizes = self.calculate_node_sizes()
        
        # Set node border colors
        node_edge_colors = ['red' if node == self.searched_node else 'orange' if node in self.selected_nodes else 'gray' for node in self.G.nodes()]
        node_line_widths = [3 if node == self.searched_node else 2 if node in self.selected_nodes else 1 for node in self.G.nodes()]
        
        # Draw edges with arrows
        edge_width = self.edge_width_spin.value()
        arrow_size = self.arrow_size_spin.value()
        
        # Draw edges with curved arrows and labels
        for (u, v, d) in self.G.edges(data=True):
            # Get edge weight for width
            weight = d.get('weight', 1.0)
            
            # Calculate edge width based on weight
            width = edge_width * (weight / max(d.get('weight', 1.0) for _, _, d in self.G.edges(data=True)))
            
            # Determine edge color based on node selection
            edge_color = 'red' if (u in self.selected_nodes and v in self.selected_nodes) else 'gray'
            edge_alpha = 0.8 if (u in self.selected_nodes or v in self.selected_nodes) else 0.5
            
            # Create curved arrow
            pos_u = self.pos[u]
            pos_v = self.pos[v]
            
            # Calculate curve control point
            mid_point = ((pos_u[0] + pos_v[0])/2, (pos_u[1] + pos_v[1])/2)
            control_point = (mid_point[0] - (pos_v[1] - pos_u[1])*0.2,
                           mid_point[1] + (pos_v[0] - pos_u[0])*0.2)
            
            # Create Bezier curve
            path = Path([pos_u, control_point, pos_v], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
            patch = PathPatch(path, facecolor='none', edgecolor=edge_color, 
                            alpha=edge_alpha, linewidth=width)
            self.ax.add_patch(patch)
            
            # Calculate arrow positions (30% and 70% along the path)
            vertices = path.vertices
            t = 0.7  # Position along the curve (0 to 1)
            arrow_pos = (
                (1-t)**2 * vertices[0] + 
                2*(1-t)*t * vertices[1] + 
                t**2 * vertices[2]
            )
            direction = (
                2*(1-t)*(vertices[1] - vertices[0]) + 
                2*t*(vertices[2] - vertices[1])
            )
            direction = direction / np.linalg.norm(direction)
            
            # Add arrow with larger size for better visibility
            arrow = FancyArrowPatch(
                (arrow_pos[0] - direction[0]*0.05, arrow_pos[1] - direction[1]*0.05),
                (arrow_pos[0] + direction[0]*0.05, arrow_pos[1] + direction[1]*0.05),
                arrowstyle='-|>',
                mutation_scale=arrow_size * 1.5,  # Increased arrow size
                color=edge_color,
                alpha=edge_alpha,
                linewidth=width
            )
            self.ax.add_patch(arrow)
            
            # Add amount label
            if weight >= 1000000:  # If amount is 1M or more
                amount_str = f'${weight/1000000:.1f}M'
            else:
                amount_str = f'${weight/1000:.0f}K'
            
            # Position the label above the edge
            label_pos = (
                (1-0.5)**2 * vertices[0] + 
                2*(1-0.5)*0.5 * vertices[1] + 
                0.5**2 * vertices[2]
            )
            
            # Add white background to text for better visibility
            self.ax.text(label_pos[0], label_pos[1], amount_str,
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize=8,
                        bbox=dict(facecolor='white', 
                                edgecolor='none',
                                alpha=0.7,
                                pad=1))
        
        # Draw nodes with distinct borders for selected nodes
        nx.draw_networkx_nodes(self.G, self.pos,
                             node_color=node_colors,
                             node_size=[node_sizes[node] for node in self.G.nodes()],
                             alpha=0.6,
                             linewidths=node_line_widths,
                             edgecolors=node_edge_colors,
                             ax=self.ax)
        
        # Draw labels with smaller font size and white background
        for node, pos in self.pos.items():
            self.ax.text(pos[0], pos[1], node,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8,
                        bbox=dict(facecolor='white',
                                edgecolor='none',
                                alpha=0.7,
                                pad=1))
        
        # Add network statistics in draggable frame
        stats_text = f'Nodes: {self.G.number_of_nodes()}\n'
        stats_text += f'Edges: {self.G.number_of_edges()}\n'
        if self.G.number_of_edges() > 0:
            stats_text += f'Total Volume: ${sum(d["weight"] for _, _, d in self.G.edges(data=True))/1000000:.2f}M\n'
            stats_text += f'Avg Transaction: ${np.mean([d["weight"] for _, _, d in self.G.edges(data=True)])/1000:.2f}K'
        
        # Store the text artist for dragging
        self.stats_text = self.ax.text(
            self.stats_pos[0], self.stats_pos[1],
            stats_text,
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.8,
                edgecolor='gray',
                pad=0.5
            )
        )
        
        # Update canvas
        self.canvas.draw()
    
    def on_stats_press(self, event):
        if event.inaxes != self.ax:
            return
        
        # Check if click is near stats text
        if abs(event.xdata - self.stats_pos[0]) < 0.1 and abs(event.ydata - self.stats_pos[1]) < 0.1:
            self.stats_dragging = True
    
    def on_stats_move(self, event):
        if self.stats_dragging and event.inaxes == self.ax:
            self.stats_pos = [event.xdata, event.ydata]
            self.update_visualization()
    
    def on_stats_release(self, event):
        self.stats_dragging = False

    def save_position_state(self):
        """Save current positions to history"""
        if self.pos:
            # Create a deep copy of current positions
            current_state = {node: (x, y) for node, (x, y) in self.pos.items()}
            self.position_history.append(current_state)
            # Keep only the last max_history states
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)

    def undo_position(self):
        """Revert to previous position state"""
        if self.position_history:
            # Get the previous state
            previous_state = self.position_history.pop()
            # Update current positions
            self.pos = previous_state
            # Update visualization
            self.draw_network()
            self.canvas.draw()

    def create_control_panel(self):
        """Create the control panel with all settings"""
        # Create panel and layout
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File controls
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout()
        
        load_button = QPushButton("Load Excel File")
        load_button.clicked.connect(self.load_file)
        file_layout.addWidget(load_button)
        
        self.status_label = QLabel("No file selected")
        self.status_label.setWordWrap(True)
        file_layout.addWidget(self.status_label)
        
        file_group.setLayout(file_layout)
        
        # Search section
        search_group = QGroupBox("Search")
        search_layout = QVBoxLayout()
        
        # Add search input and button
        search_input_layout = QHBoxLayout()
        self.search_input = QComboBox()
        self.search_input.setEditable(True)
        self.search_input.setInsertPolicy(QComboBox.InsertPolicy.InsertAtTop)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_node)
        search_input_layout.addWidget(self.search_input)
        search_input_layout.addWidget(self.search_button)
        
        # Clear search button
        self.clear_search_button = QPushButton("Clear Search")
        self.clear_search_button.clicked.connect(self.clear_search)
        
        search_layout.addLayout(search_input_layout)
        search_layout.addWidget(self.clear_search_button)
        search_group.setLayout(search_layout)
        
        # Layout controls with animation, disperse, and undo buttons
        layout_group = QGroupBox("Layout Controls")
        layout_controls = QHBoxLayout()
        
        # Animation control
        self.start_button = QPushButton("Start Animation")
        self.start_button.clicked.connect(self.toggle_animation)
        layout_controls.addWidget(self.start_button)
        
        # Disperse button
        disperse_button = QPushButton("Disperse Nodes")
        disperse_button.clicked.connect(self.disperse_nodes)
        layout_controls.addWidget(disperse_button)
        
        # Undo button
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_position)
        self.undo_button.setToolTip("Undo last position change")
        layout_controls.addWidget(self.undo_button)
        
        layout_group.setLayout(layout_controls)
        
        # Date range controls
        date_group = QGroupBox("Date Range")
        date_layout = QVBoxLayout()
        
        date_layout.addWidget(QLabel("Start Date:"))
        self.start_date = QDateTimeEdit()
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        self.start_date.setCalendarPopup(True)
        self.start_date.dateTimeChanged.connect(self.process_data)
        date_layout.addWidget(self.start_date)
        
        date_layout.addWidget(QLabel("End Date:"))
        self.end_date = QDateTimeEdit()
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        self.end_date.setCalendarPopup(True)
        self.end_date.dateTimeChanged.connect(self.process_data)
        date_layout.addWidget(self.end_date)
        
        date_group.setLayout(date_layout)
        
        # Amount filter controls
        amount_group = QGroupBox("Amount Filters")
        amount_layout = QVBoxLayout()
        
        # Individual transaction filter
        amount_layout.addWidget(QLabel("Minimum Individual Transaction:"))
        self.min_transaction_spin = QSpinBox()
        self.min_transaction_spin.setRange(0, 1000000)
        self.min_transaction_spin.setSingleStep(1000)
        self.min_transaction_spin.setValue(0)
        self.min_transaction_spin.valueChanged.connect(self.process_data)
        amount_layout.addWidget(self.min_transaction_spin)
        
        # Aggregated amount filter
        amount_layout.addWidget(QLabel("Minimum Aggregated Amount:"))
        self.min_aggregated_spin = QSpinBox()
        self.min_aggregated_spin.setRange(0, 10000000)
        self.min_aggregated_spin.setSingleStep(5000)
        self.min_aggregated_spin.setValue(0)
        self.min_aggregated_spin.valueChanged.connect(self.process_data)
        amount_layout.addWidget(self.min_aggregated_spin)
        
        amount_group.setLayout(amount_layout)
        
        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_controls = QVBoxLayout()
        
        # Node size weighting
        viz_controls.addWidget(QLabel("Node Size Weighting:"))
        weight_layout = QHBoxLayout()
        
        self.weight_type_combo = QComboBox()
        self.weight_type_combo.addItems(["Total Volume", "Outgoing Volume", "Incoming Volume"])
        self.weight_type_combo.currentTextChanged.connect(self.update_visualization)
        weight_layout.addWidget(self.weight_type_combo)
        
        self.weight_factor_spin = QSpinBox()
        self.weight_factor_spin.setRange(1, 1000)
        self.weight_factor_spin.setValue(100)
        self.weight_factor_spin.setSingleStep(10)
        self.weight_factor_spin.valueChanged.connect(self.update_visualization)
        weight_layout.addWidget(self.weight_factor_spin)
        
        viz_controls.addLayout(weight_layout)
        
        # Edge width control
        viz_controls.addWidget(QLabel("Edge Width:"))
        self.edge_width_spin = QSpinBox()
        self.edge_width_spin.setRange(1, 100)
        self.edge_width_spin.setValue(2)
        self.edge_width_spin.valueChanged.connect(self.update_visualization)
        viz_controls.addWidget(self.edge_width_spin)
        
        # Arrow size control
        viz_controls.addWidget(QLabel("Arrow Size:"))
        self.arrow_size_spin = QSpinBox()
        self.arrow_size_spin.setRange(1, 100)
        self.arrow_size_spin.setValue(10)
        self.arrow_size_spin.valueChanged.connect(self.update_visualization)
        viz_controls.addWidget(self.arrow_size_spin)
        
        # Analytics selector
        viz_controls.addWidget(QLabel("Analytics:"))
        self.analytics_combo = QComboBox()
        self.analytics_combo.addItems(["None", "Degree Centrality", "Betweenness Centrality", "Communities"])
        self.analytics_combo.currentTextChanged.connect(self.update_visualization)
        viz_controls.addWidget(self.analytics_combo)
        
        viz_group.setLayout(viz_controls)
        
        # Add all groups to main layout in the desired order
        layout.addWidget(file_group)
        layout.addWidget(search_group)
        layout.addWidget(layout_group)
        layout.addWidget(date_group)
        layout.addWidget(amount_group)
        layout.addWidget(viz_group)
        layout.addStretch()
        
        return panel
        
    def disperse_nodes(self):
        """Automatically disperse nodes to avoid overlapping"""
        if not self.G:
            return
            
        # Reset positions with random initial layout
        self.pos = nx.spring_layout(self.G)
        self.update_visualization()
        
    def search_node(self):
        """Search for a node by name and highlight it"""
        search_text = self.search_input.currentText().strip()
        if not search_text or not self.G:
            return
        
        # Find exact or partial matches
        matches = [node for node in self.G.nodes() if search_text.lower() in node.lower()]
        
        if matches:
            # Update search history
            if search_text not in [self.search_input.itemText(i) for i in range(self.search_input.count())]:
                self.search_input.addItem(search_text)
            
            # Select the first matching node
            self.searched_node = matches[0]
            self.selected_nodes = {matches[0]}
            self.update_visualization()
            
            # Center view on the found node
            self.ax.set_xlim(self.pos[matches[0]][0] - 0.5, self.pos[matches[0]][0] + 0.5)
            self.ax.set_ylim(self.pos[matches[0]][1] - 0.5, self.pos[matches[0]][1] + 0.5)
            self.canvas.draw()
        else:
            QMessageBox.warning(self, "Search Result", "No matching node found.")

    def clear_search(self):
        """Clear search highlight"""
        self.searched_node = None
        self.selected_nodes.clear()
        self.update_visualization()

def main():
    app = QApplication(sys.argv)
    window = GTTFlowNet()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
