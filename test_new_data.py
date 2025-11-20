# ===========================
# TEST NEW DATA MODULE
# ===========================
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import the required functions from your training code
from IA_Forum import (
    calculate_derivatives, detect_events, enhance_features,
    create_sliding_windows, create_feature_dataset, label_driving_styles,
    EnhancedDrivingNet, load_scaler_from_json, load_label_encoder_from_json,
    time_to_seconds
)

class DrivingStyleTester:
    """Comprehensive testing module for new driving data"""
    
    def __init__(self, model_path=None, scaler_path='feature_scaler.json', 
                 encoder_path='label_encoder.json'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        
        # Create results directories: one for all outputs and one for plots
        self.results_dir = "results"
        self.output_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load preprocessing objects
        self.load_preprocessing_objects(scaler_path, encoder_path)
        
        # Load model
        if model_path:
            self.load_model(model_path)
    
    def load_preprocessing_objects(self, scaler_path, encoder_path):
        """Load preprocessing objects"""
        try:
            self.scaler = load_scaler_from_json(scaler_path)
            self.label_encoder = load_label_encoder_from_json(encoder_path)
            print("✓ Preprocessing objects loaded successfully")
        except Exception as e:
            print(f"❌ Error loading preprocessing objects: {e}")
    
    def load_model(self, model_path=None):
        """Load the trained model with multiple fallback options"""
        candidates = []
        if model_path:
            candidates.append(model_path)
        candidates += [
            'safe_complete_model.pth',
            'complete_driving_model.pth',
            'driving_model_state_dict.pth'
        ]
    
        last_exc = None
        for path in candidates:
            try:
                checkpoint = torch.load(path, map_location=self.device)
                # If it's a state_dict style, detect and load accordingly
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    input_dim = checkpoint.get('input_dim')
                    self.feature_columns = checkpoint.get('feature_columns')
                    self.model = EnhancedDrivingNet(input_dim=input_dim)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✓ Model loaded from state dict: {path}")
                else:
                    self.model = checkpoint
                    print(f"✓ Model loaded successfully: {path}")
                self.model.to(self.device)
                self.model.eval()
                return
            except Exception as e:
                last_exc = e
                print(f"Attempt to load {path} failed: {e}")
    
        raise RuntimeError(f"Failed to load model from candidates {candidates}") from last_exc

    
    def preprocess_new_data(self, df, traffic_state="unknown"):
        """Preprocess new data using the same pipeline as training"""
        print("Step 1: Preprocessing data...")
        
        # Add traffic state if not present
        if 'traffic_state' not in df.columns:
            df['traffic_state'] = traffic_state
        
        # Apply preprocessing pipeline
        df = calculate_derivatives(df)
        df = detect_events(df)
        df = enhance_features(df)
        
        return df
    
    def extract_features_from_new_data(self, df, window_size=15.0, step_size=7.5):
        """Extract features from new data using sliding windows"""
        print("Step 2: Creating sliding windows and extracting features...")
        
        # Create sliding windows
        windows = create_sliding_windows(df, window_size, step_size)
        
        # Extract features
        df_features = create_feature_dataset(windows)
        
        # Label driving styles (this will use the same logic as training)
        df_features = label_driving_styles(df_features)
        
        return df_features
    
    def predict_driving_styles(self, df_features):
        """Make predictions on new features"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Step 3: Making predictions...")
        
        # Define feature columns (use saved ones or default)
        if self.feature_columns is None:
            self.feature_columns = [
                'rpm_mean', 'rpm_std', 'rpm_max', 
                'speed_mean', 'speed_std', 'speed_max',
                'throttle_mean', 'throttle_std',
                'maf_mean', 'maf_std',
                'accel_mean', 'accel_std', 'accel_max', 'accel_min',
                'pedal_rate_mean', 'pedal_rate_std',
                'speed_entropy', 'rpm_entropy',
                'accel_events', 'brake_events', 'total_events',
                'engine_load_mean', 'engine_load_std',
                'moving_efficiency', 'moving_ratio',
                'traffic_state_num', 'eco_score'
            ]
        
        # Prepare features
        X = df_features[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(X_scaled).to(self.device))
            probabilities = torch.softmax(predictions, dim=1)
            predicted_classes = torch.argmax(predictions, dim=1)
        
        # Convert to numpy
        probabilities_np = probabilities.cpu().numpy()
        predicted_classes_np = predicted_classes.cpu().numpy()
        
        return predicted_classes_np, probabilities_np
    
    def analyze_predictions(self, df_features, predictions, probabilities):
        """Comprehensive analysis of predictions"""
        print("Step 4: Analyzing predictions...")
        
        # Add predictions to features dataframe
        df_results = df_features.copy()
        df_results['predicted_style'] = predictions
        df_results['predicted_style_label'] = [self.label_encoder.inverse_transform([p])[0] for p in predictions]
        
        # Add probabilities
        style_mapping = {0: "calm", 1: "normal", 2: "aggressive"}
        for i, style in enumerate(['calm', 'normal', 'aggressive']):
            df_results[f'prob_{style}'] = probabilities[:, i]
        
        # Add confidence (max probability)
        df_results['confidence'] = np.max(probabilities, axis=1)
        
        return df_results
    
    def generate_comprehensive_report(self, df_results, original_df):
        """Generate detailed report and visualizations"""
        print("Step 5: Generating comprehensive report...")
        
        # Basic statistics
        total_windows = len(df_results)
        style_distribution = df_results['predicted_style_label'].value_counts()
        avg_confidence = df_results['confidence'].mean()
        
        # Driving behavior analysis
        avg_speed = df_results['speed_mean'].mean()
        avg_rpm = df_results['rpm_mean'].mean()
        total_events = df_results['total_events'].sum()
        aggressive_ratio = len(df_results[df_results['predicted_style_label'] == 'aggressive']) / total_windows
        
        print("\n" + "="*60)
        print("COMPREHENSIVE DRIVING STYLE ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   • Total time windows analyzed: {total_windows}")
        print(f"   • Average prediction confidence: {avg_confidence:.3f}")
        print(f"   • Data duration: {original_df['Time_sec'].max() - original_df['Time_sec'].min():.1f} seconds")
        
        print(f"\n🚗 DRIVING BEHAVIOR METRICS:")
        print(f"   • Average speed: {avg_speed:.1f} km/h")
        print(f"   • Average RPM: {avg_rpm:.0f}")
        print(f"   • Total sudden events: {total_events}")
        print(f"   • Aggressive driving ratio: {aggressive_ratio:.1%}")
        
        print(f"\n🎯 DRIVING STYLE DISTRIBUTION:")
        for style, count in style_distribution.items():
            percentage = (count / total_windows) * 100
            print(f"   • {style.capitalize()}: {count} windows ({percentage:.1f}%)")
        
        # Dominant style
        dominant_style = style_distribution.index[0]
        print(f"\n🏆 DOMINANT DRIVING STYLE: {dominant_style.upper()}")
        
        # Safety assessment
        if aggressive_ratio > 0.3:
            safety_level = "⚠️  NEEDS ATTENTION - High aggressive driving detected"
        elif aggressive_ratio > 0.1:
            safety_level = "ℹ️  MODERATE - Some aggressive patterns observed"
        else:
            safety_level = "✅ GOOD - Mostly calm and normal driving"
        
        print(f"\n🛡️  SAFETY ASSESSMENT: {safety_level}")
        
        return {
            'total_windows': total_windows,
            'style_distribution': style_distribution.to_dict(),
            'avg_confidence': avg_confidence,
            'avg_speed': avg_speed,
            'avg_rpm': avg_rpm,
            'total_events': total_events,
            'aggressive_ratio': aggressive_ratio,
            'dominant_style': dominant_style,
            'safety_level': safety_level
        }
    
    def create_visualizations(self, df_results, report_stats):
        """Create comprehensive visualizations and save them directly"""
        print("\nStep 6: Creating and saving visualizations...")
        
        # Create main comprehensive plot
        self._create_main_plot(df_results, report_stats)
        
        # Create detailed analysis plot
        self._create_detailed_plot(df_results)
        
        # Create individual plots for better clarity
        self._create_individual_plots(df_results, report_stats)
        
        print(f"✓ All visualizations saved to: {self.output_dir}/")
    
    def _create_main_plot(self, df_results, report_stats):
        """Create the main comprehensive visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Driving Style Analysis - Comprehensive Report', fontsize=18, fontweight='bold')
        
        # 1. Driving Style Distribution (Pie chart)
        style_counts = df_results['predicted_style_label'].value_counts()
        colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red
        axes[0, 0].pie(style_counts.values, labels=style_counts.index, autopct='%1.1f%%', 
                      colors=colors, startangle=90, textprops={'fontsize': 12})
        axes[0, 0].set_title('Driving Style Distribution', fontweight='bold', fontsize=14)
        
        # 2. Confidence Distribution
        axes[0, 1].hist(df_results['confidence'], bins=20, alpha=0.7, color='#9b59b6', edgecolor='black')
        axes[0, 1].axvline(df_results['confidence'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df_results["confidence"].mean():.3f}')
        axes[0, 1].set_xlabel('Prediction Confidence', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Prediction Confidence Distribution', fontweight='bold', fontsize=14)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=10)
        
        # 3. Speed vs Driving Style
        speed_data = [df_results[df_results['predicted_style_label'] == style]['speed_mean'] 
                     for style in ['calm', 'normal', 'aggressive']]
        axes[0, 2].boxplot(speed_data, labels=['Calm', 'Normal', 'Aggressive'])
        axes[0, 2].set_ylabel('Speed (km/h)', fontsize=12)
        axes[0, 2].set_title('Speed Distribution by Driving Style', fontweight='bold', fontsize=14)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].tick_params(axis='both', which='major', labelsize=10)
        
        # 4. Event Counts by Style
        event_data = df_results.groupby('predicted_style_label')['total_events'].mean()
        axes[1, 0].bar(event_data.index, event_data.values, 
                      color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[1, 0].set_ylabel('Average Event Count', fontsize=12)
        axes[1, 0].set_title('Average Sudden Events by Driving Style', fontweight='bold', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
        
        # 5. Timeline of Driving Style Changes
        if len(df_results) > 1:
            axes[1, 1].plot(range(len(df_results)), df_results['predicted_style'], 
                           marker='o', linestyle='-', alpha=0.7, markersize=4)
            axes[1, 1].set_xlabel('Time Window Index', fontsize=12)
            axes[1, 1].set_ylabel('Driving Style (0=Calm, 1=Normal, 2=Aggressive)', fontsize=12)
            axes[1, 1].set_title('Driving Style Timeline', fontweight='bold', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='both', which='major', labelsize=10)
        
        # 6. RPM vs Acceleration by Style
        scatter = axes[1, 2].scatter(df_results['rpm_mean'], df_results['accel_std'], 
                                    c=df_results['predicted_style'], cmap='viridis', alpha=0.6, s=50)
        axes[1, 2].set_xlabel('Average RPM', fontsize=12)
        axes[1, 2].set_ylabel('Acceleration STD', fontsize=12)
        axes[1, 2].set_title('RPM vs Acceleration Variability', fontweight='bold', fontsize=14)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].tick_params(axis='both', which='major', labelsize=10)
        plt.colorbar(scatter, ax=axes[1, 2], label='Driving Style')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_analysis.pdf'), bbox_inches='tight')
        plt.close()
    
    def _create_detailed_plot(self, df_results):
        """Create additional detailed analysis plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Driving Style Analysis', fontsize=16, fontweight='bold')
        
        # 1. Probability distributions for each style
        styles = ['calm', 'normal', 'aggressive']
        for i, style in enumerate(styles):
            axes[0, 0].hist(df_results[f'prob_{style}'], bins=20, alpha=0.6, 
                           label=style.capitalize(), density=True)
        axes[0, 0].set_xlabel('Probability', fontsize=12)
        axes[0, 0].set_ylabel('Density', fontsize=12)
        axes[0, 0].set_title('Probability Distributions by Driving Style', fontweight='bold', fontsize=14)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=10)
        
        # 2. Confidence vs Style
        confidence_by_style = [df_results[df_results['predicted_style_label'] == style]['confidence'] 
                              for style in styles]
        axes[0, 1].boxplot(confidence_by_style, labels=[s.capitalize() for s in styles])
        axes[0, 1].set_ylabel('Confidence', fontsize=12)
        axes[0, 1].set_title('Prediction Confidence by Driving Style', fontweight='bold', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=10)
        
        # 3. Feature importance correlation
        correlation_features = ['speed_mean', 'rpm_mean', 'accel_std', 'total_events', 'moving_efficiency']
        corr_matrix = df_results[correlation_features + ['predicted_style']].corr()
        
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(correlation_features) + 1))
        axes[1, 0].set_yticks(range(len(correlation_features) + 1))
        axes[1, 0].set_xticklabels(correlation_features + ['style'], rotation=45, fontsize=10)
        axes[1, 0].set_yticklabels(correlation_features + ['style'], fontsize=10)
        axes[1, 0].set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
        
        # Add correlation values as text
        for i in range(len(correlation_features) + 1):
            for j in range(len(correlation_features) + 1):
                axes[1, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', fontsize=9, 
                               color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Style transitions (if enough data)
        if len(df_results) > 10:
            style_changes = (df_results['predicted_style'].diff().abs() > 0).sum()
            axes[1, 1].bar(['Style Changes'], [style_changes], color='orange', alpha=0.7, edgecolor='black')
            axes[1, 1].set_ylabel('Number of Changes', fontsize=12)
            axes[1, 1].set_title('Driving Style Transitions', fontweight='bold', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'detailed_analysis.pdf'), bbox_inches='tight')
        plt.close()
    
    def _create_individual_plots(self, df_results, report_stats):
        """Create individual plots for better clarity"""
        
        # 1. Style distribution pie chart
        plt.figure(figsize=(10, 8))
        style_counts = df_results['predicted_style_label'].value_counts()
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        plt.pie(style_counts.values, labels=style_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90, textprops={'fontsize': 14})
        plt.title('Driving Style Distribution', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.output_dir, 'style_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df_results['confidence'], bins=20, alpha=0.7, color='#9b59b6', edgecolor='black')
        plt.axvline(df_results['confidence'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df_results["confidence"].mean():.3f}')
        plt.xlabel('Prediction Confidence', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Prediction Confidence Distribution', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Speed by driving style
        plt.figure(figsize=(10, 6))
        speed_data = [df_results[df_results['predicted_style_label'] == style]['speed_mean'] 
                     for style in ['calm', 'normal', 'aggressive']]
        plt.boxplot(speed_data, labels=['Calm', 'Normal', 'Aggressive'])
        plt.ylabel('Speed (km/h)', fontsize=12)
        plt.title('Speed Distribution by Driving Style', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'speed_by_style.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Events by driving style
        plt.figure(figsize=(10, 6))
        event_data = df_results.groupby('predicted_style_label')['total_events'].mean()
        plt.bar(event_data.index, event_data.values, 
                color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
        plt.ylabel('Average Event Count', fontsize=12)
        plt.title('Average Sudden Events by Driving Style', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'events_by_style.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Driving style timeline
        if len(df_results) > 1:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(df_results)), df_results['predicted_style'], 
                    marker='o', linestyle='-', alpha=0.7, markersize=4, linewidth=2)
            plt.xlabel('Time Window Index', fontsize=12)
            plt.ylabel('Driving Style (0=Calm, 1=Normal, 2=Aggressive)', fontsize=12)
            plt.title('Driving Style Timeline', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'style_timeline.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_detailed_report(self, df_results, report_stats, output_path=None):
        """Save detailed report to CSV and a JSON summary inside the `results/` directory."""
        # Default paths inside results/
        if output_path is None:
            output_path = os.path.join(self.results_dir, 'driving_analysis_report.csv')
        summary_path = os.path.join(self.results_dir, 'driving_analysis_summary.json')

        print(f"\nStep 7: Saving detailed report to {output_path}...")

        # Save the comprehensive results
        df_results.to_csv(output_path, index=False)

        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            else:
                return obj

        # Convert report_stats to JSON-serializable format
        serializable_stats = {}
        for key, value in report_stats.items():
            serializable_stats[key] = convert_numpy_types(value)

        # Handle nested structures
        if 'style_distribution' in serializable_stats:
            if hasattr(serializable_stats['style_distribution'], 'items'):
                serializable_stats['style_distribution'] = {
                    str(k): convert_numpy_types(v) 
                    for k, v in serializable_stats['style_distribution'].items()
                }

        summary_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'report_summary': serializable_stats,
            'model_confidence_threshold': 0.7,
            'safety_recommendations': self.generate_safety_recommendations(report_stats),
            'plot_directory': os.path.relpath(self.output_dir, start=self.results_dir)
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=convert_numpy_types)

        print(f"✓ Detailed report saved to: {output_path}")
        print(f"✓ Summary statistics saved to: {summary_path}")
        print(f"✓ Visualizations saved to: {self.output_dir}/")
    
    def generate_safety_recommendations(self, report_stats):
        """Generate safety recommendations based on analysis"""
        recommendations = []
        
        if report_stats['aggressive_ratio'] > 0.3:
            recommendations.extend([
                "High level of aggressive driving detected",
                "Recommend: Defensive driving training",
                "Suggest: Regular vehicle maintenance check",
                "Advise: Monitor driving patterns regularly"
            ])
        elif report_stats['aggressive_ratio'] > 0.1:
            recommendations.extend([
                "Moderate aggressive driving patterns observed",
                "Recommend: Conscious speed management",
                "Suggest: Smooth acceleration and braking practice"
            ])
        else:
            recommendations.extend([
                "Good driving patterns maintained",
                "Recommend: Continue current driving habits",
                "Suggest: Periodic self-assessment"
            ])
        
        if report_stats['avg_speed'] > 80:
            recommendations.append("Consider reducing average speed for better fuel efficiency")
        
        if report_stats['total_events'] > 10:
            recommendations.append("High number of sudden events - focus on smoother driving")
        
        return recommendations
    
    def test_single_driving_session(self, csv_file_path, traffic_state="normal_traffic", 
                                  window_size=15.0, step_size=7.5):
        """Complete testing pipeline for a single driving session"""
        print("🚗 STARTING COMPREHENSIVE DRIVING STYLE ANALYSIS")
        print("="*50)
        
        try:
            # 1. Load new data
            print(f"📁 Loading data from: {csv_file_path}")
            new_df = pd.read_csv(csv_file_path)
            print(f"   • Loaded {len(new_df)} data points")
            print(f"   • Columns: {list(new_df.columns)}")
            
            # 2. Preprocess data
            processed_df = self.preprocess_new_data(new_df, traffic_state)
            
            # 3. Extract features
            features_df = self.extract_features_from_new_data(processed_df, window_size, step_size)
            print(f"   • Created {len(features_df)} time windows")
            
            # 4. Make predictions
            predictions, probabilities = self.predict_driving_styles(features_df)
            
            # 5. Analyze results
            results_df = self.analyze_predictions(features_df, predictions, probabilities)
            
            # 6. Generate report
            report_stats = self.generate_comprehensive_report(results_df, processed_df)
            
            # 7. Create visualizations
            self.create_visualizations(results_df, report_stats)
            
            # 8. Save report
            self.save_detailed_report(results_df, report_stats)
            
            print("\n" + "="*50)
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*50)
            
            return results_df, report_stats
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise

# Example usage and test function
def run_comprehensive_test():
    """Run a comprehensive test on new data"""
    
    # Initialize tester
    tester = DrivingStyleTester(model_path="./")
    
    # Test with a new CSV file (replace with your file path)
    test_file_path = "./Dataset/OBD-II-Dataset/2017-07-14_Seat_Leon_KA_RT_Normal.csv"  # Replace with your test file
    
    try:
        # Run comprehensive analysis
        results, stats = tester.test_single_driving_session(
            csv_file_path=test_file_path,
            traffic_state="traffic_free",  # or "traffic_jam", "traffic_free"
            window_size=15.0,
            step_size=7.5
        )
        
        return results, stats
     
    except FileNotFoundError:
        print(f"❌ Test file not found: {test_file_path}")
        print("💡 Please update the test_file_path with your actual test data file")
        
        # Create sample test data for demonstration
        print("\n🎯 Creating sample test data for demonstration...")
        create_sample_test_data()
        
        # Test with sample data
        results, stats = tester.test_single_driving_session(
            csv_file_path="sample_test_data.csv",
            traffic_state="normal_traffic"
        )
        
        return results, stats

def create_sample_test_data():
    """Create sample test data for demonstration"""
    # Create realistic sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'Time': [f"{i//3600:02d}:{(i%3600)//60:02d}:{i%60:02d}" for i in range(n_samples)],
        'Vehicle Speed Sensor [km/h]': np.random.normal(60, 20, n_samples).clip(0, 120),
        'Engine RPM [RPM]': np.random.normal(2500, 800, n_samples).clip(800, 6000),
        'Accelerator Pedal Position D [%]': np.random.normal(30, 15, n_samples).clip(0, 100),
        'Absolute Throttle Position [%]': np.random.normal(25, 12, n_samples).clip(0, 100),
        'Air Flow Rate from Mass Flow Sensor [g/s]': np.random.normal(25, 10, n_samples).clip(0, 50)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_test_data.csv', index=False)
    print("✓ Sample test data created: sample_test_data.csv")

if __name__ == "__main__":
    # Run the comprehensive test
    results, stats = run_comprehensive_test()