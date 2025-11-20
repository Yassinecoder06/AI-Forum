# Quick testing script for new data
from test_new_data import DrivingStyleTester
import pandas as pd

def quick_test(csv_file_path, traffic_state="normal_traffic"):
    """Quick test function for new data"""
    
    print("🚗 QUICK DRIVING STYLE TEST")
    print("="*30)
    
    # Initialize tester
    tester = DrivingStyleTester(model_path='./')
    
    try:
        # Load and test data
        new_df = pd.read_csv(csv_file_path)
        processed_df = tester.preprocess_new_data(new_df, traffic_state)
        features_df = tester.extract_features_from_new_data(processed_df)
        predictions, probabilities = tester.predict_driving_styles(features_df)
        results_df = tester.analyze_predictions(features_df, predictions, probabilities)
        
        # Quick summary
        style_counts = results_df['predicted_style_label'].value_counts()
        dominant_style = style_counts.index[0]
        confidence = results_df['confidence'].mean()
        
        print(f"\n📊 QUICK RESULTS:")
        print(f"   • Total windows: {len(results_df)}")
        print(f"   • Dominant style: {dominant_style}")
        print(f"   • Average confidence: {confidence:.3f}")
        print(f"   • Style distribution:")
        for style, count in style_counts.items():
            print(f"     - {style}: {count} ({count/len(results_df):.1%})")
            
        return results_df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Usage example
if __name__ == "__main__":
    # Replace with your test file path
    test_file = "./Dataset/OBD-II-Dataset/2018-02-23_Seat_Leon_S_RT_Normal.csv"
    results = quick_test(test_file)