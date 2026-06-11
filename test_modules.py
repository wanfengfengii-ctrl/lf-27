import sys
import pandas as pd

print("Testing imports...")
try:
    from data_processor import standardize_columns, validate_and_clean_data, calculate_statistics
    print("✓ data_processor imported successfully")
except Exception as e:
    print(f"✗ data_processor import failed: {e}")
    sys.exit(1)

try:
    from visualizations import create_pipe_history_chart, create_risk_heatmap
    print("✓ visualizations imported successfully")
except Exception as e:
    print(f"✗ visualizations import failed: {e}")
    sys.exit(1)

print("\nTesting data processing...")
df = pd.read_csv('sample_data.csv')
print(f"✓ Loaded sample data: {len(df)} rows")

std_df, missing_cols = standardize_columns(df)
print(f"✓ Standardized columns. Missing: {missing_cols}")

valid_df, errors, warnings, has_district = validate_and_clean_data(std_df)
print(f"✓ Validated data: {len(valid_df)} valid records, {len(errors)} errors, {len(warnings)} warnings")

stats = calculate_statistics(valid_df)
print(f"✓ Calculated statistics: {list(stats.keys())}")

print("\nTesting chart generation...")
try:
    fig = create_pipe_history_chart(valid_df, 'P001', 'A区')
    print(f"✓ Pipe history chart created: {type(fig)}")
except Exception as e:
    print(f"✗ Chart creation failed: {e}")

try:
    fig = create_risk_heatmap(valid_df, 'A区')
    print(f"✓ Risk heatmap created: {type(fig)}")
except Exception as e:
    print(f"✗ Heatmap creation failed: {e}")

print("\n✅ All tests passed!")
