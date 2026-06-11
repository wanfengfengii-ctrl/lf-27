import sys
import pandas as pd
import numpy as np

print("Testing all fixes...\n")

print("1. Testing batches parameter for detect_abnormal_growth...")
from data_processor import detect_abnormal_growth, detect_missing_inspections, revalidate_dataframe

df = pd.read_csv('sample_data.csv')
from data_processor import standardize_columns, validate_and_clean_data
std_df, _ = standardize_columns(df)
valid_df, errors, warnings, _ = validate_and_clean_data(std_df)
print(f"   Total valid records: {len(valid_df)}")

all_batches = sorted(valid_df['巡检批次'].unique())
print(f"   All batches: {all_batches}")

abnormal_all = detect_abnormal_growth(valid_df, None, None)
abnormal_q4 = detect_abnormal_growth(valid_df, None, ['2024-Q4'])
print(f"   Abnormal (all batches): {len(abnormal_all)}")
print(f"   Abnormal (only 2024-Q4, should be 0): {len(abnormal_q4)}")
assert len(abnormal_q4) == 0, "Q4 only should have no abnormal growth (needs 2 batches)"
print("   ✓ PASS: detect_abnormal_growth respects batches filter\n")

print("2. Testing batches parameter for detect_missing_inspections...")
missing_all = detect_missing_inspections(valid_df, 'A区', None)
print(f"   Missing in A区 (all batches): {len(missing_all)}")
print(f"   Missing batches: {sorted(missing_all['缺失批次'].unique()) if not missing_all.empty else 'none'}")

missing_q4 = detect_missing_inspections(valid_df, 'A区', ['2024-Q4'])
print(f"   Missing in A区 (only 2024-Q4): {len(missing_q4)}")
if not missing_q4.empty:
    print(f"   Missing batches (filtered): {sorted(missing_q4['缺失批次'].unique())}")
    assert all(b == '2024-Q4' for b in missing_q4['缺失批次'].unique()), "Should only contain Q4"
print("   ✓ PASS: detect_missing_inspections respects batches filter\n")

print("3. Testing revalidate_dataframe...")
test_data = {
    '管段编号': ['P001', 'P002', 'P003'],
    '巡检批次': ['2024-Q1', '2024-Q1', '2024-Q1'],
    '片区': ['A区', 'A区', 'A区'],
    '检查时间': ['2024-03-15', '2024-03-16', '2024-03-17'],
    '淤积深度': [100, -50, 80],
    '管径': [500, 500, 500],
    '备注': ['OK', 'negative test', 'OK']
}
test_df = pd.DataFrame(test_data)
revalid_df, revalid_errors = revalidate_dataframe(test_df)
print(f"   Valid records: {len(revalid_df)} (expected 2)")
print(f"   Errors: {len(revalid_errors)} (expected 1)")
assert len(revalid_df) == 2, f"Expected 2 valid, got {len(revalid_df)}"
assert len(revalid_errors) == 1, f"Expected 1 error, got {len(revalid_errors)}"
print("   ✓ PASS: revalidate_dataframe works correctly\n")

print("4. Testing missing point placement in chart...")
from visualizations import create_pipe_history_chart
fig = create_pipe_history_chart(valid_df, 'P005', 'A区')
missing_traces = [t for t in fig.data if t.name == '缺失巡检']
print(f"   Missing traces found: {len(missing_traces)} (expected 2: one per subplot)")
if missing_traces:
    has_y_values = all(t.y is not None and all(v is not None for v in t.y) for t in missing_traces)
    print(f"   Missing trace has valid y values: {has_y_values}")
    assert has_y_values, "Missing traces should have valid y values (not None)"
print("   ✓ PASS: Missing traces have visible markers\n")

print("=" * 50)
print("✅ ALL TESTS PASSED!")
