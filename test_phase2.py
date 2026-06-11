import pandas as pd
from data_processor import (
    standardize_columns, validate_and_clean_data, get_districts, get_batches,
    get_pipe_ids, calculate_statistics, detect_abnormal_growth, get_high_risk_segments,
    detect_missing_inspections, revalidate_dataframe, validate_batch_data,
    validate_cross_district, evaluate_dredging_effect, batch_evaluate_dredging,
    calculate_dredging_priority, generate_inspection_quality_report,
    get_district_summary, get_effective_rules, DEFAULT_RISK_RULES
)
from visualizations import (
    create_pipe_history_chart, create_pipes_comparison_chart, create_risk_heatmap,
    create_risk_distribution_chart, create_sediment_trend_chart,
    create_dredging_effect_chart, create_district_comparison_chart,
    create_priority_dashboard_chart
)

print('=== All imports successful ===')

df = pd.read_csv('sample_data.csv')
std_df, missing = standardize_columns(df)
print(f'Standardized: {len(std_df)} rows, missing cols: {missing}')

valid_df, errors, warnings, has_district = validate_and_clean_data(std_df)
print(f'Valid: {len(valid_df)} rows, Errors: {len(errors)}, Warnings: {len(warnings)}, Has district: {has_district}')

rules = DEFAULT_RISK_RULES.copy()
print(f'Default rules: {rules}')

repaired_df, batch_errors, repair_actions = validate_batch_data(valid_df, '2024-Q2', rules=rules)
print(f'Batch validate: {len(repaired_df)} repaired, {len(batch_errors)} errors, {len(repair_actions)} repairs')

is_valid, msg = validate_cross_district(['P001', 'P006'], valid_df)
print(f'Cross-district P001+P006: valid={is_valid}')

is_valid2, msg2 = validate_cross_district(['P001', 'P002'], valid_df)
print(f'Cross-district P001+P002: valid={is_valid2}')

result = evaluate_dredging_effect(valid_df, 'P001', '2024-Q4', '2025-Q1', rules=rules)
print(f'Dredging effect P001: {result["效果评级"] if result else "None"}')

dredge_results = batch_evaluate_dredging(valid_df, rules=rules)
print(f'Batch dredge eval: {len(dredge_results)} results')

priority_df = calculate_dredging_priority(valid_df, rules=rules)
print(f'Priority: {len(priority_df)} pipes')
print(priority_df[['管段编号', '清淤优先级', '风险评分']].to_string())

report = generate_inspection_quality_report(valid_df, rules=rules)
print(f'Quality report keys: {list(report.keys())}')

dist_summary = get_district_summary(valid_df, rules=rules)
print(f'District summary: {len(dist_summary)} districts')

fig = create_dredging_effect_chart(dredge_results, rules=rules)
print(f'Dredging chart created: {type(fig).__name__}')

fig2 = create_district_comparison_chart(dist_summary, rules=rules)
print(f'District chart created: {type(fig2).__name__}')

fig3 = create_priority_dashboard_chart(priority_df, rules=rules)
print(f'Priority chart created: {type(fig3).__name__}')

print('\n=== ALL TESTS PASSED ===')
