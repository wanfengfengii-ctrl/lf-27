import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict


REQUIRED_COLUMNS = {
    '管段编号': ['管段编号', 'pipe_id', 'pipe_no', 'pipe_code', '管道编号'],
    '巡检批次': ['巡检批次', 'batch', 'inspection_batch', '批次'],
    '检查时间': ['检查时间', 'inspection_date', 'date', '时间', '巡检时间'],
    '淤积深度': ['淤积深度', 'sediment_depth', 'depth', '淤积厚度'],
    '管径': ['管径', 'diameter', 'pipe_diameter', '管道直径'],
    '备注': ['备注', 'remark', 'notes', 'comment', '说明']
}

DEFAULT_RISK_RULES = {
    'RISK_THRESHOLD_HIGH': 0.6,
    'RISK_THRESHOLD_MEDIUM': 0.3,
    'ABNORMAL_GROWTH_RATE': 0.2,
    'MISSING_INSPECTION_ALERT': True,
    'ABNORMAL_GROWTH_ALERT': True,
    'HIGH_RISK_ALERT': True,
    'DREDGING_EFFECT_THRESHOLD': 0.3,
}

RISK_THRESHOLD_HIGH = 0.6
RISK_THRESHOLD_MEDIUM = 0.3
ABNORMAL_GROWTH_RATE = 0.2


def get_effective_rules(custom_rules=None):
    rules = DEFAULT_RISK_RULES.copy()
    if custom_rules:
        for k, v in custom_rules.items():
            if v is not None and v != '':
                try:
                    rules[k] = type(rules.get(k, v))(v)
                except (ValueError, TypeError):
                    rules[k] = v
    return rules


def standardize_columns(df):
    column_mapping = {}
    for std_name, aliases in REQUIRED_COLUMNS.items():
        for col in df.columns:
            if col.strip() in aliases:
                column_mapping[col] = std_name
                break
    df = df.rename(columns=column_mapping)

    missing_cols = set(REQUIRED_COLUMNS.keys()) - set(df.columns)
    missing_cols.discard('备注')

    return df, missing_cols


def validate_and_clean_data(df, rules=None):
    rules = get_effective_rules(rules)
    errors = []
    warnings = []
    valid_rows = []

    if '片区' in df.columns:
        has_district = True
    else:
        has_district = False
        df['片区'] = '默认片区'
        warnings.append('数据中未包含"片区"字段，所有数据将归为默认片区')

    for idx, row in df.iterrows():
        row_num = idx + 2
        row_errors = []

        pipe_id = str(row.get('管段编号', '')).strip()
        if not pipe_id or pipe_id == 'nan':
            row_errors.append('管段编号为空')

        batch = str(row.get('巡检批次', '')).strip()
        if not batch or batch == 'nan':
            row_errors.append('巡检批次为空')

        try:
            check_date = pd.to_datetime(row.get('检查时间', ''))
            if pd.isna(check_date):
                row_errors.append('检查时间格式无效')
        except Exception:
            row_errors.append('检查时间格式无效')
            check_date = pd.NaT

        try:
            sediment_depth = float(row.get('淤积深度', np.nan))
            if pd.isna(sediment_depth):
                row_errors.append('淤积深度为空')
            elif sediment_depth < 0:
                row_errors.append(f'淤积深度为负数 ({sediment_depth})')
        except (ValueError, TypeError):
            row_errors.append('淤积深度不是有效数字')
            sediment_depth = np.nan

        try:
            diameter = float(row.get('管径', np.nan))
            if pd.isna(diameter):
                row_errors.append('管径为空')
            elif diameter <= 0:
                row_errors.append(f'管径必须大于0 ({diameter})')
        except (ValueError, TypeError):
            row_errors.append('管径不是有效数字')
            diameter = np.nan

        if not pd.isna(sediment_depth) and not pd.isna(diameter):
            if sediment_depth > diameter:
                row_errors.append(f'淤积深度 ({sediment_depth}) 超过管径 ({diameter})')

        if row_errors:
            errors.append({
                '行号': row_num,
                '管段编号': pipe_id if pipe_id else '未知',
                '巡检批次': batch if batch else '未知',
                '错误原因': '；'.join(row_errors),
                '原始数据': str(row.to_dict())
            })
        else:
            valid_row = {
                '管段编号': pipe_id,
                '巡检批次': batch,
                '检查时间': check_date,
                '淤积深度': sediment_depth,
                '管径': diameter,
                '备注': str(row.get('备注', '')),
                '片区': str(row.get('片区', '默认片区')),
                '淤积率': round(sediment_depth / diameter, 4) if diameter > 0 else 0
            }
            valid_rows.append(valid_row)

    valid_df = pd.DataFrame(valid_rows)

    if not valid_df.empty:
        dup_mask = valid_df.duplicated(subset=['管段编号', '巡检批次', '片区'], keep=False)
        duplicates = valid_df[dup_mask]
        if not duplicates.empty:
            for (pid, batch, district), group in duplicates.groupby(['管段编号', '巡检批次', '片区']):
                for _, dup_row in group.iloc[1:].iterrows():
                    original_idx = valid_df.index.get_loc(dup_row.name)
                    errors.append({
                        '行号': f'有效数据第{original_idx + 1}条',
                        '管段编号': pid,
                        '巡检批次': batch,
                        '错误原因': f'同一管段在同一巡检批次({batch})内重复记录',
                        '原始数据': str(dup_row.to_dict())
                    })
            valid_df = valid_df.drop_duplicates(subset=['管段编号', '巡检批次', '片区'], keep='first')
            warnings.append(f'发现 {len(duplicates) - len(duplicates.drop_duplicates(subset=["管段编号", "巡检批次", "片区"]))} 条重复记录，已保留第一条')

    return valid_df, errors, warnings, has_district


def get_districts(df):
    if df.empty:
        return []
    return sorted(df['片区'].unique().tolist())


def get_batches(df, district=None):
    if df.empty:
        return []
    filtered = df[df['片区'] == district] if district else df
    batches = filtered.groupby('巡检批次')['检查时间'].min().sort_values()
    return list(batches.index)


def get_pipe_ids(df, district=None):
    if df.empty:
        return []
    filtered = df[df['片区'] == district] if district else df
    return sorted(filtered['管段编号'].unique().tolist())


def calculate_statistics(df, district=None, batches=None, rules=None):
    rules = get_effective_rules(rules)
    if df.empty:
        return {}

    filtered = df.copy()
    if district:
        filtered = filtered[filtered['片区'] == district]
    if batches:
        filtered = filtered[filtered['巡检批次'].isin(batches)]

    if filtered.empty:
        return {}

    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))

    stats = {}
    stats['记录总数'] = len(filtered)
    stats['管段数量'] = filtered['管段编号'].nunique()
    stats['巡检批次数量'] = filtered['巡检批次'].nunique()

    stats['最大淤积深度'] = round(filtered['淤积深度'].max(), 2)
    stats['平均淤积深度'] = round(filtered['淤积深度'].mean(), 2)
    stats['最大淤积率'] = round(filtered['淤积率'].max(), 4)
    stats['平均淤积率'] = round(filtered['淤积率'].mean(), 4)

    high_risk = filtered[filtered['淤积率'] >= threshold_high]
    medium_risk = filtered[(filtered['淤积率'] >= threshold_medium) & (filtered['淤积率'] < threshold_high)]
    low_risk = filtered[filtered['淤积率'] < threshold_medium]

    stats['高风险管段数'] = high_risk['管段编号'].nunique()
    stats['中风险管段数'] = medium_risk['管段编号'].nunique()
    stats['低风险管段数'] = low_risk['管段编号'].nunique()

    stats['高风险记录数'] = len(high_risk)
    stats['中风险记录数'] = len(medium_risk)
    stats['低风险记录数'] = len(low_risk)

    return stats


def detect_abnormal_growth(df, district=None, batches=None, rules=None):
    rules = get_effective_rules(rules)
    if df.empty:
        return pd.DataFrame()

    filtered = df.copy()
    if district:
        filtered = filtered[filtered['片区'] == district]

    if filtered.empty:
        return pd.DataFrame()

    growth_rate_threshold = float(rules.get('ABNORMAL_GROWTH_RATE', ABNORMAL_GROWTH_RATE))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))

    filtered = filtered.sort_values(['管段编号', '检查时间'])
    abnormal_pipes = []

    for pipe_id, group in filtered.groupby('管段编号'):
        group = group.sort_values('检查时间')
        if len(group) >= 2:
            for i in range(1, len(group)):
                prev_row = group.iloc[i - 1]
                curr_row = group.iloc[i]

                if batches:
                    if prev_row['巡检批次'] not in batches or curr_row['巡检批次'] not in batches:
                        continue

                prev_rate = prev_row['淤积率']
                curr_rate = curr_row['淤积率']
                growth_rate = (curr_rate - prev_rate) / prev_rate if prev_rate > 0 else float('inf')

                if growth_rate >= growth_rate_threshold and curr_rate > threshold_medium:
                    abnormal_pipes.append({
                        '管段编号': pipe_id,
                        '片区': curr_row['片区'],
                        '前批次': prev_row['巡检批次'],
                        '当前批次': curr_row['巡检批次'],
                        '前淤积率': round(prev_rate, 4),
                        '当前淤积率': round(curr_rate, 4),
                        '增长率': round(growth_rate, 4),
                        '管径': curr_row['管径'],
                        '前淤积深度': prev_row['淤积深度'],
                        '当前淤积深度': curr_row['淤积深度']
                    })

    return pd.DataFrame(abnormal_pipes)


def get_high_risk_segments(df, district=None, batches=None, rules=None):
    rules = get_effective_rules(rules)
    if df.empty:
        return pd.DataFrame()

    filtered = df.copy()
    if district:
        filtered = filtered[filtered['片区'] == district]
    if batches:
        filtered = filtered[filtered['巡检批次'].isin(batches)]

    if filtered.empty:
        return pd.DataFrame()

    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    high_risk = filtered[filtered['淤积率'] >= threshold_high].copy()
    high_risk = high_risk.sort_values('淤积率', ascending=False)

    return high_risk[['管段编号', '巡检批次', '检查时间', '淤积深度', '管径', '淤积率', '片区', '备注']]


def get_pipe_history(df, pipe_id, district=None):
    if df.empty:
        return pd.DataFrame()

    filtered = df[df['管段编号'] == pipe_id]
    if district:
        filtered = filtered[filtered['片区'] == district]

    filtered = filtered.sort_values('检查时间')
    return filtered


def get_risk_heatmap_data(df, district=None, batches=None):
    if df.empty:
        return pd.DataFrame()

    filtered = df.copy()
    if district:
        filtered = filtered[filtered['片区'] == district]
    if batches:
        filtered = filtered[filtered['巡检批次'].isin(batches)]

    if filtered.empty:
        return pd.DataFrame()

    heatmap_data = filtered.pivot_table(
        index='管段编号',
        columns='巡检批次',
        values='淤积率',
        aggfunc='max'
    ).fillna(-1)

    return heatmap_data


def detect_missing_inspections(df, district=None, batches=None):
    if df.empty:
        return pd.DataFrame()

    filtered = df.copy()
    if district:
        filtered = filtered[filtered['片区'] == district]

    if filtered.empty:
        return pd.DataFrame()

    if batches:
        check_batches = sorted([b for b in batches if b in filtered['巡检批次'].unique()])
    else:
        check_batches = sorted(filtered['巡检批次'].unique().tolist())

    pipes_in_batches = filtered[filtered['巡检批次'].isin(check_batches)]['管段编号'].unique().tolist()
    all_pipes = sorted(pipes_in_batches)

    missing_records = []
    for pipe_id in all_pipes:
        pipe_data = filtered[filtered['管段编号'] == pipe_id]
        pipe_batches = set(pipe_data['巡检批次'].tolist())

        for batch in check_batches:
            if batch not in pipe_batches:
                missing_records.append({
                    '管段编号': pipe_id,
                    '片区': district if district else pipe_data['片区'].iloc[0] if len(pipe_data) > 0 else '未知',
                    '缺失批次': batch
                })

    return pd.DataFrame(missing_records)


def get_batch_comparison_data(df, pipe_ids, district=None):
    if df.empty or not pipe_ids:
        return pd.DataFrame()

    filtered = df[df['管段编号'].isin(pipe_ids)]
    if district:
        filtered = filtered[filtered['片区'] == district]

    return filtered.sort_values(['管段编号', '检查时间'])


def revalidate_dataframe(df, rules=None):
    rules = get_effective_rules(rules)
    errors = []
    valid_rows = []

    if '片区' not in df.columns:
        df['片区'] = '默认片区'

    df = df.reset_index(drop=True)

    for idx, row in df.iterrows():
        row_num = idx + 1
        row_errors = []

        pipe_id = str(row.get('管段编号', '')).strip()
        if not pipe_id or pipe_id == 'nan':
            row_errors.append('管段编号为空')

        batch = str(row.get('巡检批次', '')).strip()
        if not batch or batch == 'nan':
            row_errors.append('巡检批次为空')

        try:
            check_date_val = row.get('检查时间', '')
            if pd.isna(check_date_val):
                row_errors.append('检查时间为空')
                check_date = pd.NaT
            else:
                check_date = pd.to_datetime(check_date_val)
                if pd.isna(check_date):
                    row_errors.append('检查时间格式无效')
        except Exception:
            row_errors.append('检查时间格式无效')
            check_date = pd.NaT

        try:
            sediment_val = row.get('淤积深度', np.nan)
            if pd.isna(sediment_val) or str(sediment_val).strip() == '':
                row_errors.append('淤积深度为空')
                sediment_depth = np.nan
            else:
                sediment_depth = float(sediment_val)
                if sediment_depth < 0:
                    row_errors.append(f'淤积深度为负数 ({sediment_depth})')
        except (ValueError, TypeError):
            row_errors.append('淤积深度不是有效数字')
            sediment_depth = np.nan

        try:
            diameter_val = row.get('管径', np.nan)
            if pd.isna(diameter_val) or str(diameter_val).strip() == '':
                row_errors.append('管径为空')
                diameter = np.nan
            else:
                diameter = float(diameter_val)
                if diameter <= 0:
                    row_errors.append(f'管径必须大于0 ({diameter})')
        except (ValueError, TypeError):
            row_errors.append('管径不是有效数字')
            diameter = np.nan

        if not pd.isna(sediment_depth) and not pd.isna(diameter):
            if sediment_depth > diameter:
                row_errors.append(f'淤积深度 ({sediment_depth}) 超过管径 ({diameter})')

        remark = str(row.get('备注', '')) if pd.notna(row.get('备注', '')) else ''
        district_val = str(row.get('片区', '默认片区')) if pd.notna(row.get('片区', '默认片区')) else '默认片区'

        if row_errors:
            errors.append({
                '行号': row_num,
                '管段编号': pipe_id if pipe_id else '未知',
                '巡检批次': batch if batch else '未知',
                '错误原因': '；'.join(row_errors),
                '原始数据': str(row.to_dict())
            })
        else:
            valid_rows.append({
                '管段编号': pipe_id,
                '巡检批次': batch,
                '检查时间': check_date,
                '淤积深度': sediment_depth,
                '管径': diameter,
                '备注': remark,
                '片区': district_val,
                '淤积率': round(sediment_depth / diameter, 4) if diameter > 0 else 0
            })

    valid_df = pd.DataFrame(valid_rows)

    if not valid_df.empty:
        dup_mask = valid_df.duplicated(subset=['管段编号', '巡检批次', '片区'], keep=False)
        duplicates = valid_df[dup_mask]
        duplicate_errors = []
        if not duplicates.empty:
            for (pid, batch, district_val), group in duplicates.groupby(['管段编号', '巡检批次', '片区']):
                for _, dup_row in group.iloc[1:].iterrows():
                    original_idx = valid_df.index.get_loc(dup_row.name)
                    duplicate_errors.append({
                        '行号': f'第{original_idx + 1}条',
                        '管段编号': pid,
                        '巡检批次': batch,
                        '错误原因': f'同一管段在同一巡检批次({batch})内重复记录',
                        '原始数据': str(dup_row.to_dict())
                    })
            valid_df = valid_df.drop_duplicates(subset=['管段编号', '巡检批次', '片区'], keep='first')
            errors.extend(duplicate_errors)

    return valid_df, errors


def validate_batch_data(df, batch_name=None, rules=None):
    rules = get_effective_rules(rules)
    if df.empty:
        return pd.DataFrame(), [], []

    filtered = df.copy()
    if batch_name:
        filtered = filtered[filtered['巡检批次'] == batch_name]

    if filtered.empty:
        return pd.DataFrame(), [], ['指定批次无数据']

    errors = []
    repaired_rows = []
    repair_actions = []

    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))

    for idx, row in filtered.iterrows():
        row_num = idx + 1
        row_issues = []
        row_repairs = []
        modified = False

        pipe_id = str(row.get('管段编号', '')).strip()
        if not pipe_id or pipe_id == 'nan':
            row_issues.append('管段编号为空')

        batch = str(row.get('巡检批次', '')).strip()
        if not batch or batch == 'nan':
            row_issues.append('巡检批次为空')

        check_date = row.get('检查时间', pd.NaT)
        if pd.isna(check_date):
            row_issues.append('检查时间为空')

        sediment_depth = row.get('淤积深度', np.nan)
        if pd.isna(sediment_depth):
            row_issues.append('淤积深度为空')
        else:
            try:
                sediment_depth = float(sediment_depth)
                if sediment_depth < 0:
                    sediment_depth = abs(sediment_depth)
                    modified = True
                    row_repairs.append(f'淤积深度取绝对值: {row.get("淤积深度")} → {sediment_depth}')
            except (ValueError, TypeError):
                row_issues.append('淤积深度非数字')

        diameter = row.get('管径', np.nan)
        if pd.isna(diameter):
            row_issues.append('管径为空')
        else:
            try:
                diameter = float(diameter)
                if diameter <= 0:
                    row_issues.append(f'管径无效 ({diameter})')
            except (ValueError, TypeError):
                row_issues.append('管径非数字')

        sediment_rate = row.get('淤积率', np.nan)
        if not pd.isna(sediment_depth) and not pd.isna(diameter) and diameter > 0:
            calculated_rate = round(sediment_depth / diameter, 4)
            if pd.isna(sediment_rate) or abs(calculated_rate - float(sediment_rate)) > 0.001:
                modified = True
                row_repairs.append(f'淤积率重算: {sediment_rate} → {calculated_rate}')
                sediment_rate = calculated_rate

        if not pd.isna(sediment_depth) and not pd.isna(diameter) and sediment_depth > diameter:
            sediment_depth = diameter * 0.95
            sediment_rate = round(sediment_depth / diameter, 4)
            modified = True
            row_repairs.append(f'淤积深度超过管径，修正为管径95%: {sediment_rate:.1%}')

        district_val = str(row.get('片区', '默认片区')) if pd.notna(row.get('片区', '默认片区')) else '默认片区'
        remark = str(row.get('备注', '')) if pd.notna(row.get('备注', '')) else ''

        if row_issues:
            errors.append({
                '行号': row_num,
                '管段编号': pipe_id if pipe_id else '未知',
                '巡检批次': batch if batch else '未知',
                '问题': '；'.join(row_issues),
                '可修复': len(row_repairs) > 0
            })

        if row_issues and not row_repairs:
            continue

        repaired_rows.append({
            '管段编号': pipe_id,
            '片区': district_val,
            '巡检批次': batch,
            '检查时间': check_date,
            '淤积深度': sediment_depth if not pd.isna(sediment_depth) else 0,
            '管径': diameter if not pd.isna(diameter) else 0,
            '淤积率': sediment_rate if not pd.isna(sediment_rate) else 0,
            '备注': remark + (' [已修复]' if modified else ''),
            '_modified': modified
        })

        if row_repairs:
            repair_actions.append({
                '行号': row_num,
                '管段编号': pipe_id,
                '修复操作': '；'.join(row_repairs)
            })

    repaired_df = pd.DataFrame(repaired_rows)
    if not repaired_df.empty and '_modified' in repaired_df.columns:
        repaired_df = repaired_df.drop(columns=['_modified'])

    return repaired_df, errors, repair_actions


def validate_cross_district(pipes_compare_ids, df):
    if not pipes_compare_ids or df.empty:
        return True, ''

    pipe_districts = df[df['管段编号'].isin(pipes_compare_ids)].groupby('管段编号')['片区'].first()

    districts_in_selection = pipe_districts.unique()
    if len(districts_in_selection) > 1:
        district_pipes = {}
        for pid in pipes_compare_ids:
            d = pipe_districts.get(pid, '未知')
            if d not in district_pipes:
                district_pipes[d] = []
            district_pipes[d].append(pid)

        detail = '；'.join([f'{d}区: {", ".join(pids)}' for d, pids in district_pipes.items()])
        return False, f'禁止跨片区直接合并对比！所选管段跨越多个片区：{detail}。请按片区分别对比分析。'

    return True, ''


def evaluate_dredging_effect(df, pipe_id, pre_batch, post_batch, rules=None):
    rules = get_effective_rules(rules)
    if df.empty or not pipe_id:
        return None

    threshold = float(rules.get('DREDGING_EFFECT_THRESHOLD', 0.3))

    pipe_data = df[df['管段编号'] == pipe_id].sort_values('检查时间')
    if pipe_data.empty:
        return None

    pre_data = pipe_data[pipe_data['巡检批次'] == pre_batch]
    post_data = pipe_data[pipe_data['巡检批次'] == post_batch]

    if pre_data.empty or post_data.empty:
        return None

    pre_row = pre_data.iloc[0]
    post_row = post_data.iloc[0]

    pre_rate = pre_row['淤积率']
    post_rate = post_row['淤积率']
    pre_depth = pre_row['淤积深度']
    post_depth = post_row['淤积深度']
    diameter = pre_row['管径']

    depth_reduction = pre_depth - post_depth
    rate_reduction = pre_rate - post_rate
    reduction_pct = rate_reduction / pre_rate if pre_rate > 0 else 0

    if reduction_pct >= threshold:
        effect_level = '显著有效'
        effect_color = '#27ae60'
    elif reduction_pct >= 0.1:
        effect_level = '部分有效'
        effect_color = '#f39c12'
    elif reduction_pct >= 0:
        effect_level = '效果不明显'
        effect_color = '#e67e22'
    else:
        effect_level = '淤积加重'
        effect_color = '#e74c3c'

    suggestion = _generate_dredging_suggestion(post_rate, reduction_pct, rules)

    return {
        '管段编号': pipe_id,
        '片区': pre_row['片区'],
        '清淤前批次': pre_batch,
        '清淤后批次': post_batch,
        '清淤前淤积深度': round(pre_depth, 2),
        '清淤后淤积深度': round(post_depth, 2),
        '深度减少量': round(depth_reduction, 2),
        '清淤前淤积率': round(pre_rate, 4),
        '清淤后淤积率': round(post_rate, 4),
        '淤积率降幅': round(rate_reduction, 4),
        '降幅百分比': round(reduction_pct, 4),
        '效果评级': effect_level,
        '效果颜色': effect_color,
        '处理建议': suggestion
    }


def batch_evaluate_dredging(df, rules=None):
    rules = get_effective_rules(rules)
    if df.empty:
        return pd.DataFrame()

    results = []
    for pipe_id in df['管段编号'].unique():
        pipe_data = df[df['管段编号'] == pipe_id].sort_values('检查时间')
        if len(pipe_data) < 2:
            continue

        for i in range(1, len(pipe_data)):
            prev_row = pipe_data.iloc[i - 1]
            curr_row = pipe_data.iloc[i]

            prev_rate = prev_row['淤积率']
            curr_rate = curr_row['淤积率']

            if curr_rate < prev_rate and prev_rate > 0:
                depth_reduction = prev_row['淤积深度'] - curr_row['淤积深度']
                rate_reduction = prev_rate - curr_rate
                reduction_pct = rate_reduction / prev_rate

                if reduction_pct >= 0.05:
                    result = evaluate_dredging_effect(
                        df, pipe_id,
                        prev_row['巡检批次'], curr_row['巡检批次'],
                        rules
                    )
                    if result:
                        results.append(result)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values('降幅百分比', ascending=False)


def calculate_dredging_priority(df, rules=None):
    rules = get_effective_rules(rules)
    if df.empty:
        return pd.DataFrame()

    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))
    growth_threshold = float(rules.get('ABNORMAL_GROWTH_RATE', ABNORMAL_GROWTH_RATE))

    latest_per_pipe = df.sort_values('检查时间').groupby('管段编号').last().reset_index()

    priority_records = []
    for _, row in latest_per_pipe.iterrows():
        pipe_id = row['管段编号']
        current_rate = row['淤积率']
        current_depth = row['淤积深度']
        diameter = row['管径']
        district = row['片区']

        pipe_history = df[df['管段编号'] == pipe_id].sort_values('检查时间')

        growth_rate = 0
        if len(pipe_history) >= 2:
            prev_rate = pipe_history.iloc[-2]['淤积率']
            if prev_rate > 0:
                growth_rate = (current_rate - prev_rate) / prev_rate

        is_abnormal = growth_rate >= growth_threshold
        is_missing = len(pipe_history) < df['巡检批次'].nunique()

        risk_score = 0
        if current_rate >= threshold_high:
            risk_score += 40
        elif current_rate >= threshold_medium:
            risk_score += 20

        risk_score += min(growth_rate * 30, 25)

        if is_abnormal:
            risk_score += 20
        if is_missing:
            risk_score += 10

        if current_rate >= threshold_high and is_abnormal:
            priority_level = '紧急'
        elif current_rate >= threshold_high:
            priority_level = '高'
        elif current_rate >= threshold_medium and is_abnormal:
            priority_level = '高'
        elif current_rate >= threshold_medium:
            priority_level = '中'
        elif is_abnormal:
            priority_level = '中'
        else:
            priority_level = '低'

        suggestion = _generate_dredging_suggestion(current_rate, growth_rate, rules)

        priority_records.append({
            '管段编号': pipe_id,
            '片区': district,
            '最新淤积率': round(current_rate, 4),
            '最新淤积深度': round(current_depth, 2),
            '管径': diameter,
            '增长率': round(growth_rate, 4),
            '异常增长': '是' if is_abnormal else '否',
            '巡检缺失': '是' if is_missing else '否',
            '风险评分': round(risk_score, 1),
            '清淤优先级': priority_level,
            '处理建议': suggestion
        })

    priority_df = pd.DataFrame(priority_records)
    if not priority_df.empty:
        priority_order = {'紧急': 0, '高': 1, '中': 2, '低': 3}
        priority_df['_sort'] = priority_df['清淤优先级'].map(priority_order)
        priority_df = priority_df.sort_values(['_sort', '风险评分'], ascending=[True, False])
        priority_df = priority_df.drop(columns=['_sort'])

    return priority_df


def _generate_dredging_suggestion(current_rate, reduction_or_growth, rules=None):
    rules = get_effective_rules(rules)
    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))

    if current_rate >= threshold_high:
        return '建议立即安排清淤作业，优先使用高压冲洗方式，清理后安排复检确认效果'
    elif current_rate >= threshold_medium:
        if reduction_or_growth > 0.2:
            return '淤积增长较快，建议近期安排清淤，可采用机械清淤方式，并加密巡检频次'
        return '建议列入本季度清淤计划，采用常规清淤方式处理'
    else:
        if reduction_or_growth > 0.2:
            return '淤积增长较快但尚在安全范围，建议加强监控，缩短巡检周期至每月一次'
        return '暂无需清淤，保持当前巡检频率即可'


def generate_inspection_quality_report(df, rules=None):
    rules = get_effective_rules(rules)
    if df.empty:
        return {}

    report = {}

    total_records = len(df)
    total_pipes = df['管段编号'].nunique()
    total_batches = df['巡检批次'].nunique()
    total_districts = df['片区'].nunique()

    report['总览'] = {
        '记录总数': total_records,
        '管段总数': total_pipes,
        '巡检批次数': total_batches,
        '片区数': total_districts
    }

    missing_df = detect_missing_inspections(df)
    missing_count = len(missing_df)
    expected_records = total_pipes * total_batches
    coverage_rate = (total_records / expected_records * 100) if expected_records > 0 else 100

    report['巡检覆盖率'] = {
        '应检记录数': expected_records,
        '实检记录数': total_records,
        '覆盖率': f'{coverage_rate:.1f}%',
        '缺失巡检数': missing_count
    }

    if missing_count > 0:
        missing_by_district = missing_df.groupby('片区').size().to_dict()
        missing_by_batch = missing_df.groupby('缺失批次').size().to_dict()
        report['巡检覆盖率']['按片区缺失'] = missing_by_district
        report['巡检覆盖率']['按批次缺失'] = missing_by_batch

    abnormal_df = detect_abnormal_growth(df, rules=rules)
    report['异常增长统计'] = {
        '异常增长管段数': abnormal_df['管段编号'].nunique() if not abnormal_df.empty else 0,
        '异常增长记录数': len(abnormal_df)
    }

    if not abnormal_df.empty:
        abnormal_by_district = abnormal_df.groupby('片区')['管段编号'].nunique().to_dict()
        report['异常增长统计']['按片区统计'] = abnormal_by_district

    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    high_risk_df = df[df['淤积率'] >= threshold_high]
    report['高风险统计'] = {
        '高风险管段数': high_risk_df['管段编号'].nunique(),
        '高风险记录数': len(high_risk_df)
    }

    if not high_risk_df.empty:
        high_risk_by_district = high_risk_df.groupby('片区')['管段编号'].nunique().to_dict()
        report['高风险统计']['按片区统计'] = high_risk_by_district

    depth_negative = df[df['淤积深度'] < 0]
    rate_over_one = df[df['淤积率'] > 1]
    report['数据质量'] = {
        '淤积深度为负': len(depth_negative),
        '淤积率超100%': len(rate_over_one),
        '重复记录': len(df[df.duplicated(subset=['管段编号', '巡检批次', '片区'], keep=False)])
    }

    quality_issues = []
    if coverage_rate < 90:
        quality_issues.append(f'巡检覆盖率仅{coverage_rate:.1f}%，低于90%标准，存在{missing_count}条缺失巡检')
    if not abnormal_df.empty and abnormal_df['管段编号'].nunique() > total_pipes * 0.2:
        quality_issues.append(f'异常增长管段占比超过20%，需关注巡检数据准确性')
    if len(depth_negative) > 0:
        quality_issues.append(f'存在{len(depth_negative)}条淤积深度为负的异常记录')
    if len(rate_over_one) > 0:
        quality_issues.append(f'存在{len(rate_over_one)}条淤积率超100%的异常记录')

    if not quality_issues:
        quality_issues.append('巡检数据整体质量良好，未发现重大质量问题')

    report['质量评估结论'] = quality_issues

    return report


def merge_repaired_batch(original_df, repaired_df, batch_name=None):
    if original_df is None or original_df.empty:
        return repaired_df if repaired_df is not None else pd.DataFrame()

    if repaired_df is None or repaired_df.empty:
        if batch_name:
            return original_df[original_df['巡检批次'] != batch_name].copy()
        return original_df.copy()

    result = original_df.copy()

    if batch_name:
        result = result[result['巡检批次'] != batch_name]

    result = pd.concat([result, repaired_df], ignore_index=True)
    result = result.sort_values(['片区', '管段编号', '检查时间']).reset_index(drop=True)

    return result


def merge_edited_subset(original_df, edited_df, district=None, batches=None):
    if original_df is None or original_df.empty:
        return edited_df if edited_df is not None else pd.DataFrame()

    if edited_df is None:
        return original_df.copy()

    result = original_df.copy()

    mask = pd.Series([True] * len(result), index=result.index)
    if district:
        mask &= (result['片区'] == district)
    if batches:
        mask &= result['巡检批次'].isin(batches)

    result = result[~mask]

    if not edited_df.empty:
        result = pd.concat([result, edited_df], ignore_index=True)

    result = result.sort_values(['片区', '管段编号', '检查时间']).reset_index(drop=True)

    return result


def get_district_summary(df, rules=None):
    rules = get_effective_rules(rules)
    if df.empty:
        return pd.DataFrame()

    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))

    summaries = []
    for district in df['片区'].unique():
        district_data = df[df['片区'] == district]
        stats = calculate_statistics(district_data, rules=rules)
        abnormal = detect_abnormal_growth(district_data, rules=rules)
        missing = detect_missing_inspections(district_data)

        summaries.append({
            '片区': district,
            '管段数': district_data['管段编号'].nunique(),
            '记录数': len(district_data),
            '平均淤积率': round(district_data['淤积率'].mean(), 4),
            '最大淤积率': round(district_data['淤积率'].max(), 4),
            '高风险管段数': stats.get('高风险管段数', 0),
            '中风险管段数': stats.get('中风险管段数', 0),
            '异常增长数': len(abnormal),
            '缺失巡检数': len(missing)
        })

    return pd.DataFrame(summaries)


TASK_TYPES = {
    'INSPECTION': '巡检',
    'REINSPECTION': '复检',
    'DREDGING': '清淤'
}

TASK_STATUS = {
    'PENDING': '待派发',
    'ASSIGNED': '已派发',
    'IN_PROGRESS': '处理中',
    'COMPLETED': '已完成',
    'OVERDUE': '已超期',
    'CLOSED': '已闭环'
}

PRIORITY_LEVELS = {
    'URGENT': {'name': '紧急', 'score_range': (80, 100), 'deadline_days': 3, 'color': '#c0392b'},
    'HIGH': {'name': '高', 'score_range': (60, 80), 'deadline_days': 7, 'color': '#e67e22'},
    'MEDIUM': {'name': '中', 'score_range': (30, 60), 'deadline_days': 14, 'color': '#f39c12'},
    'LOW': {'name': '低', 'score_range': (0, 30), 'deadline_days': 30, 'color': '#27ae60'}
}


def _calculate_task_priority_score(pipe_info, rules=None):
    rules = get_effective_rules(rules)
    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))
    growth_threshold = float(rules.get('ABNORMAL_GROWTH_RATE', ABNORMAL_GROWTH_RATE))

    score = 0
    current_rate = pipe_info.get('最新淤积率', 0)
    growth_rate = pipe_info.get('增长率', 0)
    is_abnormal = pipe_info.get('异常增长', '否') == '是'
    is_missing = pipe_info.get('巡检缺失', '否') == '是'

    if current_rate >= threshold_high:
        score += 40
    elif current_rate >= threshold_medium:
        score += 20

    score += min(growth_rate * 30, 25)

    if is_abnormal:
        score += 15
    if is_missing:
        score += 10

    pipe_diameter = pipe_info.get('管径', 500)
    if pipe_diameter >= 800:
        score += 10
    elif pipe_diameter >= 600:
        score += 5

    score = min(score, 100)

    priority_key = 'LOW'
    for key, level in PRIORITY_LEVELS.items():
        min_score, max_score = level['score_range']
        if min_score <= score < max_score or (key == 'URGENT' and score >= min_score):
            priority_key = key
            break

    return score, priority_key


def _determine_task_type(pipe_info, rules=None):
    rules = get_effective_rules(rules)
    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))
    growth_threshold = float(rules.get('ABNORMAL_GROWTH_RATE', ABNORMAL_GROWTH_RATE))

    current_rate = pipe_info.get('最新淤积率', 0)
    growth_rate = pipe_info.get('增长率', 0)
    is_missing = pipe_info.get('巡检缺失', '否') == '是'
    remark = str(pipe_info.get('备注', ''))

    if is_missing:
        return 'INSPECTION'

    if '清淤后' in remark or '复检' in remark:
        if current_rate >= threshold_medium:
            return 'REINSPECTION'
        return 'REINSPECTION'

    if current_rate >= threshold_high:
        return 'DREDGING'
    elif current_rate >= threshold_medium and growth_rate >= growth_threshold:
        return 'DREDGING'
    elif current_rate >= threshold_medium:
        return 'INSPECTION'
    elif growth_rate >= growth_threshold:
        return 'INSPECTION'
    else:
        return 'INSPECTION'


def generate_inspection_tasks(df, district=None, batch_name=None, rules=None,
                              max_tasks_per_batch=50, exclude_completed=True,
                              existing_tasks_df=None):
    rules = get_effective_rules(rules)
    if df.empty:
        return pd.DataFrame()

    priority_df = calculate_dredging_priority(df, rules=rules)
    if priority_df.empty:
        return pd.DataFrame()

    if district:
        priority_df = priority_df[priority_df['片区'] == district]

    latest_per_pipe = df.sort_values('检查时间').groupby('管段编号').last().reset_index()
    latest_map = latest_per_pipe.set_index('管段编号').to_dict('index')

    abnormal_df = detect_abnormal_growth(df, district=district, rules=rules)
    abnormal_pipes = set(abnormal_df['管段编号'].unique()) if not abnormal_df.empty else set()

    missing_df = detect_missing_inspections(df, district=district, batches=[batch_name] if batch_name else None)
    missing_pipes = set(zip(missing_df['管段编号'], missing_df['缺失批次'])) if not missing_df.empty else set()
    missing_pipe_ids = set(missing_df['管段编号'].unique()) if not missing_df.empty else set()

    completed_pipes = set()
    if exclude_completed and existing_tasks_df is not None and not existing_tasks_df.empty:
        completed = existing_tasks_df[existing_tasks_df['任务状态'].isin(['已完成', '已闭环'])]
        if not completed.empty:
            completed_pipes = set(completed['管段编号'].unique())

    tasks = []
    task_counter = 0

    for _, pipe_row in priority_df.iterrows():
        if task_counter >= max_tasks_per_batch:
            break

        pipe_id = pipe_row['管段编号']
        if pipe_id in completed_pipes:
            continue

        pipe_latest = latest_map.get(pipe_id, {})
        latest_batch = pipe_latest.get('巡检批次', '')
        latest_depth = pipe_latest.get('淤积深度', 0)
        latest_rate = pipe_latest.get('淤积率', 0)
        remark = str(pipe_latest.get('备注', ''))

        pipe_info = pipe_row.to_dict()
        pipe_info['备注'] = remark

        task_type = _determine_task_type(pipe_info, rules=rules)
        score, priority_key = _calculate_task_priority_score(pipe_info, rules=rules)
        priority_info = PRIORITY_LEVELS[priority_key]

        missing_batches = []
        for (mp, mb) in missing_pipes:
            if mp == pipe_id:
                missing_batches.append(mb)

        trigger_reasons = []
        if latest_rate >= float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH)):
            trigger_reasons.append('高风险淤积')
        if latest_rate >= float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM)):
            trigger_reasons.append('中风险淤积')
        if pipe_id in abnormal_pipes:
            trigger_reasons.append('异常增长')
        if pipe_id in missing_pipe_ids:
            trigger_reasons.append('缺失巡检')
        if '清淤后' in remark:
            trigger_reasons.append('清淤后待复检')

        if not trigger_reasons:
            trigger_reasons.append('常规巡检')

        created_at = datetime.now()
        deadline_days = priority_info['deadline_days']
        deadline = created_at + pd.Timedelta(days=deadline_days)

        task_id = f"TASK{created_at.strftime('%Y%m%d%H%M%S')}{task_counter + 1:04d}"

        tasks.append({
            '任务编号': task_id,
            '管段编号': pipe_id,
            '片区': pipe_row['片区'],
            '任务类型': TASK_TYPES.get(task_type, task_type),
            '任务类型编码': task_type,
            '清淤优先级': pipe_row['清淤优先级'],
            '动态优先级': priority_info['name'],
            '优先级评分': round(score, 1),
            '优先级颜色': priority_info['color'],
            '截止天数': deadline_days,
            '最新淤积率': round(latest_rate, 4),
            '最新淤积深度(mm)': round(latest_depth, 2),
            '管径(mm)': pipe_row.get('管径', 0),
            '增长率': round(pipe_row.get('增长率', 0), 4),
            '异常增长': pipe_row.get('异常增长', '否'),
            '巡检缺失': '是' if pipe_id in missing_pipe_ids else '否',
            '缺失批次列表': ';'.join(missing_batches) if missing_batches else '',
            '关联巡检批次': latest_batch,
            '触发原因': '、'.join(trigger_reasons),
            '任务状态': '待派发',
            '任务状态编码': 'PENDING',
            '派发人员': '',
            '派发时间': None,
            '处理人员': '',
            '处理开始时间': None,
            '处理完成时间': None,
            '处理结果': '',
            '处理备注': '',
            '处理后淤积深度(mm)': None,
            '处理后淤积率': None,
            '整改前淤积率': round(latest_rate, 4),
            '整改效果评级': '',
            '闭环时间': None,
            '闭环确认人': '',
            '创建时间': created_at,
            '截止时间': deadline,
            '是否超期': '否',
            '超期天数': 0
        })
        task_counter += 1

    if not tasks:
        return pd.DataFrame()

    tasks_df = pd.DataFrame(tasks)
    priority_order = {'紧急': 0, '高': 1, '中': 2, '低': 3}
    tasks_df['_sort_priority'] = tasks_df['动态优先级'].map(priority_order)
    tasks_df = tasks_df.sort_values(['_sort_priority', '优先级评分'], ascending=[True, False])
    tasks_df = tasks_df.drop(columns=['_sort_priority'])
    tasks_df = tasks_df.reset_index(drop=True)

    return tasks_df


def assign_task(tasks_df, task_id, assignee, assigner='系统'):
    if tasks_df is None or tasks_df.empty:
        return tasks_df

    mask = tasks_df['任务编号'] == task_id
    if not mask.any():
        return tasks_df

    now = datetime.now()
    tasks_df.loc[mask, '任务状态'] = '已派发'
    tasks_df.loc[mask, '任务状态编码'] = 'ASSIGNED'
    tasks_df.loc[mask, '派发人员'] = f"{assigner} → {assignee}"
    tasks_df.loc[mask, '派发时间'] = now
    tasks_df.loc[mask, '处理人员'] = assignee

    return tasks_df


def batch_assign_tasks(tasks_df, task_ids, assignee, assigner='系统'):
    if tasks_df is None or tasks_df.empty:
        return tasks_df

    for tid in task_ids:
        tasks_df = assign_task(tasks_df, tid, assignee, assigner)

    return tasks_df


def start_task(tasks_df, task_id, processor=None):
    if tasks_df is None or tasks_df.empty:
        return tasks_df

    mask = tasks_df['任务编号'] == task_id
    if not mask.any():
        return tasks_df

    now = datetime.now()
    current_assignee = tasks_df.loc[mask, '处理人员'].iloc[0]
    tasks_df.loc[mask, '任务状态'] = '处理中'
    tasks_df.loc[mask, '任务状态编码'] = 'IN_PROGRESS'
    tasks_df.loc[mask, '处理开始时间'] = now
    if processor and not current_assignee:
        tasks_df.loc[mask, '处理人员'] = processor

    return tasks_df


def complete_task(tasks_df, task_id, result, result_note='',
                  post_depth=None, post_rate=None, rules=None):
    rules = get_effective_rules(rules)
    if tasks_df is None or tasks_df.empty:
        return tasks_df

    mask = tasks_df['任务编号'] == task_id
    if not mask.any():
        return tasks_df

    now = datetime.now()
    tasks_df.loc[mask, '任务状态'] = '已完成'
    tasks_df.loc[mask, '任务状态编码'] = 'COMPLETED'
    tasks_df.loc[mask, '处理完成时间'] = now
    tasks_df.loc[mask, '处理结果'] = result
    tasks_df.loc[mask, '处理备注'] = result_note

    pre_rate = tasks_df.loc[mask, '整改前淤积率'].iloc[0]
    task_type_code = tasks_df.loc[mask, '任务类型编码'].iloc[0]

    if post_depth is not None:
        tasks_df.loc[mask, '处理后淤积深度(mm)'] = float(post_depth)
    if post_rate is not None:
        tasks_df.loc[mask, '处理后淤积率'] = float(post_rate)
    elif post_depth is not None and task_type_code == 'DREDGING':
        diameter = tasks_df.loc[mask, '管径(mm)'].iloc[0]
        if diameter > 0:
            calc_rate = round(float(post_depth) / diameter, 4)
            tasks_df.loc[mask, '处理后淤积率'] = calc_rate

    if task_type_code == 'DREDGING' and tasks_df.loc[mask, '处理后淤积率'].iloc[0] is not None:
        actual_post_rate = tasks_df.loc[mask, '处理后淤积率'].iloc[0]
        if pre_rate > 0:
            reduction = (pre_rate - actual_post_rate) / pre_rate
            threshold = float(rules.get('DREDGING_EFFECT_THRESHOLD', 0.3))
            if reduction >= threshold:
                effect = '显著有效'
            elif reduction >= 0.1:
                effect = '部分有效'
            elif reduction >= 0:
                effect = '效果不明显'
            else:
                effect = '淤积加重'
            tasks_df.loc[mask, '整改效果评级'] = effect

    _check_task_overdue(tasks_df, task_id)

    return tasks_df


def close_task(tasks_df, task_id, confirmer='系统管理员'):
    if tasks_df is None or tasks_df.empty:
        return tasks_df

    mask = tasks_df['任务编号'] == task_id
    if not mask.any():
        return tasks_df

    status = tasks_df.loc[mask, '任务状态'].iloc[0]
    if status not in ['已完成', '已超期']:
        return tasks_df

    now = datetime.now()
    tasks_df.loc[mask, '任务状态'] = '已闭环'
    tasks_df.loc[mask, '任务状态编码'] = 'CLOSED'
    tasks_df.loc[mask, '闭环时间'] = now
    tasks_df.loc[mask, '闭环确认人'] = confirmer

    return tasks_df


def _check_task_overdue(tasks_df, task_id=None):
    if tasks_df is None or tasks_df.empty:
        return tasks_df

    now = datetime.now()

    if task_id:
        mask = tasks_df['任务编号'] == task_id
    else:
        mask = pd.Series([True] * len(tasks_df), index=tasks_df.index)

    for idx in tasks_df[mask].index:
        deadline = tasks_df.loc[idx, '截止时间']
        status = tasks_df.loc[idx, '任务状态编码']

        if pd.notna(deadline):
            if status in ['PENDING', 'ASSIGNED', 'IN_PROGRESS']:
                if now > deadline:
                    overdue_days = (now - deadline).days
                    tasks_df.loc[idx, '是否超期'] = '是'
                    tasks_df.loc[idx, '超期天数'] = overdue_days
                    if status != 'IN_PROGRESS':
                        tasks_df.loc[idx, '任务状态'] = '已超期'
                        tasks_df.loc[idx, '任务状态编码'] = 'OVERDUE'
            elif status == 'COMPLETED':
                completed_time = tasks_df.loc[idx, '处理完成时间']
                if pd.notna(completed_time) and completed_time > deadline:
                    overdue_days = (completed_time - deadline).days
                    tasks_df.loc[idx, '是否超期'] = '是'
                    tasks_df.loc[idx, '超期天数'] = overdue_days

    return tasks_df


def refresh_all_task_status(tasks_df):
    return _check_task_overdue(tasks_df)


def calculate_task_statistics(tasks_df, rules=None):
    if tasks_df is None or tasks_df.empty:
        return {}

    rules = get_effective_rules(rules)
    stats = {}

    total = len(tasks_df)
    stats['任务总数'] = total

    status_counts = tasks_df['任务状态'].value_counts().to_dict()
    for status in ['待派发', '已派发', '处理中', '已完成', '已超期', '已闭环']:
        stats[f'{status}任务数'] = status_counts.get(status, 0)

    stats['待处理任务数'] = stats.get('待派发任务数', 0) + stats.get('已派发任务数', 0) + stats.get('处理中任务数', 0)
    stats['已完成含超期'] = stats.get('已完成任务数', 0) + stats.get('已超期任务数', 0)

    if total > 0:
        stats['完成率'] = round(stats['已完成含超期'] / total, 4)
        stats['闭环完成率'] = round(stats.get('已闭环任务数', 0) / total, 4)
    else:
        stats['完成率'] = 0
        stats['闭环完成率'] = 0

    type_counts = tasks_df['任务类型'].value_counts().to_dict()
    for t in ['巡检', '复检', '清淤']:
        stats[f'{t}任务数'] = type_counts.get(t, 0)

    priority_counts = tasks_df['动态优先级'].value_counts().to_dict()
    for p in ['紧急', '高', '中', '低']:
        stats[f'{p}优先级任务数'] = priority_counts.get(p, 0)

    district_stats = {}
    for district in tasks_df['片区'].unique():
        dist_tasks = tasks_df[tasks_df['片区'] == district]
        dist_total = len(dist_tasks)
        dist_closed = len(dist_tasks[dist_tasks['任务状态'] == '已闭环'])
        dist_completed = len(dist_tasks[dist_tasks['任务状态'].isin(['已完成', '已超期', '已闭环'])])
        district_stats[district] = {
            '任务数': dist_total,
            '已完成数': dist_completed,
            '已闭环数': dist_closed,
            '完成率': round(dist_completed / dist_total, 4) if dist_total > 0 else 0,
            '闭环率': round(dist_closed / dist_total, 4) if dist_total > 0 else 0
        }
    stats['分片区统计'] = district_stats

    dredging_tasks = tasks_df[tasks_df['任务类型编码'] == 'DREDGING']
    if not dredging_tasks.empty:
        effect_counts = dredging_tasks['整改效果评级'].value_counts().to_dict()
        stats['清淤效果统计'] = effect_counts
        total_dredge = len(dredging_tasks[dredging_tasks['整改效果评级'] != ''])
        effective = effect_counts.get('显著有效', 0) + effect_counts.get('部分有效', 0)
        stats['清淤有效率'] = round(effective / total_dredge, 4) if total_dredge > 0 else 0
    else:
        stats['清淤效果统计'] = {}
        stats['清淤有效率'] = 0

    overdue_tasks = tasks_df[tasks_df['是否超期'] == '是']
    stats['超期任务总数'] = len(overdue_tasks)
    if total > 0:
        stats['超期率'] = round(len(overdue_tasks) / total, 4)
    else:
        stats['超期率'] = 0

    if not overdue_tasks.empty:
        stats['平均超期天数'] = round(overdue_tasks['超期天数'].mean(), 1)
    else:
        stats['平均超期天数'] = 0

    return stats


def compare_before_after(tasks_df, task_id):
    if tasks_df is None or tasks_df.empty or not task_id:
        return None

    task = tasks_df[tasks_df['任务编号'] == task_id]
    if task.empty:
        return None

    task = task.iloc[0]
    pre_rate = task.get('整改前淤积率')
    post_rate = task.get('处理后淤积率')
    pre_depth = task.get('最新淤积深度(mm)')
    post_depth = task.get('处理后淤积深度(mm)')

    if pre_rate is None or post_rate is None:
        return None

    rate_reduction = pre_rate - post_rate
    reduction_pct = rate_reduction / pre_rate if pre_rate > 0 else 0
    depth_reduction = (pre_depth - post_depth) if (pre_depth is not None and post_depth is not None) else None

    threshold = 0.3
    if reduction_pct >= threshold:
        effect_level = '显著有效'
        effect_color = '#27ae60'
    elif reduction_pct >= 0.1:
        effect_level = '部分有效'
        effect_color = '#f39c12'
    elif reduction_pct >= 0:
        effect_level = '效果不明显'
        effect_color = '#e67e22'
    else:
        effect_level = '淤积加重'
        effect_color = '#e74c3c'

    return {
        '任务编号': task_id,
        '管段编号': task.get('管段编号'),
        '片区': task.get('片区'),
        '任务类型': task.get('任务类型'),
        '清淤前淤积率': round(pre_rate, 4) if pre_rate is not None else None,
        '清淤后淤积率': round(post_rate, 4) if post_rate is not None else None,
        '淤积率变化量': round(rate_reduction, 4),
        '淤积率变化百分比': round(reduction_pct, 4),
        '清淤前淤积深度': pre_depth,
        '清淤后淤积深度': post_depth,
        '深度变化量': depth_reduction,
        '管径(mm)': task.get('管径(mm)'),
        '效果评级': effect_level,
        '效果颜色': effect_color,
        '触发原因': task.get('触发原因'),
        '处理结果': task.get('处理结果'),
        '处理备注': task.get('处理备注')
    }
