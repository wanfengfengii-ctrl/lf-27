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

RISK_THRESHOLD_HIGH = 0.6
RISK_THRESHOLD_MEDIUM = 0.3
ABNORMAL_GROWTH_RATE = 0.2


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


def validate_and_clean_data(df):
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


def calculate_statistics(df, district=None, batches=None):
    if df.empty:
        return {}
    
    filtered = df.copy()
    if district:
        filtered = filtered[filtered['片区'] == district]
    if batches:
        filtered = filtered[filtered['巡检批次'].isin(batches)]
    
    if filtered.empty:
        return {}
    
    stats = {}
    stats['记录总数'] = len(filtered)
    stats['管段数量'] = filtered['管段编号'].nunique()
    stats['巡检批次数量'] = filtered['巡检批次'].nunique()
    
    stats['最大淤积深度'] = round(filtered['淤积深度'].max(), 2)
    stats['平均淤积深度'] = round(filtered['淤积深度'].mean(), 2)
    stats['最大淤积率'] = round(filtered['淤积率'].max(), 4)
    stats['平均淤积率'] = round(filtered['淤积率'].mean(), 4)
    
    high_risk = filtered[filtered['淤积率'] >= RISK_THRESHOLD_HIGH]
    medium_risk = filtered[(filtered['淤积率'] >= RISK_THRESHOLD_MEDIUM) & (filtered['淤积率'] < RISK_THRESHOLD_HIGH)]
    low_risk = filtered[filtered['淤积率'] < RISK_THRESHOLD_MEDIUM]
    
    stats['高风险管段数'] = high_risk['管段编号'].nunique()
    stats['中风险管段数'] = medium_risk['管段编号'].nunique()
    stats['低风险管段数'] = low_risk['管段编号'].nunique()
    
    stats['高风险记录数'] = len(high_risk)
    stats['中风险记录数'] = len(medium_risk)
    stats['低风险记录数'] = len(low_risk)
    
    return stats


def detect_abnormal_growth(df, district=None, batches=None):
    if df.empty:
        return pd.DataFrame()
    
    filtered = df.copy()
    if district:
        filtered = filtered[filtered['片区'] == district]
    
    if filtered.empty:
        return pd.DataFrame()
    
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
                
                if growth_rate >= ABNORMAL_GROWTH_RATE and curr_rate > RISK_THRESHOLD_MEDIUM:
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


def get_high_risk_segments(df, district=None, batches=None):
    if df.empty:
        return pd.DataFrame()
    
    filtered = df.copy()
    if district:
        filtered = filtered[filtered['片区'] == district]
    if batches:
        filtered = filtered[filtered['巡检批次'].isin(batches)]
    
    if filtered.empty:
        return pd.DataFrame()
    
    high_risk = filtered[filtered['淤积率'] >= RISK_THRESHOLD_HIGH].copy()
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


def revalidate_dataframe(df):
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
