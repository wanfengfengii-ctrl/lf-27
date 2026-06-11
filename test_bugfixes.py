#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, date
from data_processor import (
    initialize_crews, generate_daily_schedule, update_schedule_status,
    sync_schedule_with_tasks, reassign_task, compare_before_after,
    generate_inspection_tasks, standardize_columns, validate_and_clean_data
)

print("=" * 60)
print("修复验证测试")
print("=" * 60)

# 1. 准备数据
print("\n1. 准备测试数据...")
data = pd.read_csv('sample_data.csv')
std_df, _ = standardize_columns(data)
valid_df, _, _, _ = validate_and_clean_data(std_df)
tasks_df = generate_inspection_tasks(valid_df, max_tasks_per_batch=30)
crews_df = initialize_crews()

print(f"   生成任务数: {len(tasks_df)}")
print(f"   班组数量: {len(crews_df)}")

# 2. 生成调度计划
print("\n2. 生成调度计划...")
schedule_df, unassigned_df, summary = generate_daily_schedule(
    tasks_df, crews_df, schedule_date=date.today()
)
print(f"   计划任务数: {len(schedule_df)}")
print(f"   班组分配: {schedule_df.groupby('班组名称').size().to_dict()}")

# 3. 测试任务改派 - 修复问题2
print("\n3. 测试任务改派（修复问题2：班组名称同步）...")
test_task = schedule_df.iloc[0]
task_id = test_task['任务编号']
from_crew = test_task['班组编号']
from_crew_name = test_task['班组名称']

print(f"   原任务: {task_id}, 原班组: {from_crew}({from_crew_name})")

to_crew = 'CREW_B'
to_crew_name_expected = None
if 'crew_id' in crews_df.columns:
    to_crew_row = crews_df[crews_df['crew_id'] == to_crew].iloc[0]
    to_crew_name_expected = to_crew_row['crew_name']
elif '班组编号' in crews_df.columns:
    to_crew_row = crews_df[crews_df['班组编号'] == to_crew].iloc[0]
    to_crew_name_expected = to_crew_row['班组名称']

print(f"   目标班组: {to_crew}({to_crew_name_expected})")

schedule_df_updated, result = reassign_task(
    schedule_df.copy(), task_id, from_crew, to_crew,
    schedule_date=date.today(), crews_df=crews_df
)

if result and result.get('success'):
    updated_task = schedule_df_updated[schedule_df_updated['任务编号'] == task_id].iloc[0]
    actual_crew_name = updated_task['班组名称']
    actual_crew_id = updated_task['班组编号']
    print(f"   改派后班组: {actual_crew_id}({actual_crew_name})")
    if actual_crew_name == to_crew_name_expected and actual_crew_id == to_crew:
        print("   ✅ 问题2修复成功：班组名称正确同步更新")
    else:
        print(f"   ❌ 问题2修复失败：期望 {to_crew_name_expected}，实际 {actual_crew_name}")
else:
    print(f"   ❌ 改派失败: {result}")

# 4. 测试整改前后对比图 - 修复问题1
print("\n4. 测试整改前后对比图（修复问题1：完成任务后数据同步）...")
schedule_df_for_test = schedule_df.head(3).copy()

for i, (idx, row) in enumerate(schedule_df_for_test.iterrows()):
    if i == 0:
        schedule_df_for_test.loc[idx, '执行状态'] = '已完成'
        schedule_df_for_test.loc[idx, '实际开始时间'] = datetime.now()
        schedule_df_for_test.loc[idx, '实际完成时间'] = datetime.now()

tasks_df_before = tasks_df.copy()
tasks_df_after = sync_schedule_with_tasks(schedule_df_for_test, tasks_df.copy())

completed_task_id = schedule_df_for_test.iloc[0]['任务编号']
task_before = tasks_df_before[tasks_df_before['任务编号'] == completed_task_id].iloc[0]
task_after = tasks_df_after[tasks_df_after['任务编号'] == completed_task_id].iloc[0]

print(f"   任务ID: {completed_task_id}")
print(f"   任务类型: {task_after.get('任务类型', '')}")
print(f"   任务状态: {task_after.get('任务状态', '')}")
print(f"   处理后淤积深度: {task_after.get('处理后淤积深度(mm)', 'None')}")
print(f"   处理后淤积率: {task_after.get('处理后淤积率', 'None')}")

comparison_data = compare_before_after(tasks_df_after, completed_task_id)

if comparison_data is not None:
    print(f"   对比数据: 清淤前={comparison_data.get('清淤前淤积率')}, 清淤后={comparison_data.get('清淤后淤积率')}")
    print(f"   效果评级: {comparison_data.get('效果评级', '')}")
    print("   ✅ 问题1修复成功：整改前后对比数据正确生成")
else:
    print("   ❌ 问题1修复失败：对比数据为None")

# 5. 运行原有测试确保无破坏
print("\n5. 运行原有测试确保无破坏...")
import subprocess
result = subprocess.run(
    ['python3', 'test_schedule_module.py'],
    capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
)

if "FAILED" in result.stdout or result.returncode != 0:
    print("   ⚠️  部分测试失败，请检查")
    print(result.stdout[-500:])
else:
    print("   ✅ 所有原有测试通过")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
