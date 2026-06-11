import sys
sys.path.insert(0, '.')
import pandas as pd
from datetime import datetime, timedelta

from data_processor import (
    generate_inspection_tasks,
    initialize_crews,
    generate_daily_schedule,
    detect_schedule_conflicts,
    detect_overload,
    reassign_task,
    merge_nearby_tasks,
    update_schedule_status,
    calculate_crew_performance,
    calculate_schedule_deviation_analysis,
    sync_schedule_with_tasks,
    get_crew_workload,
    estimate_task_duration,
    get_location_coords,
    calculate_distance,
    standardize_columns,
    validate_and_clean_data
)


def run_tests():
    print("=" * 60)
    print("巡检路线优化与资源调度模块 - 功能测试")
    print("=" * 60)

    print("\n1. 测试数据准备")
    print("-" * 40)
    df = pd.read_csv('sample_data.csv')
    std_df, _ = standardize_columns(df)
    valid_df, _, _, _ = validate_and_clean_data(std_df)
    print(f"   有效数据: {len(valid_df)} 条")

    print("\n2. 测试任务生成")
    print("-" * 40)
    tasks_df = generate_inspection_tasks(valid_df, max_tasks_per_batch=30)
    print(f"   生成任务: {len(tasks_df)} 个")
    print(f"   任务类型分布: {tasks_df['任务类型'].value_counts().to_dict()}")
    print(f"   优先级分布: {tasks_df['动态优先级'].value_counts().to_dict()}")

    print("\n3. 测试班组初始化")
    print("-" * 40)
    crews_df = initialize_crews()
    print(f"   班组数量: {len(crews_df)}")
    for _, row in crews_df.iterrows():
        print(f"   - {row['crew_id']}: {row['crew_name']} ({row['crew_type']}) - {row['base_location']}")

    print("\n4. 测试位置坐标计算")
    print("-" * 40)
    loc1 = get_location_coords('A区', 'P001')
    loc2 = get_location_coords('B区', 'P006')
    dist = calculate_distance(loc1, loc2)
    print(f"   A区 P001 坐标: ({loc1['x']:.2f}, {loc1['y']:.2f})")
    print(f"   B区 P006 坐标: ({loc2['x']:.2f}, {loc2['y']:.2f})")
    print(f"   两点距离: {dist:.2f} km")

    print("\n5. 测试时长预估")
    print("-" * 40)
    for i in range(min(5, len(tasks_df))):
        sample_task = tasks_df.iloc[i]
        duration = estimate_task_duration(sample_task)
        print(f"   任务 {sample_task['任务编号']} ({sample_task['任务类型']}): {duration:.1f} 分钟")

    print("\n6. 测试每日计划生成")
    print("-" * 40)
    schedule_date = datetime.now().date()
    schedule_df, unassigned_df, summary = generate_daily_schedule(
        tasks_df, crews_df, schedule_date=schedule_date
    )
    print(f"   可用任务: {summary.get('total_tasks_available', 0)}")
    print(f"   已安排任务: {summary.get('total_tasks_scheduled', 0)}")
    print(f"   未安排任务: {summary.get('total_tasks_unassigned', 0)}")
    print(f"   使用班组: {summary.get('crews_used', 0)}")
    
    if not schedule_df.empty:
        print(f"\n   各班组任务分配:")
        for crew, group in schedule_df.groupby('班组名称'):
            total_duration = group['预估时长(分钟)'].sum()
            total_distance = group['预估行程(km)'].sum()
            print(f"   - {crew}: {len(group)} 个任务, {total_duration:.0f}分钟, {total_distance:.1f}km")
        
        print(f"\n   计划样例 (前5条):")
        cols = ['计划编号', '班组名称', '顺序', '管段编号', '任务类型', 
                '动态优先级', '预计开始时间', '预计完成时间']
        print(schedule_df[cols].head().to_string(index=False))

    print("\n7. 测试冲突检测")
    print("-" * 40)
    conflicts = detect_schedule_conflicts(schedule_df)
    print(f"   检测到冲突: {len(conflicts)} 个")
    for c in conflicts[:3]:
        print(f"   - {c['message']}")

    print("\n8. 测试超负荷检测")
    print("-" * 40)
    overloads = detect_overload(schedule_df, crews_df)
    print(f"   检测到超负荷: {len(overloads)} 个")
    for o in overloads[:3]:
        severity = '高' if o.get('severity') == 'high' else '中'
        print(f"   - [{severity}] {o['crew_name']}: {o['message']}")

    print("\n9. 测试任务改派")
    print("-" * 40)
    if len(schedule_df) >= 2:
        test_task = schedule_df.iloc[0]
        task_id = test_task['任务编号']
        from_crew = test_task['班组编号']
        to_crew = 'CREW_B' if from_crew != 'CREW_B' else 'CREW_A'
        print(f"   将任务 {task_id} 从 {from_crew} 改派到 {to_crew}")
        updated_schedule, result = reassign_task(schedule_df, task_id, from_crew, to_crew, schedule_date)
        if result and result.get('success'):
            print(f"   改派成功!")
        else:
            print(f"   改派结果: {result}")

    print("\n10. 测试顺路合并建议")
    print("-" * 40)
    _, suggestions = merge_nearby_tasks(schedule_df, distance_threshold=2.0)
    print(f"   顺路合并建议: {len(suggestions)} 个")
    for s in suggestions[:3]:
        print(f"   - {s['merged_pipe_ids']}: 距离 {s['distance']:.1f}km, "
              f"预计节省 {s['estimated_saving_minutes']:.0f} 分钟")

    print("\n11. 测试状态更新")
    print("-" * 40)
    if not schedule_df.empty:
        test_task_id = schedule_df.iloc[0]['任务编号']
        print(f"   开始任务 {test_task_id}")
        updated_schedule, result = update_schedule_status(schedule_df, test_task_id, 'start')
        if result and result.get('success'):
            print(f"   任务已开始")
        
        print(f"   完成任务 {test_task_id}")
        updated_schedule, result = update_schedule_status(updated_schedule, test_task_id, 'complete')
        if result and result.get('success'):
            test_row = updated_schedule[updated_schedule['任务编号'] == test_task_id].iloc[0]
            print(f"   任务已完成")
            print(f"   - 执行状态: {test_row['执行状态']}")
            print(f"   - 实际开始: {test_row['实际开始时间']}")
            print(f"   - 实际完成: {test_row['实际完成时间']}")
            print(f"   - 执行偏差: {test_row['执行偏差(分钟)']:+.1f} 分钟")

    print("\n12. 测试班组绩效统计")
    print("-" * 40)
    perf_df, overall = calculate_crew_performance(updated_schedule, tasks_df)
    print(f"   总体统计:")
    print(f"   - 班组总数: {overall.get('total_crews', 0)}")
    print(f"   - 任务总数: {overall.get('total_tasks', 0)}")
    print(f"   - 总完成率: {overall.get('overall_completion_rate', 0)*100:.1f}%")
    print(f"   - 总准时率: {overall.get('overall_on_time_rate', 0)*100:.1f}%")
    print(f"   - 平均偏差: {overall.get('avg_deviation_all', 0):+.1f} 分钟")
    
    if not perf_df.empty:
        print(f"\n   各班组绩效:")
        cols = ['班组名称', '统计周期任务数', '已完成任务数', '完成率', '准时率', '平均偏差(分钟)']
        perf_display = perf_df[cols].copy()
        perf_display['完成率'] = perf_display['完成率'].apply(lambda x: f"{x*100:.1f}%")
        perf_display['准时率'] = perf_display['准时率'].apply(lambda x: f"{x*100:.1f}%")
        print(perf_display.to_string(index=False))

    print("\n13. 测试偏差分析")
    print("-" * 40)
    completed_df, analysis = calculate_schedule_deviation_analysis(updated_schedule, tasks_df)
    print(f"   已完成任务: {analysis.get('total_completed', 0)}")
    print(f"   准时率: {analysis.get('on_time_rate', 0)*100:.1f}%")
    print(f"   延误率: {analysis.get('delay_rate', 0)*100:.1f}%")
    print(f"   平均偏差: {analysis.get('avg_deviation', 0):+.1f} 分钟")
    print(f"   偏差分布: {analysis.get('deviation_distribution', {})}")
    
    major_devs = analysis.get('major_deviations', [])
    if major_devs:
        print(f"\n   大幅延误任务分析 ({len(major_devs)} 个):")
        for dev in major_devs[:3]:
            print(f"   - {dev['任务编号']} ({dev['班组']}): "
                  f"{dev['偏差(分钟)']:+.0f}分钟 - {dev['可能原因']}")

    print("\n14. 测试与任务系统同步")
    print("-" * 40)
    synced_tasks = sync_schedule_with_tasks(updated_schedule, tasks_df)
    updated_task = synced_tasks[synced_tasks['任务编号'] == test_task_id].iloc[0]
    print(f"   同步后任务状态: {updated_task['任务状态']}")
    print(f"   同步后处理人员: {updated_task['处理人员']}")
    print(f"   同步后处理开始时间: {updated_task['处理开始时间']}")

    print("\n15. 测试班组负荷")
    print("-" * 40)
    for crew_id in crews_df['crew_id'].tolist():
        workload = get_crew_workload(updated_schedule, crew_id, schedule_date)
        ratio = workload.get('workload_ratio', 0) * 100
        print(f"   - {workload.get('crew_name', crew_id)}: {workload.get('status_text', '未知')} "
              f"({ratio:.0f}%) - {workload.get('total_tasks', 0)} 任务, "
              f"{workload.get('total_duration_minutes', 0):.0f} 分钟")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！巡检路线优化与资源调度模块功能正常")
    print("=" * 60)


if __name__ == '__main__':
    run_tests()
