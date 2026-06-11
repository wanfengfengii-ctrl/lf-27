import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from data_processor import (
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    get_pipe_history,
    get_batch_comparison_data,
    get_risk_heatmap_data,
    detect_missing_inspections,
    get_batches,
    get_effective_rules
)

COLOR_SCALE_RISK = [
    [0.0, '#2ecc71'],
    [0.3, '#f1c40f'],
    [0.6, '#e67e22'],
    [1.0, '#e74c3c']
]


def create_pipe_history_chart(df, pipe_id, district=None, rules=None):
    rules = get_effective_rules(rules)
    history = get_pipe_history(df, pipe_id, district)

    if history.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f'管段 {pipe_id} 无数据',
            template='plotly_white',
            height=600
        )
        return fig

    all_batches = get_batches(df, district)
    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('淤积深度变化 (mm)', '淤积率变化 (%)'),
        row_heights=[0.55, 0.45]
    )

    fig.add_trace(
        go.Scatter(
            x=history['巡检批次'],
            y=history['淤积深度'],
            mode='lines+markers+text',
            name='淤积深度',
            line=dict(color='#3498db', width=3),
            marker=dict(size=12, line=dict(width=2, color='#2980b9')),
            text=history.apply(lambda r: f"{r['淤积深度']:.1f}mm", axis=1),
            textposition='top center',
            hovertemplate=(
                '<b>批次</b>: %{x}<br>'
                '<b>淤积深度</b>: %{y:.1f} mm<br>'
                '<b>管径</b>: %{customdata[0]:.0f} mm<br>'
                '<b>淤积率</b>: %{customdata[1]:.1%}<br>'
                '<b>检查时间</b>: %{customdata[2]}<br>'
                '<b>备注</b>: %{customdata[3]}'
            ),
            customdata=history[['管径', '淤积率', '检查时间', '备注']].values
        ),
        row=1, col=1
    )

    fig.add_hline(
        y=history['管径'].iloc[0] if len(history) > 0 else 0,
        line_dash='dash',
        line_color='#e74c3c',
        annotation_text=f"管径上限: {history['管径'].iloc[0]:.0f}mm" if len(history) > 0 else '',
        annotation_position='top right',
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=history['巡检批次'],
            y=history['淤积率'] * 100,
            mode='lines+markers+text',
            name='淤积率',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=12, line=dict(width=2, color='#8e44ad')),
            text=history.apply(lambda r: f"{r['淤积率']*100:.1f}%", axis=1),
            textposition='top center',
            hovertemplate=(
                '<b>批次</b>: %{x}<br>'
                '<b>淤积率</b>: %{y:.1f}%'
            )
        ),
        row=2, col=1
    )

    fig.add_hline(
        y=threshold_high * 100,
        line_dash='dash',
        line_color='#e74c3c',
        annotation_text=f'高风险线 ({threshold_high*100:.0f}%)',
        annotation_position='top right',
        row=2, col=1
    )
    fig.add_hline(
        y=threshold_medium * 100,
        line_dash='dash',
        line_color='#f39c12',
        annotation_text=f'中风险线 ({threshold_medium*100:.0f}%)',
        annotation_position='bottom right',
        row=2, col=1
    )

    if all_batches and len(history) < len(all_batches):
        existing_batches = set(history['巡检批次'].tolist())
        missing_batches = [b for b in all_batches if b not in existing_batches]

        if missing_batches:
            y_depth_min = history['淤积深度'].min() if len(history) > 0 else 0
            y_depth_marker = max(0, y_depth_min * 0.1)

            y_rate_min = history['淤积率'].min() * 100 if len(history) > 0 else 0
            y_rate_marker = max(0, y_rate_min * 0.1)

            fig.add_trace(
                go.Scatter(
                    x=missing_batches,
                    y=[y_depth_marker] * len(missing_batches),
                    mode='markers+text',
                    name='缺失巡检',
                    marker=dict(
                        symbol='x-thin',
                        size=18,
                        color='#e74c3c',
                        line=dict(width=3, color='#c0392b')
                    ),
                    text=['⚠️缺失'] * len(missing_batches),
                    textposition='top center',
                    textfont=dict(color='#c0392b', size=11, family='Arial, sans-serif'),
                    hovertemplate='<b>批次</b>: %{x}<br><b>状态</b>: 缺失巡检记录<extra></extra>'
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=missing_batches,
                    y=[y_rate_marker] * len(missing_batches),
                    mode='markers+text',
                    name='缺失巡检',
                    showlegend=False,
                    marker=dict(
                        symbol='x-thin',
                        size=18,
                        color='#e74c3c',
                        line=dict(width=3, color='#c0392b')
                    ),
                    text=['⚠️缺失'] * len(missing_batches),
                    textposition='top center',
                    textfont=dict(color='#c0392b', size=11, family='Arial, sans-serif'),
                    hovertemplate='<b>批次</b>: %{x}<br><b>状态</b>: 缺失巡检记录<extra></extra>'
                ),
                row=2, col=1
            )

    fig.update_layout(
        title=dict(
            text=f'管段淤积过程分析 - {pipe_id}' + (f'（片区：{district}）' if district else ''),
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        template='plotly_white',
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        showlegend=True
    )

    fig.update_yaxes(title_text='淤积深度 (mm)', row=1, col=1)
    fig.update_yaxes(title_text='淤积率 (%)', row=2, col=1)
    fig.update_xaxes(title_text='巡检批次', row=2, col=1)

    return fig


def create_pipes_comparison_chart(df, pipe_ids, district=None, compare_by='淤积率', rules=None):
    rules = get_effective_rules(rules)
    comparison_data = get_batch_comparison_data(df, pipe_ids, district)

    if comparison_data.empty or not pipe_ids:
        fig = go.Figure()
        fig.update_layout(
            title='无对比数据',
            template='plotly_white',
            height=450
        )
        return fig

    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))

    fig = go.Figure()

    colors = px.colors.qualitative.Plotly
    for i, pipe_id in enumerate(pipe_ids):
        pipe_data = comparison_data[comparison_data['管段编号'] == pipe_id]

        y_values = pipe_data['淤积率'] * 100 if compare_by == '淤积率' else pipe_data['淤积深度']
        y_unit = '%' if compare_by == '淤积率' else ' mm'

        fig.add_trace(
            go.Scatter(
                x=pipe_data['巡检批次'],
                y=y_values,
                mode='lines+markers',
                name=f'管段 {pipe_id}',
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=10),
                hovertemplate=(
                    f'<b>管段</b>: {pipe_id}<br>'
                    '<b>批次</b>: %{x}<br>'
                    f'<b>{compare_by}</b>: %{{y:.1f}}{y_unit}<br>'
                    '<b>管径</b>: %{customdata[0]:.0f}mm<br>'
                    '<b>淤积率</b>: %{customdata[1]:.1%}<br>'
                    '<b>检查时间</b>: %{customdata[2]}'
                ),
                customdata=pipe_data[['管径', '淤积率', '检查时间']].values
            )
        )

    district_label = f' - 片区：{district}' if district else ' - 全部片区'
    fig.update_layout(
        title=dict(
            text=f'多管段淤积对比分析（按{compare_by}）' + district_label,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        template='plotly_white',
        height=500,
        xaxis_title='巡检批次',
        yaxis_title=f'{compare_by} ({y_unit})',
        hovermode='x unified',
        legend=dict(
            title='管段编号',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    if compare_by == '淤积率':
        fig.add_hline(
            y=threshold_high * 100,
            line_dash='dash',
            line_color='#e74c3c',
            annotation_text=f'高风险线',
            annotation_position='top right'
        )
        fig.add_hline(
            y=threshold_medium * 100,
            line_dash='dash',
            line_color='#f39c12',
            annotation_text=f'中风险线',
            annotation_position='bottom right'
        )

    return fig


def create_risk_heatmap(df, district=None, batches=None, rules=None):
    rules = get_effective_rules(rules)
    heatmap_data = get_risk_heatmap_data(df, district, batches)

    if heatmap_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title='无热力图数据',
            template='plotly_white',
            height=500
        )
        return fig

    display_data = heatmap_data.copy()
    display_data = display_data * 100

    text_data = display_data.map(lambda x: '缺失' if x < 0 else f'{x:.1f}%')

    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))

    colorscale = [
        [0.0, '#d5dbdb'],
        [0.0001, '#2ecc71'],
        [threshold_medium, '#f1c40f'],
        [threshold_high, '#e67e22'],
        [1.0, '#e74c3c']
    ]

    display_data_for_plot = display_data.copy()
    display_data_for_plot[display_data_for_plot < 0] = -5

    fig = go.Figure(
        data=go.Heatmap(
            z=display_data_for_plot.values,
            x=list(display_data.columns),
            y=list(display_data.index),
            text=text_data.values,
            texttemplate='%{text}',
            colorscale=colorscale,
            zmin=-5,
            zmax=100,
            colorbar=dict(
                title=dict(text='淤积率 (%)', side='right'),
                tickvals=[0, threshold_medium * 100, threshold_high * 100, 100],
                ticktext=[f'0%', f'{threshold_medium*100:.0f}%', f'{threshold_high*100:.0f}%', '100%'],
                lenmode='pixels',
                len=300
            ),
            hovertemplate=(
                '<b>管段</b>: %{y}<br>'
                '<b>批次</b>: %{x}<br>'
                '<b>淤积率</b>: %{text}<extra></extra>'
            )
        )
    )

    fig.update_layout(
        title=dict(
            text='风险分布热力图' + (f' - 片区：{district}' if district else ''),
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        template='plotly_white',
        height=max(400, 50 + 30 * len(heatmap_data.index)),
        xaxis=dict(
            title='巡检批次',
            side='top'
        ),
        yaxis=dict(
            title='管段编号',
            autorange='reversed'
        )
    )

    return fig


def create_risk_distribution_chart(stats, rules=None):
    rules = get_effective_rules(rules)
    if not stats:
        fig = go.Figure()
        fig.update_layout(
            title='无统计数据',
            template='plotly_white',
            height=350
        )
        return fig

    labels = ['高风险', '中风险', '低风险']
    values = [stats.get('高风险管段数', 0), stats.get('中风险管段数', 0), stats.get('低风险管段数', 0)]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'bar'}]],
        subplot_titles=('风险等级分布（管段数）', '各风险等级管段数统计')
    )

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent+value',
            hole=0.4,
            hovertemplate='<b>%{label}</b><br>管段数: %{value}<br>占比: %{percent}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>管段数: %{y}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text='管段风险分布总览',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def create_sediment_trend_chart(df, district=None, batches=None, rules=None):
    rules = get_effective_rules(rules)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title='无趋势数据',
            template='plotly_white',
            height=400
        )
        return fig

    filtered = df.copy()
    if district:
        filtered = filtered[filtered['片区'] == district]
    if batches:
        filtered = filtered[filtered['巡检批次'].isin(batches)]

    if filtered.empty:
        fig = go.Figure()
        fig.update_layout(title='无数据', template='plotly_white', height=400)
        return fig

    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', RISK_THRESHOLD_HIGH))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', RISK_THRESHOLD_MEDIUM))

    batch_stats = filtered.groupby('巡检批次').agg(
        平均淤积率=('淤积率', 'mean'),
        最大淤积率=('淤积率', 'max'),
        中位数淤积率=('淤积率', 'median'),
        管段数=('管段编号', 'nunique')
    ).reset_index()

    batch_stats = batch_stats.sort_values('巡检批次')

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=batch_stats['巡检批次'],
            y=batch_stats['平均淤积率'] * 100,
            mode='lines+markers+text',
            name='平均淤积率',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10),
            text=batch_stats.apply(lambda r: f"{r['平均淤积率']*100:.1f}%", axis=1),
            textposition='top center'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=batch_stats['巡检批次'],
            y=batch_stats['最大淤积率'] * 100,
            mode='lines+markers+text',
            name='最大淤积率',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            marker=dict(size=8),
            text=batch_stats.apply(lambda r: f"{r['最大淤积率']*100:.1f}%", axis=1),
            textposition='top center'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=batch_stats['巡检批次'],
            y=batch_stats['中位数淤积率'] * 100,
            mode='lines+markers+text',
            name='中位数淤积率',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=8),
            text=batch_stats.apply(lambda r: f"{r['中位数淤积率']*100:.1f}%", axis=1),
            textposition='bottom center'
        )
    )

    fig.update_layout(
        title=dict(
            text='区域淤积趋势分析' + (f' - 片区：{district}' if district else ''),
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        template='plotly_white',
        height=450,
        xaxis_title='巡检批次',
        yaxis_title='淤积率 (%)',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    fig.add_hline(
        y=threshold_high * 100,
        line_dash='dash',
        line_color='#e74c3c',
        opacity=0.5,
        annotation_text='高风险线',
        annotation_position='top right'
    )
    fig.add_hline(
        y=threshold_medium * 100,
        line_dash='dash',
        line_color='#f39c12',
        opacity=0.5,
        annotation_text='中风险线',
        annotation_position='bottom right'
    )

    return fig


def create_dredging_effect_chart(dredging_results, rules=None):
    if dredging_results is None or (isinstance(dredging_results, pd.DataFrame) and dredging_results.empty):
        fig = go.Figure()
        fig.update_layout(
            title='无清淤效果评估数据',
            template='plotly_white',
            height=500
        )
        return fig

    if isinstance(dredging_results, dict):
        dredging_results = pd.DataFrame([dredging_results])

    if dredging_results.empty:
        fig = go.Figure()
        fig.update_layout(title='无清淤效果数据', template='plotly_white', height=500)
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'domain'}]],
        subplot_titles=('清淤前后淤积率对比', '清淤效果分布')
    )

    fig.add_trace(
        go.Bar(
            x=dredging_results['管段编号'],
            y=dredging_results['清淤前淤积率'] * 100,
            name='清淤前',
            marker_color='#e74c3c',
            text=dredging_results['清淤前淤积率'].apply(lambda x: f'{x*100:.1f}%'),
            textposition='auto'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=dredging_results['管段编号'],
            y=dredging_results['清淤后淤积率'] * 100,
            name='清淤后',
            marker_color='#27ae60',
            text=dredging_results['清淤后淤积率'].apply(lambda x: f'{x*100:.1f}%'),
            textposition='auto'
        ),
        row=1, col=1
    )

    effect_counts = dredging_results['效果评级'].value_counts()
    effect_colors = {
        '显著有效': '#27ae60',
        '部分有效': '#f39c12',
        '效果不明显': '#e67e22',
        '淤积加重': '#e74c3c'
    }

    fig.add_trace(
        go.Pie(
            labels=effect_counts.index.tolist(),
            values=effect_counts.values.tolist(),
            marker_colors=[effect_colors.get(l, '#95a5a6') for l in effect_counts.index],
            textinfo='label+percent+value',
            hole=0.4
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text='清淤前后效果评估',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        template='plotly_white',
        height=500,
        showlegend=True,
        barmode='group'
    )

    fig.update_yaxes(title_text='淤积率 (%)', row=1, col=1)
    fig.update_xaxes(title_text='管段编号', row=1, col=1)

    return fig


def create_district_comparison_chart(district_summary_df, rules=None):
    if district_summary_df is None or district_summary_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title='无片区对比数据',
            template='plotly_white',
            height=500
        )
        return fig

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('各片区平均淤积率', '各片区高风险管段数', '各片区异常增长数', '各片区缺失巡检数'),
        vertical_spacing=0.12
    )

    colors = px.colors.qualitative.Set2

    fig.add_trace(
        go.Bar(
            x=district_summary_df['片区'],
            y=district_summary_df['平均淤积率'] * 100,
            marker_color=colors[0],
            text=district_summary_df['平均淤积率'].apply(lambda x: f'{x*100:.1f}%'),
            textposition='auto',
            showlegend=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=district_summary_df['片区'],
            y=district_summary_df['高风险管段数'],
            marker_color='#e74c3c',
            text=district_summary_df['高风险管段数'],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(
            x=district_summary_df['片区'],
            y=district_summary_df['异常增长数'],
            marker_color='#e67e22',
            text=district_summary_df['异常增长数'],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=district_summary_df['片区'],
            y=district_summary_df['缺失巡检数'],
            marker_color='#7f8c8d',
            text=district_summary_df['缺失巡检数'],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=2
    )

    fig.update_yaxes(title_text='淤积率 (%)', row=1, col=1)
    fig.update_yaxes(title_text='管段数', row=1, col=2)
    fig.update_yaxes(title_text='管段数', row=2, col=1)
    fig.update_yaxes(title_text='缺失数', row=2, col=2)

    fig.update_layout(
        title=dict(
            text='跨片区隔离分析（各片区独立统计，禁止合并对比）',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        template='plotly_white',
        height=700,
        showlegend=False
    )

    return fig


def create_priority_dashboard_chart(priority_df, rules=None):
    if priority_df is None or priority_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title='无优先级数据',
            template='plotly_white',
            height=500
        )
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'xy'}]],
        subplot_titles=('清淤优先级分布', '管段风险评分排名（Top 15）')
    )

    priority_counts = priority_df['清淤优先级'].value_counts()
    priority_colors = {
        '紧急': '#c0392b',
        '高': '#e74c3c',
        '中': '#f39c12',
        '低': '#27ae60'
    }

    fig.add_trace(
        go.Pie(
            labels=priority_counts.index.tolist(),
            values=priority_counts.values.tolist(),
            marker_colors=[priority_colors.get(l, '#95a5a6') for l in priority_counts.index],
            textinfo='label+percent+value',
            hole=0.4
        ),
        row=1, col=1
    )

    top_pipes = priority_df.head(15)
    fig.add_trace(
        go.Bar(
            x=top_pipes['管段编号'],
            y=top_pipes['风险评分'],
            marker_color=[priority_colors.get(p, '#95a5a6') for p in top_pipes['清淤优先级']],
            text=top_pipes['风险评分'].apply(lambda x: f'{x:.0f}'),
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text='清淤优先级排序看板',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        template='plotly_white',
        height=500,
        showlegend=False
    )

    fig.update_xaxes(title_text='管段编号', row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text='风险评分', row=1, col=2)

    return fig
