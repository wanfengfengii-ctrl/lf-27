import io
import base64
import pandas as pd
import numpy as np
from datetime import datetime

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from data_processor import (
    standardize_columns,
    validate_and_clean_data,
    get_districts,
    get_batches,
    get_pipe_ids,
    calculate_statistics,
    detect_abnormal_growth,
    get_high_risk_segments,
    detect_missing_inspections,
    revalidate_dataframe,
    validate_batch_data,
    validate_cross_district,
    evaluate_dredging_effect,
    batch_evaluate_dredging,
    calculate_dredging_priority,
    generate_inspection_quality_report,
    get_district_summary,
    get_effective_rules,
    DEFAULT_RISK_RULES,
    merge_repaired_batch,
    merge_edited_subset,
    generate_inspection_tasks,
    assign_task,
    batch_assign_tasks,
    start_task,
    complete_task,
    close_task,
    refresh_all_task_status,
    calculate_task_statistics,
    compare_before_after
)
from visualizations import (
    create_pipe_history_chart,
    create_pipes_comparison_chart,
    create_risk_heatmap,
    create_risk_distribution_chart,
    create_sediment_trend_chart,
    create_dredging_effect_chart,
    create_district_comparison_chart,
    create_priority_dashboard_chart,
    create_task_status_chart,
    create_task_completion_chart,
    create_before_after_comparison_chart,
    create_dredging_effect_summary_chart
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=True)
app.title = '城市雨水管网淤积巡检分析台（二期升级版）'
server = app.server

GLOBAL_DATA = {
    'raw_df': None,
    'valid_df': pd.DataFrame(),
    'errors': [],
    'warnings': [],
    'has_district': False,
    'custom_rules': None,
    'tasks_df': pd.DataFrame(),
    'selected_task_id': None
}

HEADER_STYLE = {
    'background': 'linear-gradient(135deg, #1a5276 0%, #2c3e50 50%, #154360 100%)',
    'padding': '20px 30px',
    'marginBottom': '25px',
    'borderRadius': '8px',
    'boxShadow': '0 4px 12px rgba(0,0,0,0.2)'
}

CARD_STYLE = {
    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
    'borderRadius': '8px',
    'marginBottom': '20px'
}


def parse_contents(contents, filename):
    if contents is None:
        return None, '未选择文件', []

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename.lower():
            raw_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xlsx' in filename.lower() or 'xls' in filename.lower():
            raw_df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, '不支持的文件格式，请上传 CSV 或 Excel 文件', []

        std_df, missing_cols = standardize_columns(raw_df)

        if missing_cols:
            return std_df, f'缺少必需列: {", ".join(missing_cols)}', []

        valid_df, errors, warnings, has_district = validate_and_clean_data(std_df)

        return valid_df, errors, warnings, has_district, std_df

    except Exception as e:
        return None, f'文件解析失败: {str(e)}', []


def make_empty_fig(title='暂无数据'):
    fig = go.Figure()
    fig.update_layout(title=title, template='plotly_white', height=400)
    return fig


def make_empty_msg(text='暂无数据'):
    return html.Div(text, style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '20px'})


def render_quality_report(report):
    if not report:
        return make_empty_msg('暂无质量报告数据')

    sections = []

    overview = report.get('总览', {})
    sections.append(html.Div([
        html.H6('总览', style={'color': '#2c3e50', 'fontWeight': 'bold', 'marginBottom': '8px'}),
        html.Ul([
            html.Li(f"记录总数: {overview.get('记录总数', '-')}"),
            html.Li(f"管段总数: {overview.get('管段总数', '-')}"),
            html.Li(f"巡检批次数: {overview.get('巡检批次数', '-')}"),
            html.Li(f"片区数: {overview.get('片区数', '-')}"),
        ], style={'marginBottom': '15px'})
    ]))

    coverage = report.get('巡检覆盖率', {})
    coverage_items = [
        html.Li(f"应检记录数: {coverage.get('应检记录数', '-')}"),
        html.Li(f"实检记录数: {coverage.get('实检记录数', '-')}"),
        html.Li(f"覆盖率: {coverage.get('覆盖率', '-')}"),
        html.Li(f"缺失巡检数: {coverage.get('缺失巡检数', '-')}"),
    ]
    if '按片区缺失' in coverage:
        parts = [f"{k}: {v}" for k, v in coverage['按片区缺失'].items()]
        coverage_items.append(html.Li(f"按片区缺失: {', '.join(parts)}"))
    if '按批次缺失' in coverage:
        parts = [f"{k}: {v}" for k, v in coverage['按批次缺失'].items()]
        coverage_items.append(html.Li(f"按批次缺失: {', '.join(parts)}"))
    sections.append(html.Div([
        html.H6('巡检覆盖率', style={'color': '#2980b9', 'fontWeight': 'bold', 'marginBottom': '8px'}),
        html.Ul(coverage_items, style={'marginBottom': '15px'})
    ]))

    abnormal = report.get('异常增长统计', {})
    abnormal_items = [
        html.Li(f"异常增长管段数: {abnormal.get('异常增长管段数', '-')}"),
        html.Li(f"异常增长记录数: {abnormal.get('异常增长记录数', '-')}"),
    ]
    if '按片区统计' in abnormal:
        parts = [f"{k}: {v}" for k, v in abnormal['按片区统计'].items()]
        abnormal_items.append(html.Li(f"按片区统计: {', '.join(parts)}"))
    sections.append(html.Div([
        html.H6('异常增长统计', style={'color': '#e67e22', 'fontWeight': 'bold', 'marginBottom': '8px'}),
        html.Ul(abnormal_items, style={'marginBottom': '15px'})
    ]))

    high_risk = report.get('高风险统计', {})
    high_risk_items = [
        html.Li(f"高风险管段数: {high_risk.get('高风险管段数', '-')}"),
        html.Li(f"高风险记录数: {high_risk.get('高风险记录数', '-')}"),
    ]
    if '按片区统计' in high_risk:
        parts = [f"{k}: {v}" for k, v in high_risk['按片区统计'].items()]
        high_risk_items.append(html.Li(f"按片区统计: {', '.join(parts)}"))
    sections.append(html.Div([
        html.H6('高风险统计', style={'color': '#c0392b', 'fontWeight': 'bold', 'marginBottom': '8px'}),
        html.Ul(high_risk_items, style={'marginBottom': '15px'})
    ]))

    quality = report.get('数据质量', {})
    sections.append(html.Div([
        html.H6('数据质量', style={'color': '#8e44ad', 'fontWeight': 'bold', 'marginBottom': '8px'}),
        html.Ul([
            html.Li(f"淤积深度为负: {quality.get('淤积深度为负', '-')}"),
            html.Li(f"淤积率超100%: {quality.get('淤积率超100%', '-')}"),
            html.Li(f"重复记录: {quality.get('重复记录', '-')}"),
        ], style={'marginBottom': '15px'})
    ]))

    conclusion = report.get('质量评估结论', [])
    conclusion_color = '#27ae60' if any('良好' in c for c in conclusion) else '#c0392b'
    sections.append(html.Div([
        html.H6('质量评估结论', style={'color': '#2c3e50', 'fontWeight': 'bold', 'marginBottom': '8px'}),
        html.Ul([html.Li(c, style={'color': conclusion_color}) for c in conclusion],
                style={'marginBottom': '15px'})
    ]))

    return html.Div(sections, style={'padding': '15px', 'background': '#fafafa', 'borderRadius': '8px'})


app.layout = dbc.Container([
    html.Div([
        html.H1('🌧️ 城市雨水管网淤积巡检分析台（二期升级版）',
                 style={'color': 'white', 'margin': 0, 'fontSize': '26px'}),
        html.P('市政巡检数据管理与淤积风险分析系统',
               style={'color': '#bdc3c7', 'margin': '8px 0 0 0', 'fontSize': '14px'})
    ], style=HEADER_STYLE),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('数据导入',
                               style={'fontWeight': 'bold', 'fontSize': '16px', 'background': '#ecf0f1'}),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.Div('拖拽文件到此 或 ', style={'fontSize': '16px'}),
                            html.A('点击选择文件',
                                   style={'color': '#3498db', 'textDecoration': 'underline',
                                          'cursor': 'pointer'}),
                            html.Br(),
                            html.Small('支持 CSV、Excel 格式', style={'color': '#7f8c8d'})
                        ], style={'textAlign': 'center', 'padding': '30px 20px'}),
                        style={
                            'width': '100%', 'borderWidth': '2px', 'borderStyle': 'dashed',
                            'borderRadius': '8px', 'borderColor': '#bdc3c7',
                            'backgroundColor': '#fafafa', 'cursor': 'pointer'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-status', style={'marginTop': '15px'}),
                    html.Div(id='file-info', style={'marginTop': '10px', 'fontSize': '13px',
                                                    'color': '#7f8c8d'})
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('导入问题报告',
                               style={'fontWeight': 'bold', 'fontSize': '16px', 'background': '#fdf2e9'}),
                dbc.CardBody([
                    html.Div(id='import-report', children=[
                        html.Div('尚未导入数据',
                                 style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '30px'})
                    ])
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ]),

    html.Div(id='main-content', style={'display': 'none'}, children=[
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('筛选条件',
                                   style={'fontWeight': 'bold', 'fontSize': '16px',
                                          'background': '#eaf2f8'}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label('选择片区:',
                                           style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Dropdown(id='district-filter', placeholder='全部片区',
                                             clearable=True)
                            ], md=3),
                            dbc.Col([
                                html.Label('巡检批次:',
                                           style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Dropdown(id='batch-filter', placeholder='全部批次', multi=True)
                            ], md=6),
                            dbc.Col([
                                html.Label('数据概览:',
                                           style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                html.Div(id='data-summary',
                                         style={'fontSize': '13px', 'padding': '6px 10px',
                                                'background': '#f8f9fa', 'borderRadius': '4px'})
                            ], md=3)
                        ])
                    ])
                ], style=CARD_STYLE)
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('统计概览',
                                   style={'fontWeight': 'bold', 'fontSize': '16px',
                                          'background': '#e8f8f5'}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Div('记录总数',
                                             style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-total', style={
                                        'fontSize': '28px', 'fontWeight': 'bold', 'color': '#2c3e50'})
                                ], style={'textAlign': 'center', 'padding': '15px',
                                          'background': '#f8f9fa', 'borderRadius': '8px'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.Div('管段数量',
                                             style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-pipes', style={
                                        'fontSize': '28px', 'fontWeight': 'bold', 'color': '#2980b9'})
                                ], style={'textAlign': 'center', 'padding': '15px',
                                          'background': '#ebf5fb', 'borderRadius': '8px'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.Div('最大淤积深度(mm)',
                                             style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-max-depth', style={
                                        'fontSize': '28px', 'fontWeight': 'bold', 'color': '#8e44ad'})
                                ], style={'textAlign': 'center', 'padding': '15px',
                                          'background': '#f5eef8', 'borderRadius': '8px'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.Div('平均淤积率',
                                             style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-avg-rate', style={
                                        'fontSize': '28px', 'fontWeight': 'bold', 'color': '#d35400'})
                                ], style={'textAlign': 'center', 'padding': '15px',
                                          'background': '#fef5e7', 'borderRadius': '8px'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.Div('高风险管段',
                                             style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-high-risk', style={
                                        'fontSize': '28px', 'fontWeight': 'bold', 'color': '#c0392b'})
                                ], style={'textAlign': 'center', 'padding': '15px',
                                          'background': '#fdedec', 'borderRadius': '8px'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.Div('异常增长管段',
                                             style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-abnormal', style={
                                        'fontSize': '28px', 'fontWeight': 'bold', 'color': '#e67e22'})
                                ], style={'textAlign': 'center', 'padding': '15px',
                                          'background': '#fef9e7', 'borderRadius': '8px'})
                            ], md=2)
                        ])
                    ])
                ], style=CARD_STYLE)
            ], width=12)
        ]),

        dbc.Tabs([
            dbc.Tab(label='📈 单管段淤积过程', tab_id='tab-single', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label('选择管段编号:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(id='pipe-select',
                                                     placeholder='请选择管段编号...')
                                    ], md=6)
                                ], style={'marginBottom': '20px'}),
                                dcc.Graph(id='pipe-history-chart')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ]),

            dbc.Tab(label='📊 管段对比分析', tab_id='tab-compare', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label('选择对比管段 (可多选):',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(id='pipes-compare-select', multi=True,
                                                     placeholder='请选择多个管段进行对比...')
                                    ], md=6),
                                    dbc.Col([
                                        html.Label('对比指标:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id='compare-by',
                                            options=[
                                                {'label': '淤积率 (%)', 'value': '淤积率'},
                                                {'label': '淤积深度 (mm)', 'value': '淤积深度'}
                                            ],
                                            value='淤积率', clearable=False
                                        )
                                    ], md=3)
                                ], style={'marginBottom': '15px'}),
                                html.Div(id='compare-warning'),
                                dcc.Graph(id='pipes-comparison-chart')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ]),

            dbc.Tab(label='🔥 风险分布热力图', tab_id='tab-heatmap', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='risk-heatmap')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ]),

            dbc.Tab(label='📉 区域趋势分析', tab_id='tab-trend', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([dcc.Graph(id='trend-chart')])
                        ], style=CARD_STYLE)
                    ], md=7),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([dcc.Graph(id='risk-distribution-chart')])
                        ], style=CARD_STYLE)
                    ], md=5)
                ])
            ]),

            dbc.Tab(label='⚠️ 风险与异常', tab_id='tab-risk', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('高风险区段列表',
                                           style={'fontWeight': 'bold', 'background': '#fdedec'}),
                            dbc.CardBody([html.Div(id='high-risk-table')])
                        ], style=CARD_STYLE)
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('异常增长管段',
                                           style={'fontWeight': 'bold', 'background': '#fef9e7'}),
                            dbc.CardBody([html.Div(id='abnormal-growth-table')])
                        ], style=CARD_STYLE)
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('缺失巡检记录',
                                           style={'fontWeight': 'bold', 'background': '#f2f3f4'}),
                            dbc.CardBody([html.Div(id='missing-inspections-table')])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ]),

            dbc.Tab(label='📋 巡检台账编辑', tab_id='tab-edit', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('巡检数据编辑',
                                           style={'fontWeight': 'bold', 'background': '#e8f6f3'}),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.ButtonGroup([
                                            dbc.Button('添加记录', id='btn-add-row',
                                                       color='success', outline=True, size='sm',
                                                       className='me-2'),
                                            dbc.Button('删除选中行', id='btn-delete-row',
                                                       color='danger', outline=True, size='sm',
                                                       className='me-2'),
                                            dbc.Button('保存修改', id='btn-save-edits',
                                                       color='primary', size='sm', className='me-2'),
                                            dbc.Button('重置修改', id='btn-reset-edits',
                                                       color='secondary', outline=True, size='sm')
                                        ])
                                    ])
                                ], style={'marginBottom': '15px'}),
                                html.Div(id='edit-status', style={'marginBottom': '10px'}),
                                html.Div(id='detail-table')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ]),

            dbc.Tab(label='🔧 批次校验修复', tab_id='tab-batch-validate', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('批次数据校验与修复',
                                           style={'fontWeight': 'bold', 'background': '#eaf2f8'}),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label('选择批次:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(id='batch-validate-select',
                                                     placeholder='选择要校验的批次（留空校验全部）')
                                    ], md=4),
                                    dbc.Col([
                                        html.Div([
                                            dbc.Button('校验批次', id='btn-batch-validate',
                                                       color='primary', size='sm', className='me-2'),
                                            dbc.Button('修复数据', id='btn-batch-repair',
                                                       color='warning', size='sm')
                                        ], style={'marginTop': '25px'})
                                    ], md=4)
                                ], style={'marginBottom': '20px'}),
                                html.Div(id='batch-validate-result')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ]),

            dbc.Tab(label='🗺️ 跨片区隔离分析', tab_id='tab-district', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                '跨片区隔离分析（各片区独立统计，禁止合并对比）',
                                style={'fontWeight': 'bold', 'background': '#fdf2e9'}),
                            dbc.CardBody([dcc.Graph(id='district-comparison-chart')])
                        ], style=CARD_STYLE)
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('片区汇总表',
                                           style={'fontWeight': 'bold', 'background': '#eaf2f8'}),
                            dbc.CardBody([html.Div(id='district-summary-table')])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ]),

            dbc.Tab(label='⚙️ 风险预警配置', tab_id='tab-rules', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('风险预警规则配置',
                                           style={'fontWeight': 'bold', 'background': '#f5eef8'}),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label('高风险阈值:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Input(id='risk-threshold-high', type='number',
                                                  value=DEFAULT_RISK_RULES['RISK_THRESHOLD_HIGH'],
                                                  step=0.05, min=0, max=1,
                                                  style={'width': '100%'})
                                    ], md=3),
                                    dbc.Col([
                                        html.Label('中风险阈值:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Input(id='risk-threshold-medium', type='number',
                                                  value=DEFAULT_RISK_RULES['RISK_THRESHOLD_MEDIUM'],
                                                  step=0.05, min=0, max=1,
                                                  style={'width': '100%'})
                                    ], md=3),
                                    dbc.Col([
                                        html.Label('异常增长率阈值:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Input(id='abnormal-growth-rate-input', type='number',
                                                  value=DEFAULT_RISK_RULES['ABNORMAL_GROWTH_RATE'],
                                                  step=0.05, min=0,
                                                  style={'width': '100%'})
                                    ], md=3),
                                    dbc.Col([
                                        html.Label('清淤效果阈值:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Input(id='dredging-threshold', type='number',
                                                  value=DEFAULT_RISK_RULES['DREDGING_EFFECT_THRESHOLD'],
                                                  step=0.05, min=0, max=1,
                                                  style={'width': '100%'})
                                    ], md=3)
                                ], style={'marginBottom': '20px'}),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label('预警开关:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dbc.Checklist(
                                            id='alert-switches',
                                            options=[
                                                {'label': ' 缺失巡检预警',
                                                 'value': 'MISSING_INSPECTION_ALERT'},
                                                {'label': ' 异常增长预警',
                                                 'value': 'ABNORMAL_GROWTH_ALERT'},
                                                {'label': ' 高风险预警',
                                                 'value': 'HIGH_RISK_ALERT'},
                                            ],
                                            value=['MISSING_INSPECTION_ALERT',
                                                   'ABNORMAL_GROWTH_ALERT',
                                                   'HIGH_RISK_ALERT'],
                                            style={'fontSize': '14px'}
                                        )
                                    ], md=6),
                                    dbc.Col([
                                        html.Div([
                                            dbc.Button('应用配置', id='btn-apply-rules',
                                                       color='primary', size='sm',
                                                       style={'marginTop': '25px'})
                                        ])
                                    ], md=3)
                                ], style={'marginBottom': '15px'}),
                                html.Div(id='rules-apply-status')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ]),

            dbc.Tab(label='🧹 清淤效果评估', tab_id='tab-dredge', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('单管段清淤效果评估',
                                           style={'fontWeight': 'bold', 'background': '#e8f8f5'}),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label('选择管段:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(id='dredge-pipe-select',
                                                     placeholder='请选择管段...')
                                    ], md=3),
                                    dbc.Col([
                                        html.Label('清淤前批次:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(id='dredge-pre-batch',
                                                     placeholder='选择清淤前批次...')
                                    ], md=3),
                                    dbc.Col([
                                        html.Label('清淤后批次:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(id='dredge-post-batch',
                                                     placeholder='选择清淤后批次...')
                                    ], md=3),
                                    dbc.Col([
                                        html.Div([
                                            dbc.Button('评估效果', id='btn-dredge-eval',
                                                       color='primary', size='sm',
                                                       style={'marginTop': '25px'})
                                        ])
                                    ], md=3)
                                ], style={'marginBottom': '20px'}),
                                html.Div(id='dredge-eval-result')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('批量清淤效果评估',
                                           style={'fontWeight': 'bold', 'background': '#fef9e7'}),
                            dbc.CardBody([
                                dbc.Button('批量评估所有管段清淤效果',
                                           id='btn-dredge-batch-eval',
                                           color='warning', size='sm', style={'marginBottom': '15px'}),
                                html.Div(id='dredge-batch-result')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ]),
                dcc.Graph(id='dredge-effect-chart')
            ]),

            dbc.Tab(label='📋 处理建议看板', tab_id='tab-priority', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('清淤优先级排序看板',
                                           style={'fontWeight': 'bold', 'background': '#fdedec'}),
                            dbc.CardBody([dcc.Graph(id='priority-dashboard-chart')])
                        ], style=CARD_STYLE)
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('处理优先级明细表',
                                           style={'fontWeight': 'bold', 'background': '#eaf2f8'}),
                            dbc.CardBody([html.Div(id='priority-table')])
                        ], style=CARD_STYLE)
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('巡检质量报告',
                                           style={'fontWeight': 'bold', 'background': '#f5eef8'}),
                            dbc.CardBody([html.Div(id='quality-report')])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ]),

            dbc.Tab(label='🎯 任务编排与闭环处置', tab_id='tab-tasks', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('任务生成配置',
                                           style={'fontWeight': 'bold', 'background': '#e8f8f5', 'fontSize': '16px'}),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label('选择片区:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(id='task-district-filter', placeholder='全部片区',
                                                     clearable=True)
                                    ], md=2),
                                    dbc.Col([
                                        html.Label('关联巡检批次:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(id='task-batch-filter', placeholder='全部批次')
                                    ], md=2),
                                    dbc.Col([
                                        html.Label('每批最大任务数:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Input(id='task-max-count', type='number',
                                                  value=50, min=1, max=500, step=10,
                                                  style={'width': '100%', 'height': '36px'})
                                    ], md=2),
                                    dbc.Col([
                                        html.Label('派发人员:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id='task-assignee-select',
                                            options=[
                                                {'label': '巡检组A - 张工', 'value': '张工(巡检组A)'},
                                                {'label': '巡检组B - 李工', 'value': '李工(巡检组B)'},
                                                {'label': '清淤组1 - 王队长', 'value': '王队长(清淤组1)'},
                                                {'label': '清淤组2 - 赵队长', 'value': '赵队长(清淤组2)'},
                                                {'label': '机动组 - 刘工', 'value': '刘工(机动组)'}
                                            ],
                                            placeholder='选择派发人员',
                                            clearable=True, multi=False
                                        )
                                    ], md=3),
                                    dbc.Col([
                                        html.Div([
                                            dbc.ButtonGroup([
                                                dbc.Button('🔄 自动生成任务', id='btn-generate-tasks',
                                                           color='primary', size='md',
                                                           style={'marginTop': '25px', 'width': '100%'}),
                                                dbc.Button('📊 刷新状态/预警', id='btn-refresh-task-status',
                                                           color='info', size='md',
                                                           style={'marginTop': '25px', 'width': '100%'})
                                            ], style={'width': '100%'})
                                        ])
                                    ], md=3)
                                ], style={'marginBottom': '15px'}),
                                html.Div(id='task-generate-status', style={'marginBottom': '5px'})
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ]),

                dbc.Row([
                    dbc.Col(md=3, children=[
                        dbc.Card([
                            dbc.CardHeader('📈 任务统计概览',
                                           style={'fontWeight': 'bold', 'background': '#eaf2f8'}),
                            dbc.CardBody([
                                html.Div(id='task-stat-cards', style={'marginBottom': '15px'}),
                                dcc.Graph(id='task-status-chart')
                            ])
                        ], style=CARD_STYLE)
                    ]),
                    dbc.Col(md=9, children=[
                        dbc.Card([
                            dbc.CardHeader('📋 任务清单（支持多选批量派发/处理）',
                                           style={'fontWeight': 'bold', 'background': '#fdf2e9'}),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col(md=3, children=[
                                        html.Label('任务状态筛选:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id='task-status-filter',
                                            options=[
                                                {'label': '全部状态', 'value': 'ALL'},
                                                {'label': '待派发', 'value': '待派发'},
                                                {'label': '已派发', 'value': '已派发'},
                                                {'label': '处理中', 'value': '处理中'},
                                                {'label': '已完成', 'value': '已完成'},
                                                {'label': '已超期', 'value': '已超期'},
                                                {'label': '已闭环', 'value': '已闭环'}
                                            ],
                                            value='ALL', clearable=False
                                        )
                                    ]),
                                    dbc.Col(md=2, children=[
                                        html.Label('任务类型:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id='task-type-filter',
                                            options=[
                                                {'label': '全部类型', 'value': 'ALL'},
                                                {'label': '巡检', 'value': '巡检'},
                                                {'label': '复检', 'value': '复检'},
                                                {'label': '清淤', 'value': '清淤'}
                                            ],
                                            value='ALL', clearable=False
                                        )
                                    ]),
                                    dbc.Col(md=2, children=[
                                        html.Label('优先级:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id='task-priority-filter',
                                            options=[
                                                {'label': '全部优先级', 'value': 'ALL'},
                                                {'label': '紧急', 'value': '紧急'},
                                                {'label': '高', 'value': '高'},
                                                {'label': '中', 'value': '中'},
                                                {'label': '低', 'value': '低'}
                                            ],
                                            value='ALL', clearable=False
                                        )
                                    ]),
                                    dbc.Col(md=5, children=[
                                        html.Label('批量操作:',
                                                   style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dbc.ButtonGroup([
                                            dbc.Button('📤 批量派发', id='btn-batch-assign',
                                                       color='primary', size='sm', className='me-1'),
                                            dbc.Button('✅ 批量闭环', id='btn-batch-close',
                                                       color='success', size='sm', className='me-1'),
                                            dbc.Button('🗑️ 删除选中', id='btn-batch-delete',
                                                       color='danger', size='sm'),
                                        ], style={'marginTop': '25px'})
                                    ])
                                ], style={'marginBottom': '12px'}),
                                html.Div(id='task-list-table')
                            ])
                        ], style=CARD_STYLE)
                    ])
                ]),

                dbc.Row([
                    dbc.Col(md=6, children=[
                        dbc.Card([
                            dbc.CardHeader('🔍 任务详情与处理',
                                           style={'fontWeight': 'bold', 'background': '#f5eef8'}),
                            dbc.CardBody([
                                html.Div(id='task-detail-panel',
                                         children=[make_empty_msg('请从任务清单中选择一个任务查看详情并处理')])
                            ])
                        ], style=CARD_STYLE)
                    ]),
                    dbc.Col(md=6, children=[
                        dbc.Card([
                            dbc.CardHeader('📊 整改前后效果对比',
                                           style={'fontWeight': 'bold', 'background': '#e8f8f5'}),
                            dbc.CardBody([
                                dcc.Graph(id='task-before-after-chart')
                            ])
                        ], style=CARD_STYLE)
                    ])
                ]),

                dbc.Row([
                    dbc.Col(md=6, children=[
                        dbc.Card([
                            dbc.CardHeader('📊 完成率与闭环率统计',
                                           style={'fontWeight': 'bold', 'background': '#eaf2f8'}),
                            dbc.CardBody([
                                dcc.Graph(id='task-completion-chart')
                            ])
                        ], style=CARD_STYLE)
                    ]),
                    dbc.Col(md=6, children=[
                        dbc.Card([
                            dbc.CardHeader('🧹 清淤效果评估汇总',
                                           style={'fontWeight': 'bold', 'background': '#fef9e7'}),
                            dbc.CardBody([
                                dcc.Graph(id='task-dredging-effect-chart')
                            ])
                        ], style=CARD_STYLE)
                    ])
                ])
            ])
        ], id='main-tabs', active_tab='tab-single', style={'marginBottom': '20px'})
    ]),

    html.Footer([
        html.Hr(),
        html.Div('城市雨水管网淤积巡检分析台（二期升级版） | Python + Dash',
                 style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px'})
    ]),

    dcc.Store(id='save-signal', data=0),
    dcc.Store(id='risk-rules-store', data=None),
    dcc.Store(id='task-signal', data=0),
    dcc.Store(id='selected-tasks-store', data=[])

], fluid=True, style={'padding': '20px', 'backgroundColor': '#f5f6fa', 'minHeight': '100vh'})


@callback(
    [Output('upload-status', 'children'),
     Output('file-info', 'children'),
     Output('import-report', 'children'),
     Output('main-content', 'style'),
     Output('district-filter', 'options'),
     Output('district-filter', 'value'),
     Output('batch-filter', 'options'),
     Output('pipe-select', 'options'),
     Output('pipes-compare-select', 'options'),
     Output('batch-validate-select', 'options'),
     Output('dredge-pipe-select', 'options'),
     Output('task-district-filter', 'options'),
     Output('task-district-filter', 'value'),
     Output('task-batch-filter', 'options'),
     Output('task-batch-filter', 'value')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_upload(contents, filename):
    if contents is None:
        return (
            '', '',
            make_empty_msg('尚未导入数据'),
            {'display': 'none'},
            [], None, [], [], [], [], [],
            [], None, [], None
        )

    result = parse_contents(contents, filename)

    if len(result) == 2:
        _, msg = result
        return (
            dbc.Alert(msg, color='danger'), '',
            make_empty_msg('尚未导入数据'),
            {'display': 'none'},
            [], None, [], [], [], [], [],
            [], None, [], None
        )

    valid_df, errors, warnings, has_district, raw_df = result
    GLOBAL_DATA['valid_df'] = valid_df
    GLOBAL_DATA['errors'] = errors
    GLOBAL_DATA['warnings'] = warnings
    GLOBAL_DATA['has_district'] = has_district
    GLOBAL_DATA['raw_df'] = raw_df
    GLOBAL_DATA['tasks_df'] = pd.DataFrame()
    GLOBAL_DATA['selected_task_id'] = None

    upload_status = dbc.Alert(f'成功导入 {len(valid_df)} 条有效记录', color='success')
    file_info = (f'文件: {filename} | 原始记录: {len(raw_df)} 条 | '
                 f'有效记录: {len(valid_df)} 条 | 错误: {len(errors)} 条')

    report_parts = []
    if warnings:
        report_parts.append(html.Div([
            html.H6('⚠️ 警告:', style={'color': '#e67e22', 'marginBottom': '10px'}),
            html.Ul([html.Li(w, style={'color': '#d35400'}) for w in warnings])
        ], style={'marginBottom': '15px'}))

    if errors:
        error_df = pd.DataFrame(errors)
        report_parts.append(html.Div([
            html.H6(f'❌ 共 {len(errors)} 条记录导入失败:',
                    style={'color': '#c0392b', 'marginBottom': '10px'}),
            dash_table.DataTable(
                data=error_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in error_df.columns],
                style_table={'overflowX': 'auto', 'maxHeight': '300px', 'overflowY': 'auto'},
                style_header={'backgroundColor': '#f8d7da', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                page_size=10
            )
        ]))

    if not errors and not warnings:
        report_parts.append(html.Div('✅ 所有数据校验通过，无异常！',
                                     style={'color': '#27ae60', 'textAlign': 'center',
                                            'padding': '20px', 'fontSize': '16px'}))

    import_report = html.Div(report_parts) if report_parts else \
        html.Div('✅ 所有数据校验通过',
                 style={'textAlign': 'center', 'color': '#27ae60', 'padding': '20px'})

    districts = get_districts(valid_df)
    district_options = [{'label': d, 'value': d} for d in districts]

    batches = get_batches(valid_df)
    batch_options = [{'label': b, 'value': b} for b in batches]
    latest_batch = batches[-1] if batches else None

    pipes = get_pipe_ids(valid_df)
    pipe_options = [{'label': p, 'value': p} for p in pipes]

    return (
        upload_status, file_info, import_report,
        {'display': 'block'},
        district_options, None, batch_options,
        pipe_options, pipe_options, batch_options, pipe_options,
        district_options, None, batch_options, latest_batch
    )


@callback(
    [Output('batch-filter', 'value'),
     Output('data-summary', 'children'),
     Output('stat-total', 'children'),
     Output('stat-pipes', 'children'),
     Output('stat-max-depth', 'children'),
     Output('stat-avg-rate', 'children'),
     Output('stat-high-risk', 'children'),
     Output('stat-abnormal', 'children'),
     Output('pipe-select', 'value'),
     Output('pipes-compare-select', 'value')],
    [Input('district-filter', 'value')]
)
def update_on_district_change(district):
    if GLOBAL_DATA['valid_df'].empty:
        return [], '无数据', '-', '-', '-', '-', '-', '-', None, None

    df = GLOBAL_DATA['valid_df']
    batches = get_batches(df, district)
    rules = GLOBAL_DATA.get('custom_rules')

    stats = calculate_statistics(df, district, batches, rules=rules)
    abnormal = detect_abnormal_growth(df, district, batches, rules=rules)

    summary = (f"共 {stats.get('记录总数', 0)} 条记录 / "
               f"{stats.get('管段数量', 0)} 个管段 / "
               f"{stats.get('巡检批次数量', 0)} 个批次")

    return (
        batches, summary,
        stats.get('记录总数', 0),
        stats.get('管段数量', 0),
        stats.get('最大淤积深度', '-'),
        f"{stats.get('平均淤积率', 0) * 100:.1f}%" if stats.get('平均淤积率') is not None else '-',
        stats.get('高风险管段数', 0),
        len(abnormal),
        None, None
    )


@callback(
    [Output('stat-total', 'children', allow_duplicate=True),
     Output('stat-pipes', 'children', allow_duplicate=True),
     Output('stat-max-depth', 'children', allow_duplicate=True),
     Output('stat-avg-rate', 'children', allow_duplicate=True),
     Output('stat-high-risk', 'children', allow_duplicate=True),
     Output('stat-abnormal', 'children', allow_duplicate=True),
     Output('data-summary', 'children', allow_duplicate=True)],
    [Input('batch-filter', 'value'),
     Input('save-signal', 'data'),
     Input('risk-rules-store', 'data')],
    [State('district-filter', 'value')],
    prevent_initial_call=True
)
def update_stats_on_change(selected_batches, save_signal, rules_data, district):
    if GLOBAL_DATA['valid_df'].empty:
        return '-', '-', '-', '-', '-', '-', '无数据'

    df = GLOBAL_DATA['valid_df']
    batches = selected_batches if selected_batches else get_batches(df, district)

    stats = calculate_statistics(df, district, batches, rules=rules_data)
    abnormal = detect_abnormal_growth(df, district, rules=rules_data)

    summary = (f"共 {stats.get('记录总数', 0)} 条记录 / "
               f"{stats.get('管段数量', 0)} 个管段 / "
               f"{len(batches)} 个批次")

    return (
        stats.get('记录总数', 0),
        stats.get('管段数量', 0),
        stats.get('最大淤积深度', '-'),
        f"{stats.get('平均淤积率', 0) * 100:.1f}%" if stats.get('平均淤积率') is not None else '-',
        stats.get('高风险管段数', 0),
        len(abnormal),
        summary
    )


@callback(
    Output('pipe-history-chart', 'figure'),
    [Input('pipe-select', 'value'),
     Input('district-filter', 'value'),
     Input('save-signal', 'data'),
     Input('risk-rules-store', 'data')]
)
def update_pipe_history(pipe_id, district, save_signal, rules_data):
    if GLOBAL_DATA['valid_df'].empty or not pipe_id:
        return make_empty_fig('请选择管段编号查看淤积过程')

    return create_pipe_history_chart(GLOBAL_DATA['valid_df'], pipe_id, district, rules=rules_data)


@callback(
    [Output('compare-warning', 'children'),
     Output('pipes-comparison-chart', 'figure')],
    [Input('pipes-compare-select', 'value'),
     Input('compare-by', 'value'),
     Input('district-filter', 'value'),
     Input('save-signal', 'data'),
     Input('risk-rules-store', 'data')]
)
def update_pipes_comparison(pipe_ids, compare_by, district, save_signal, rules_data):
    if GLOBAL_DATA['valid_df'].empty or not pipe_ids:
        return '', make_empty_fig('请选择至少一个管段进行对比分析')

    is_valid, msg = validate_cross_district(pipe_ids, GLOBAL_DATA['valid_df'])
    if not is_valid:
        warning = dbc.Alert(msg, color='warning', style={'marginBottom': '15px'})
        return warning, make_empty_fig('跨片区对比已阻止')

    return '', create_pipes_comparison_chart(
        GLOBAL_DATA['valid_df'], pipe_ids, district, compare_by, rules=rules_data
    )


@callback(
    Output('risk-heatmap', 'figure'),
    [Input('district-filter', 'value'),
     Input('batch-filter', 'value'),
     Input('save-signal', 'data'),
     Input('risk-rules-store', 'data')]
)
def update_risk_heatmap(district, batches, save_signal, rules_data):
    if GLOBAL_DATA['valid_df'].empty:
        return make_empty_fig('暂无数据')

    return create_risk_heatmap(GLOBAL_DATA['valid_df'], district, batches, rules=rules_data)


@callback(
    [Output('trend-chart', 'figure'),
     Output('risk-distribution-chart', 'figure')],
    [Input('district-filter', 'value'),
     Input('batch-filter', 'value'),
     Input('save-signal', 'data'),
     Input('risk-rules-store', 'data')]
)
def update_trend_and_distribution(district, batches, save_signal, rules_data):
    if GLOBAL_DATA['valid_df'].empty:
        return make_empty_fig(), make_empty_fig()

    trend_fig = create_sediment_trend_chart(
        GLOBAL_DATA['valid_df'], district, batches, rules=rules_data
    )
    stats = calculate_statistics(GLOBAL_DATA['valid_df'], district, batches, rules=rules_data)
    dist_fig = create_risk_distribution_chart(stats, rules=rules_data)

    return trend_fig, dist_fig


@callback(
    [Output('high-risk-table', 'children'),
     Output('abnormal-growth-table', 'children'),
     Output('missing-inspections-table', 'children')],
    [Input('district-filter', 'value'),
     Input('batch-filter', 'value'),
     Input('save-signal', 'data'),
     Input('risk-rules-store', 'data')]
)
def update_risk_tables(district, batches, save_signal, rules_data):
    if GLOBAL_DATA['valid_df'].empty:
        return make_empty_msg(), make_empty_msg(), make_empty_msg()

    high_risk_df = get_high_risk_segments(
        GLOBAL_DATA['valid_df'], district, batches, rules=rules_data
    )
    abnormal_df = detect_abnormal_growth(
        GLOBAL_DATA['valid_df'], district, batches, rules=rules_data
    )
    missing_df = detect_missing_inspections(
        GLOBAL_DATA['valid_df'], district, batches
    )

    def make_table(df, cols, empty_text='暂无数据'):
        if df.empty:
            return make_empty_msg(empty_text)
        display_df = df.copy()
        for col in display_df.columns:
            if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')
        return dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=cols,
            style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
            style_header={'backgroundColor': '#2c3e50', 'color': 'white',
                          'fontWeight': 'bold', 'textAlign': 'center'},
            style_cell={'textAlign': 'left', 'padding': '8px 12px', 'fontSize': '13px'},
            style_data_conditional=[{
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            }],
            page_size=15, sort_action='native', filter_action='native', export_format='csv'
        )

    high_risk_cols = [
        {"name": "管段编号", "id": "管段编号"},
        {"name": "片区", "id": "片区"},
        {"name": "巡检批次", "id": "巡检批次"},
        {"name": "检查时间", "id": "检查时间"},
        {"name": "淤积深度(mm)", "id": "淤积深度"},
        {"name": "管径(mm)", "id": "管径"},
        {"name": "淤积率", "id": "淤积率", "type": "numeric", "format": {"specifier": ".1%"}},
        {"name": "备注", "id": "备注"}
    ]

    abnormal_cols = [
        {"name": "管段编号", "id": "管段编号"},
        {"name": "片区", "id": "片区"},
        {"name": "前批次", "id": "前批次"},
        {"name": "当前批次", "id": "当前批次"},
        {"name": "前淤积率", "id": "前淤积率", "type": "numeric", "format": {"specifier": ".1%"}},
        {"name": "当前淤积率", "id": "当前淤积率", "type": "numeric", "format": {"specifier": ".1%"}},
        {"name": "增长率", "id": "增长率", "type": "numeric", "format": {"specifier": ".1%"}},
        {"name": "管径(mm)", "id": "管径"}
    ]

    missing_cols = [
        {"name": "管段编号", "id": "管段编号"},
        {"name": "片区", "id": "片区"},
        {"name": "缺失批次", "id": "缺失批次"}
    ]

    return (
        make_table(high_risk_df, high_risk_cols, '暂无高风险数据'),
        make_table(abnormal_df, abnormal_cols, '暂无异常增长数据'),
        make_table(missing_df, missing_cols, '暂无缺失巡检数据')
    )


@callback(
    Output('detail-table', 'children'),
    [Input('district-filter', 'value'),
     Input('batch-filter', 'value'),
     Input('save-signal', 'data'),
     Input('btn-reset-edits', 'n_clicks')]
)
def regenerate_detail_table(district, batches, save_signal, reset_clicks):
    if GLOBAL_DATA['valid_df'].empty:
        return make_empty_msg()

    detail_df = GLOBAL_DATA['valid_df'].copy()
    if district:
        detail_df = detail_df[detail_df['片区'] == district]
    if batches:
        detail_df = detail_df[detail_df['巡检批次'].isin(batches)]
    detail_df = detail_df.sort_values(['片区', '管段编号', '检查时间'])

    display_df = detail_df.copy()
    for col in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[col]):
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')

    detail_cols = [
        {"name": "管段编号", "id": "管段编号", "editable": True},
        {"name": "片区", "id": "片区", "editable": True},
        {"name": "巡检批次", "id": "巡检批次", "editable": True},
        {"name": "检查时间", "id": "检查时间", "editable": True, "type": "datetime"},
        {"name": "淤积深度(mm)", "id": "淤积深度", "editable": True, "type": "numeric"},
        {"name": "管径(mm)", "id": "管径", "editable": True, "type": "numeric"},
        {"name": "淤积率", "id": "淤积率", "editable": False, "type": "numeric",
         "format": {"specifier": ".1%"}},
        {"name": "备注", "id": "备注", "editable": True}
    ]

    rules_data = GLOBAL_DATA.get('custom_rules')
    rules = get_effective_rules(rules_data)
    threshold_high = float(rules.get('RISK_THRESHOLD_HIGH', 0.6))
    threshold_medium = float(rules.get('RISK_THRESHOLD_MEDIUM', 0.3))

    return dash_table.DataTable(
        id='editable-detail-table',
        data=display_df.to_dict('records'),
        columns=detail_cols,
        editable=True,
        row_deletable=True,
        row_selectable='multi',
        selected_rows=[],
        style_table={'overflowX': 'auto', 'maxHeight': '450px', 'overflowY': 'auto'},
        style_header={
            'backgroundColor': '#16a085', 'color': 'white',
            'fontWeight': 'bold', 'textAlign': 'center'
        },
        style_cell={'textAlign': 'left', 'padding': '8px 12px', 'fontSize': '13px'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'},
            {'if': {'filter_query': f'{{淤积率}} >= {threshold_high}',
                    'column_id': '淤积率'},
             'backgroundColor': '#fdedec', 'color': '#c0392b', 'fontWeight': 'bold'},
            {'if': {'filter_query': f'{{淤积率}} >= {threshold_medium} && '
                                    f'{{淤积率}} < {threshold_high}',
                    'column_id': '淤积率'},
             'backgroundColor': '#fef9e7', 'color': '#d68910'}
        ],
        page_size=15, sort_action='native', filter_action='native',
        export_format='csv', fill_width=False,
        style_cell_conditional=[
            {'if': {'column_id': '管段编号'}, 'width': '100px'},
            {'if': {'column_id': '片区'}, 'width': '80px'},
            {'if': {'column_id': '巡检批次'}, 'width': '100px'},
            {'if': {'column_id': '检查时间'}, 'width': '120px'},
            {'if': {'column_id': '淤积深度'}, 'width': '110px'},
            {'if': {'column_id': '管径'}, 'width': '90px'},
            {'if': {'column_id': '淤积率'}, 'width': '90px'},
            {'if': {'column_id': '备注'}, 'width': '200px'}
        ]
    )


@callback(
    [Output('edit-status', 'children'),
     Output('save-signal', 'data'),
     Output('import-report', 'children', allow_duplicate=True)],
    [Input('btn-save-edits', 'n_clicks'),
     Input('btn-add-row', 'n_clicks'),
     Input('btn-delete-row', 'n_clicks')],
    [State('district-filter', 'value'),
     State('batch-filter', 'value'),
     State('editable-detail-table', 'data'),
     State('editable-detail-table', 'selected_rows'),
     State('save-signal', 'data'),
     State('risk-rules-store', 'data')],
    prevent_initial_call=True
)
def handle_edit_actions(save_clicks, add_clicks, delete_clicks,
                        district, batches, table_data, selected_rows,
                        current_signal, rules_data):
    triggered = ctx.triggered_id
    new_signal = current_signal + 1 if current_signal else 1

    if triggered == 'btn-save-edits' and save_clicks and table_data is not None:
        edited_df = pd.DataFrame(table_data)

        required_cols = ['管段编号', '巡检批次', '检查时间', '淤积深度', '管径']
        for col in required_cols:
            if col not in edited_df.columns:
                edited_df[col] = None
        if '片区' not in edited_df.columns:
            edited_df['片区'] = district if district else '默认片区'
        if '备注' not in edited_df.columns:
            edited_df['备注'] = ''

        valid_df, errors = revalidate_dataframe(edited_df, rules=rules_data)

        if errors:
            error_df = pd.DataFrame(errors)
            report = html.Div([
                html.H6(f'⚠️ 保存后校验发现 {len(errors)} 条问题数据：',
                        style={'color': '#c0392b', 'marginBottom': '10px'}),
                dash_table.DataTable(
                    data=error_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in error_df.columns],
                    style_table={'overflowX': 'auto', 'maxHeight': '200px', 'overflowY': 'auto'},
                    style_header={'backgroundColor': '#f8d7da', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                    page_size=5
                )
            ])
        else:
            report = html.Div('✅ 所有数据校验通过！',
                              style={'color': '#27ae60', 'textAlign': 'center', 'padding': '20px'})

        if not valid_df.empty or district or batches:
            merged_df = merge_edited_subset(
                GLOBAL_DATA['valid_df'], valid_df, district, batches
            )
            GLOBAL_DATA['valid_df'] = merged_df
            GLOBAL_DATA['errors'] = errors
            status = dbc.Alert(
                f'保存成功！已将 {len(valid_df)} 条有效记录合并回数据集'
                + (f'，共 {len(merged_df)} 条总记录' if merged_df is not None else '')
                + (f'，忽略 {len(errors)} 条问题记录' if errors else '')
                + '，风险统计已重新计算',
                color='success', duration=5000
            )
        else:
            status = dbc.Alert('无有效记录被保存，请检查数据格式',
                               color='warning', duration=5000)

        return status, new_signal, report

    elif triggered == 'btn-add-row' and add_clicks:
        default_date = datetime.now().strftime('%Y-%m-%d')
        new_row = {
            '管段编号': f'NEW{add_clicks:03d}',
            '片区': district if district else '默认片区',
            '巡检批次': batches[-1] if batches else '新批次',
            '检查时间': default_date,
            '淤积深度': 0,
            '管径': 500,
            '淤积率': 0,
            '备注': '新增记录'
        }
        if table_data is None:
            table_data = []
        table_data.append(new_row)

        status = dbc.Alert('已添加新行，请填写数据后点击"保存修改"按钮生效',
                           color='info', duration=4000)
        return status, dash.no_update, dash.no_update

    elif triggered == 'btn-delete-row' and delete_clicks:
        if table_data is None or not selected_rows:
            status = dbc.Alert('请先勾选要删除的行', color='warning', duration=3000)
            return status, dash.no_update, dash.no_update

        table_data = [row for i, row in enumerate(table_data) if i not in selected_rows]

        status = dbc.Alert(f'已删除 {len(selected_rows)} 行，点击"保存修改"使删除生效',
                           color='info', duration=4000)
        return status, dash.no_update, dash.no_update

    return '', dash.no_update, dash.no_update


@callback(
    [Output('batch-validate-result', 'children'),
     Output('save-signal', 'data', allow_duplicate=True)],
    [Input('btn-batch-validate', 'n_clicks'),
     Input('btn-batch-repair', 'n_clicks')],
    [State('batch-validate-select', 'value'),
     State('risk-rules-store', 'data'),
     State('save-signal', 'data')],
    prevent_initial_call=True
)
def handle_batch_validate(validate_clicks, repair_clicks, batch_name, rules_data, current_save_signal):
    if GLOBAL_DATA['valid_df'].empty:
        return make_empty_msg('暂无数据'), dash.no_update

    triggered = ctx.triggered_id
    df = GLOBAL_DATA['valid_df']

    repaired_df, errors, repair_actions = validate_batch_data(df, batch_name, rules=rules_data)

    result_parts = []
    new_save_signal = current_save_signal if current_save_signal else 0

    if triggered == 'btn-batch-validate':
        if errors:
            error_df = pd.DataFrame(errors)
            result_parts.append(html.Div([
                html.H6(f'校验发现 {len(errors)} 条问题数据：',
                        style={'color': '#c0392b', 'marginBottom': '10px'}),
                dash_table.DataTable(
                    data=error_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in error_df.columns],
                    style_table={'overflowX': 'auto', 'maxHeight': '300px', 'overflowY': 'auto'},
                    style_header={'backgroundColor': '#f8d7da', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                    page_size=10
                )
            ]))
        else:
            result_parts.append(html.Div('✅ 批次数据校验通过，无问题数据',
                                         style={'color': '#27ae60', 'textAlign': 'center',
                                                'padding': '20px'}))

        if repair_actions:
            repair_df = pd.DataFrame(repair_actions)
            result_parts.append(html.Div([
                html.H6(f'可自动修复 {len(repair_actions)} 条数据：',
                        style={'color': '#f39c12', 'marginBottom': '10px', 'marginTop': '15px'}),
                dash_table.DataTable(
                    data=repair_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in repair_df.columns],
                    style_table={'overflowX': 'auto', 'maxHeight': '200px', 'overflowY': 'auto'},
                    style_header={'backgroundColor': '#fef9e7', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                    page_size=5
                )
            ]))

        return html.Div(result_parts) if result_parts else make_empty_msg('无结果'), dash.no_update

    elif triggered == 'btn-batch-repair':
        if repair_actions:
            repair_df = pd.DataFrame(repair_actions)
            result_parts.append(html.Div([
                html.H6(f'已修复 {len(repair_actions)} 条数据：',
                        style={'color': '#27ae60', 'marginBottom': '10px'}),
                dash_table.DataTable(
                    data=repair_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in repair_df.columns],
                    style_table={'overflowX': 'auto', 'maxHeight': '200px', 'overflowY': 'auto'},
                    style_header={'backgroundColor': '#d5f5e3', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                    page_size=5
                )
            ]))

            if not repaired_df.empty or batch_name:
                merged_df = merge_repaired_batch(GLOBAL_DATA['valid_df'], repaired_df, batch_name)
                GLOBAL_DATA['valid_df'] = merged_df
                new_save_signal = (current_save_signal if current_save_signal else 0) + 1
                result_parts.append(html.Div(
                    f'✅ 数据已修复并合并回数据集，当前共 {len(merged_df)} 条有效记录，风险统计已自动刷新',
                    style={'color': '#27ae60', 'marginTop': '10px'}
                ))
        else:
            result_parts.append(html.Div('无需修复的数据',
                                         style={'textAlign': 'center', 'color': '#7f8c8d',
                                                'padding': '20px'}))

        if errors:
            error_df = pd.DataFrame(errors)
            result_parts.append(html.Div([
                html.H6(f'仍有 {len(errors)} 条无法自动修复的问题：',
                        style={'color': '#c0392b', 'marginBottom': '10px', 'marginTop': '15px'}),
                dash_table.DataTable(
                    data=error_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in error_df.columns],
                    style_table={'overflowX': 'auto', 'maxHeight': '200px', 'overflowY': 'auto'},
                    style_header={'backgroundColor': '#f8d7da', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                    page_size=5
                )
            ]))

        return html.Div(result_parts) if result_parts else make_empty_msg('无结果'), new_save_signal

    return html.Div(result_parts) if result_parts else make_empty_msg('无结果'), dash.no_update


@callback(
    [Output('district-comparison-chart', 'figure'),
     Output('district-summary-table', 'children')],
    [Input('save-signal', 'data'),
     Input('risk-rules-store', 'data')]
)
def update_district_analysis(save_signal, rules_data):
    if GLOBAL_DATA['valid_df'].empty:
        return make_empty_fig('暂无数据'), make_empty_msg()

    summary_df = get_district_summary(GLOBAL_DATA['valid_df'], rules=rules_data)
    chart = create_district_comparison_chart(summary_df, rules=rules_data)

    if summary_df.empty:
        return chart, make_empty_msg()

    display_df = summary_df.copy()
    for col in ['平均淤积率', '最大淤积率']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.2%}')

    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in display_df.columns],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white',
                      'fontWeight': 'bold', 'textAlign': 'center'},
        style_cell={'textAlign': 'center', 'padding': '8px 12px', 'fontSize': '13px'},
        style_data_conditional=[{
            'if': {'row_index': 'odd'},
            'backgroundColor': '#f8f9fa'
        }],
        page_size=10, sort_action='native'
    )

    return chart, table


@callback(
    [Output('risk-rules-store', 'data'),
     Output('rules-apply-status', 'children')],
    [Input('btn-apply-rules', 'n_clicks')],
    [State('risk-threshold-high', 'value'),
     State('risk-threshold-medium', 'value'),
     State('abnormal-growth-rate-input', 'value'),
     State('dredging-threshold', 'value'),
     State('alert-switches', 'value'),
     State('risk-rules-store', 'data')],
    prevent_initial_call=True
)
def apply_risk_rules(clicks, threshold_high, threshold_medium, growth_rate,
                     dredging_threshold, alert_switches, current_rules):
    if not clicks:
        return dash.no_update, ''

    errors = []
    th_high = float(threshold_high) if threshold_high is not None and threshold_high != '' else None
    th_medium = float(threshold_medium) if threshold_medium is not None and threshold_medium != '' else None
    th_growth = float(growth_rate) if growth_rate is not None and growth_rate != '' else None
    th_dredge = float(dredging_threshold) if dredging_threshold is not None and dredging_threshold != '' else None

    if th_high is not None and th_medium is not None:
        if th_medium >= th_high:
            errors.append(f'中风险阈值 ({th_medium:.0%}) 必须低于高风险阈值 ({th_high:.0%})')

    if th_high is not None and (th_high <= 0 or th_high > 1):
        errors.append('高风险阈值必须在 (0, 1] 之间')
    if th_medium is not None and (th_medium < 0 or th_medium >= 1):
        errors.append('中风险阈值必须在 [0, 1) 之间')
    if th_growth is not None and th_growth < 0:
        errors.append('异常增长率阈值必须大于等于 0')
    if th_dredge is not None and (th_dredge < 0 or th_dredge > 1):
        errors.append('清淤效果阈值必须在 [0, 1] 之间')

    if errors:
        error_status = dbc.Alert([
            html.Div('❌ 风险预警配置校验失败，请修正以下问题：',
                     style={'fontWeight': 'bold', 'marginBottom': '8px', 'color': '#c0392b'}),
            html.Ul([html.Li(e, style={'color': '#c0392b'}) for e in errors])
        ], color='danger', duration=8000)
        return dash.no_update, error_status

    custom_rules = {}

    if th_high is not None:
        custom_rules['RISK_THRESHOLD_HIGH'] = th_high
    if th_medium is not None:
        custom_rules['RISK_THRESHOLD_MEDIUM'] = th_medium
    if th_growth is not None:
        custom_rules['ABNORMAL_GROWTH_RATE'] = th_growth
    if th_dredge is not None:
        custom_rules['DREDGING_EFFECT_THRESHOLD'] = th_dredge

    custom_rules['MISSING_INSPECTION_ALERT'] = 'MISSING_INSPECTION_ALERT' in (alert_switches or [])
    custom_rules['ABNORMAL_GROWTH_ALERT'] = 'ABNORMAL_GROWTH_ALERT' in (alert_switches or [])
    custom_rules['HIGH_RISK_ALERT'] = 'HIGH_RISK_ALERT' in (alert_switches or [])

    GLOBAL_DATA['custom_rules'] = custom_rules

    effective = get_effective_rules(custom_rules)
    status = dbc.Alert([
        html.Div('✅ 风险预警配置已更新并生效！', style={'fontWeight': 'bold', 'marginBottom': '8px'}),
        html.Ul([
            html.Li(f"高风险阈值: {effective['RISK_THRESHOLD_HIGH']:.0%}"),
            html.Li(f"中风险阈值: {effective['RISK_THRESHOLD_MEDIUM']:.0%}"),
            html.Li(f"异常增长率阈值: {effective['ABNORMAL_GROWTH_RATE']:.0%}"),
            html.Li(f"清淤效果阈值: {effective['DREDGING_EFFECT_THRESHOLD']:.0%}"),
            html.Li(f"缺失巡检预警: {'开启' if effective['MISSING_INSPECTION_ALERT'] else '关闭'}"),
            html.Li(f"异常增长预警: {'开启' if effective['ABNORMAL_GROWTH_ALERT'] else '关闭'}"),
            html.Li(f"高风险预警: {'开启' if effective['HIGH_RISK_ALERT'] else '关闭'}"),
        ])
    ], color='success', duration=6000)

    return custom_rules, status


@callback(
    [Output('dredge-pre-batch', 'options'),
     Output('dredge-pre-batch', 'value'),
     Output('dredge-post-batch', 'options'),
     Output('dredge-post-batch', 'value')],
    [Input('dredge-pipe-select', 'value')]
)
def update_dredge_batches(pipe_id):
    if GLOBAL_DATA['valid_df'].empty or not pipe_id:
        return [], None, [], None

    pipe_data = GLOBAL_DATA['valid_df'][GLOBAL_DATA['valid_df']['管段编号'] == pipe_id]
    if pipe_data.empty:
        return [], None, [], None

    pipe_batches = get_batches(pipe_data)
    batch_options = [{'label': b, 'value': b} for b in pipe_batches]
    return batch_options, None, batch_options, None


@callback(
    [Output('dredge-eval-result', 'children'),
     Output('dredge-effect-chart', 'figure')],
    [Input('btn-dredge-eval', 'n_clicks'),
     Input('btn-dredge-batch-eval', 'n_clicks')],
    [State('dredge-pipe-select', 'value'),
     State('dredge-pre-batch', 'value'),
     State('dredge-post-batch', 'value'),
     State('risk-rules-store', 'data')],
    prevent_initial_call=True
)
def handle_dredge_eval(individual_clicks, batch_clicks, pipe_id, pre_batch,
                       post_batch, rules_data):
    triggered = ctx.triggered_id

    if triggered == 'btn-dredge-eval':
        if not pipe_id or not pre_batch or not post_batch:
            return dbc.Alert('请选择管段和清淤前后批次', color='warning'), make_empty_fig()

        result = evaluate_dredging_effect(
            GLOBAL_DATA['valid_df'], pipe_id, pre_batch, post_batch, rules=rules_data
        )

        if result is None:
            return dbc.Alert('未找到该管段的清淤前后数据', color='warning'), make_empty_fig()

        color = result.get('效果颜色', '#95a5a6')
        detail = html.Div([
            dbc.Alert([
                html.H6(f"管段 {result['管段编号']} 清淤效果评估", style={'fontWeight': 'bold'}),
                html.Hr(style={'margin': '8px 0'}),
                html.Div(f"片区: {result['片区']}"),
                html.Div(f"清淤前批次: {result['清淤前批次']} | 清淤后批次: {result['清淤后批次']}"),
                html.Div(f"清淤前淤积深度: {result['清淤前淤积深度']}mm → "
                         f"清淤后: {result['清淤后淤积深度']}mm "
                         f"(减少 {result['深度减少量']}mm)"),
                html.Div(f"清淤前淤积率: {result['清淤前淤积率']:.1%} → "
                         f"清淤后: {result['清淤后淤积率']:.1%} "
                         f"(降幅 {result['降幅百分比']:.1%})"),
                html.Div([
                    html.Span(f"效果评级: {result['效果评级']}",
                              style={'fontWeight': 'bold', 'fontSize': '16px', 'color': color})
                ], style={'margin': '8px 0'}),
                html.Div(f"处理建议: {result['处理建议']}")
            ], color='info')
        ])

        chart = create_dredging_effect_chart(result, rules=rules_data)
        return detail, chart

    elif triggered == 'btn-dredge-batch-eval':
        if GLOBAL_DATA['valid_df'].empty:
            return make_empty_msg('暂无数据'), make_empty_fig()

        batch_results = batch_evaluate_dredging(GLOBAL_DATA['valid_df'], rules=rules_data)

        if batch_results.empty:
            return dbc.Alert('未发现有效的清淤效果数据', color='info'), make_empty_fig()

        display_df = batch_results.copy()
        for col in ['清淤前淤积率', '清淤后淤积率', '淤积率降幅']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.2%}')
        if '降幅百分比' in display_df.columns:
            display_df['降幅百分比'] = display_df['降幅百分比'].apply(lambda x: f'{x:.1%}')

        table = dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in display_df.columns],
            style_table={'overflowX': 'auto', 'maxHeight': '300px', 'overflowY': 'auto'},
            style_header={'backgroundColor': '#2c3e50', 'color': 'white',
                          'fontWeight': 'bold', 'textAlign': 'center'},
            style_cell={'textAlign': 'left', 'padding': '8px 12px', 'fontSize': '12px'},
            style_data_conditional=[{
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            }],
            page_size=10, sort_action='native'
        )

        result_html = html.Div([
            html.H6(f'批量评估完成，共发现 {len(batch_results)} 条清淤效果记录',
                    style={'marginBottom': '10px'}),
            table
        ])

        chart = create_dredging_effect_chart(batch_results, rules=rules_data)
        return result_html, chart

    return '', make_empty_fig()


@callback(
    [Output('priority-dashboard-chart', 'figure'),
     Output('priority-table', 'children'),
     Output('quality-report', 'children')],
    [Input('save-signal', 'data'),
     Input('risk-rules-store', 'data')]
)
def update_priority_dashboard(save_signal, rules_data):
    if GLOBAL_DATA['valid_df'].empty:
        return make_empty_fig(), make_empty_msg(), make_empty_msg()

    priority_df = calculate_dredging_priority(GLOBAL_DATA['valid_df'], rules=rules_data)
    chart = create_priority_dashboard_chart(priority_df, rules=rules_data)

    if priority_df.empty:
        table = make_empty_msg('暂无优先级数据')
    else:
        display_df = priority_df.copy()
        for col in ['最新淤积率', '增长率']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.2%}')
        if '风险评分' in display_df.columns:
            display_df['风险评分'] = display_df['风险评分'].apply(lambda x: f'{x:.1f}')

        priority_colors = {'紧急': '#c0392b', '高': '#e74c3c', '中': '#f39c12', '低': '#27ae60'}
        style_conditions = [
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
        ]
        for level, color in priority_colors.items():
            style_conditions.append({
                'if': {'filter_query': f'{{清淤优先级}} = "{level}"',
                       'column_id': '清淤优先级'},
                'color': color, 'fontWeight': 'bold'
            })

        table = dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in display_df.columns],
            style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
            style_header={'backgroundColor': '#2c3e50', 'color': 'white',
                          'fontWeight': 'bold', 'textAlign': 'center'},
            style_cell={'textAlign': 'left', 'padding': '8px 12px', 'fontSize': '12px'},
            style_data_conditional=style_conditions,
            page_size=15, sort_action='native', filter_action='native', export_format='csv'
        )

    report = generate_inspection_quality_report(GLOBAL_DATA['valid_df'], rules=rules_data)
    report_html = render_quality_report(report)

    return chart, table, report_html


@callback(
    [Output('task-generate-status', 'children'),
     Output('task-signal', 'data')],
    [Input('btn-generate-tasks', 'n_clicks'),
     Input('btn-refresh-task-status', 'n_clicks')],
    [State('task-district-filter', 'value'),
     State('task-batch-filter', 'value'),
     State('task-max-count', 'value'),
     State('task-assignee-select', 'value'),
     State('risk-rules-store', 'data'),
     State('task-signal', 'data')],
    prevent_initial_call=True
)
def handle_task_generate_and_refresh(gen_clicks, refresh_clicks,
                                     district, batch_name, max_count,
                                     assignee, rules_data, current_signal):
    triggered = ctx.triggered_id
    new_signal = (current_signal + 1) if current_signal else 1
    df = GLOBAL_DATA['valid_df']

    if df.empty:
        return dbc.Alert('请先导入巡检数据', color='warning'), dash.no_update

    if triggered == 'btn-generate-tasks':
        max_tasks = int(max_count) if max_count and max_count > 0 else 50
        existing_tasks = GLOBAL_DATA['tasks_df'] if not GLOBAL_DATA['tasks_df'].empty else None

        new_tasks = generate_inspection_tasks(
            df, district=district, batch_name=batch_name,
            rules=rules_data, max_tasks_per_batch=max_tasks,
            exclude_completed=True, existing_tasks_df=existing_tasks
        )

        if new_tasks.empty:
            status_msg = dbc.Alert('未生成新任务，可能所有管段都已有任务或无满足条件的数据',
                                   color='info', duration=5000)
            return status_msg, new_signal

        if GLOBAL_DATA['tasks_df'].empty:
            GLOBAL_DATA['tasks_df'] = new_tasks
        else:
            GLOBAL_DATA['tasks_df'] = pd.concat(
                [GLOBAL_DATA['tasks_df'], new_tasks], ignore_index=True
            )

        gen_count = len(new_tasks)
        type_count = new_tasks['任务类型'].value_counts().to_dict()
        type_str = '、'.join([f'{k}:{v}' for k, v in type_count.items()])
        urgent_count = len(new_tasks[new_tasks['动态优先级'] == '紧急'])
        high_count = len(new_tasks[new_tasks['动态优先级'] == '高'])

        status_parts = [
            html.Div(f'✅ 成功生成 {gen_count} 个任务！',
                     style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            html.Div(f'任务类型: {type_str}'),
            html.Div(f'其中 紧急: {urgent_count} 个, 高优先级: {high_count} 个'),
        ]

        if assignee:
            pending_ids = new_tasks[new_tasks['任务状态'] == '待派发']['任务编号'].tolist()
            if pending_ids:
                GLOBAL_DATA['tasks_df'] = batch_assign_tasks(
                    GLOBAL_DATA['tasks_df'], pending_ids, assignee
                )
                status_parts.append(html.Div(
                    f'已自动派发给: {assignee}（共 {len(pending_ids)} 个任务）',
                    style={'marginTop': '5px', 'color': '#2980b9'}
                ))

        status_msg = dbc.Alert(status_parts, color='success', duration=8000)
        return status_msg, new_signal

    elif triggered == 'btn-refresh-task-status':
        if GLOBAL_DATA['tasks_df'].empty:
            return dbc.Alert('暂无任务数据，请先生成任务', color='info', duration=4000), dash.no_update

        before_overdue = len(GLOBAL_DATA['tasks_df'][GLOBAL_DATA['tasks_df']['是否超期'] == '是'])
        GLOBAL_DATA['tasks_df'] = refresh_all_task_status(GLOBAL_DATA['tasks_df'])
        after_overdue = len(GLOBAL_DATA['tasks_df'][GLOBAL_DATA['tasks_df']['是否超期'] == '是'])
        new_overdue = after_overdue - before_overdue

        total = len(GLOBAL_DATA['tasks_df'])
        pending = len(GLOBAL_DATA['tasks_df'][GLOBAL_DATA['tasks_df']['任务状态编码'] == 'PENDING'])
        in_progress = len(GLOBAL_DATA['tasks_df'][GLOBAL_DATA['tasks_df']['任务状态编码'] == 'IN_PROGRESS'])
        closed = len(GLOBAL_DATA['tasks_df'][GLOBAL_DATA['tasks_df']['任务状态编码'] == 'CLOSED'])

        parts = [
            html.Div('📊 任务状态已刷新！', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            html.Div(f'任务总数: {total} | 待派发: {pending} | 处理中: {in_progress} | 已闭环: {closed}'),
            html.Div(f'超期任务: {after_overdue} 个' + (f'（新增 {new_overdue} 个超期预警）' if new_overdue > 0 else ''))
        ]
        if after_overdue > 0:
            status_msg = dbc.Alert(parts, color='warning', duration=6000)
        else:
            status_msg = dbc.Alert(parts, color='info', duration=5000)

        return status_msg, new_signal

    return '', dash.no_update


@callback(
    [Output('task-stat-cards', 'children'),
     Output('task-status-chart', 'figure'),
     Output('task-list-table', 'children'),
     Output('task-completion-chart', 'figure'),
     Output('task-dredging-effect-chart', 'figure')],
    [Input('task-signal', 'data'),
     Input('task-status-filter', 'value'),
     Input('task-type-filter', 'value'),
     Input('task-priority-filter', 'value'),
     Input('save-signal', 'data'),
     Input('risk-rules-store', 'data')]
)
def render_task_dashboard(task_signal, status_filter, type_filter, priority_filter,
                          save_signal, rules_data):
    tasks_df = GLOBAL_DATA['tasks_df']

    if tasks_df is None or tasks_df.empty:
        empty_stats = dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div('任务总数', style={'fontSize': '12px', 'color': '#7f8c8d'}),
                    html.Div('0', style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#2c3e50'})
                ], style={'textAlign': 'center', 'padding': '10px', 'background': '#f8f9fa', 'borderRadius': '8px'})
            ], xs=6, md=3),
            dbc.Col([
                html.Div([
                    html.Div('待处理', style={'fontSize': '12px', 'color': '#7f8c8d'}),
                    html.Div('0', style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#e67e22'})
                ], style={'textAlign': 'center', 'padding': '10px', 'background': '#fef9e7', 'borderRadius': '8px'})
            ], xs=6, md=3),
            dbc.Col([
                html.Div([
                    html.Div('已完成', style={'fontSize': '12px', 'color': '#7f8c8d'}),
                    html.Div('0', style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#27ae60'})
                ], style={'textAlign': 'center', 'padding': '10px', 'background': '#eafaf1', 'borderRadius': '8px'})
            ], xs=6, md=3),
            dbc.Col([
                html.Div([
                    html.Div('闭环率', style={'fontSize': '12px', 'color': '#7f8c8d'}),
                    html.Div('0%', style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#8e44ad'})
                ], style={'textAlign': 'center', 'padding': '10px', 'background': '#f5eef8', 'borderRadius': '8px'})
            ], xs=6, md=3),
        ])
        return (
            empty_stats,
            make_empty_fig('暂无任务数据，请先生成任务'),
            make_empty_msg('暂无任务数据，请点击上方"自动生成任务"按钮创建任务'),
            make_empty_fig('暂无完成率数据'),
            make_empty_fig('暂无清淤效果数据')
        )

    if not tasks_df.empty:
        tasks_df = refresh_all_task_status(tasks_df.copy())

    stats = calculate_task_statistics(tasks_df, rules=rules_data)

    def _make_stat_card(title, value, color, bg):
        if isinstance(value, float) and 0 <= value <= 1:
            display_val = f'{value * 100:.1f}%'
        else:
            display_val = str(value)
        return dbc.Col([
            html.Div([
                html.Div(title, style={'fontSize': '12px', 'color': '#7f8c8d'}),
                html.Div(display_val, style={'fontSize': '22px', 'fontWeight': 'bold', 'color': color})
            ], style={'textAlign': 'center', 'padding': '10px', 'background': bg, 'borderRadius': '8px'})
        ], xs=6, md=3)

    stat_cards = dbc.Row([
        _make_stat_card('任务总数', stats.get('任务总数', 0), '#2c3e50', '#f8f9fa'),
        _make_stat_card('待处理', stats.get('待处理任务数', 0), '#e67e22', '#fef9e7'),
        _make_stat_card('已完成', stats.get('已完成含超期', 0), '#27ae60', '#eafaf1'),
        _make_stat_card('闭环率', stats.get('闭环完成率', 0), '#8e44ad', '#f5eef8'),
    ])

    status_chart = create_task_status_chart(stats, tasks_df)
    completion_chart = create_task_completion_chart(stats)
    dredge_effect_chart = create_dredging_effect_summary_chart(stats)

    filtered_df = tasks_df.copy()
    if status_filter and status_filter != 'ALL':
        filtered_df = filtered_df[filtered_df['任务状态'] == status_filter]
    if type_filter and type_filter != 'ALL':
        filtered_df = filtered_df[filtered_df['任务类型'] == type_filter]
    if priority_filter and priority_filter != 'ALL':
        filtered_df = filtered_df[filtered_df['动态优先级'] == priority_filter]

    if filtered_df.empty:
        task_table = make_empty_msg('当前筛选条件下无任务数据')
    else:
        priority_order = {'紧急': 0, '高': 1, '中': 2, '低': 3}
        filtered_df['_sort'] = filtered_df['动态优先级'].map(priority_order)
        filtered_df = filtered_df.sort_values(['_sort', '优先级评分'], ascending=[True, False])
        filtered_df = filtered_df.drop(columns=['_sort'])

        display_df = filtered_df.copy()
        for col in display_df.columns:
            if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M')
            elif col in ['最新淤积率', '增长率', '处理后淤积率', '整改前淤积率']:
                display_df[col] = display_df[col].apply(
                    lambda x: f'{x * 100:.1f}%' if pd.notna(x) and x != '' else ''
                )

        status_color_map = {
            '待派发': '#95a5a6',
            '已派发': '#3498db',
            '处理中': '#f39c12',
            '已完成': '#27ae60',
            '已超期': '#e74c3c',
            '已闭环': '#8e44ad'
        }
        priority_color_map = {'紧急': '#c0392b', '高': '#e74c3c', '中': '#f39c12', '低': '#27ae60'}

        style_conditions = [{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}]
        for status, color in status_color_map.items():
            style_conditions.append({
                'if': {'filter_query': f'{{任务状态}} = "{status}"', 'column_id': '任务状态'},
                'color': color, 'fontWeight': 'bold'
            })
        for priority, color in priority_color_map.items():
            style_conditions.append({
                'if': {'filter_query': f'{{动态优先级}} = "{priority}"', 'column_id': '动态优先级'},
                'color': color, 'fontWeight': 'bold'
            })
        style_conditions.append({
            'if': {'filter_query': '{是否超期} = "是"', 'column_id': '是否超期'},
            'color': '#e74c3c', 'fontWeight': 'bold', 'backgroundColor': '#fdedec'
        })

        display_cols = [
            {"name": "任务编号", "id": "任务编号"},
            {"name": "管段编号", "id": "管段编号"},
            {"name": "片区", "id": "片区"},
            {"name": "任务类型", "id": "任务类型"},
            {"name": "动态优先级", "id": "动态优先级"},
            {"name": "优先级评分", "id": "优先级评分"},
            {"name": "任务状态", "id": "任务状态"},
            {"name": "是否超期", "id": "是否超期"},
            {"name": "超期天数", "id": "超期天数"},
            {"name": "触发原因", "id": "触发原因"},
            {"name": "处理人员", "id": "处理人员"},
            {"name": "派发时间", "id": "派发时间"},
            {"name": "截止时间", "id": "截止时间"},
            {"name": "整改效果评级", "id": "整改效果评级"},
        ]

        task_table = html.Div([
            dash_table.DataTable(
                id='task-datatable',
                data=display_df.to_dict('records'),
                columns=display_cols,
                style_table={'overflowX': 'auto', 'maxHeight': '550px', 'overflowY': 'auto'},
                style_header={
                    'backgroundColor': '#2c3e50', 'color': 'white',
                    'fontWeight': 'bold', 'textAlign': 'center',
                    'position': 'sticky', 'top': 0, 'zIndex': 100
                },
                style_cell={'textAlign': 'left', 'padding': '6px 10px', 'fontSize': '12px'},
                style_data_conditional=style_conditions,
                page_size=15, sort_action='native', filter_action='native',
                export_format='csv', row_selectable='multi',
                selected_rows=[],
                style_cell_conditional=[
                    {'if': {'column_id': '任务编号'}, 'width': '160px'},
                    {'if': {'column_id': '管段编号'}, 'width': '80px'},
                    {'if': {'column_id': '片区'}, 'width': '60px'},
                    {'if': {'column_id': '任务类型'}, 'width': '65px'},
                    {'if': {'column_id': '动态优先级'}, 'width': '80px'},
                    {'if': {'column_id': '优先级评分'}, 'width': '75px'},
                    {'if': {'column_id': '任务状态'}, 'width': '75px'},
                    {'if': {'column_id': '是否超期'}, 'width': '70px'},
                    {'if': {'column_id': '超期天数'}, 'width': '70px'},
                    {'if': {'column_id': '触发原因'}, 'width': '150px'},
                    {'if': {'column_id': '处理人员'}, 'width': '110px'},
                ]
            ),
            html.Div(
                f'共显示 {len(display_df)} 个任务（双击行可查看详情并处理）',
                style={'marginTop': '8px', 'fontSize': '12px', 'color': '#7f8c8d', 'textAlign': 'right'}
            )
        ])

    return stat_cards, status_chart, task_table, completion_chart, dredge_effect_chart


@callback(
    [Output('task-detail-panel', 'children'),
     Output('task-before-after-chart', 'figure'),
     Output('selected-tasks-store', 'data')],
    [Input('task-datatable', 'active_cell'),
     Input('task-datatable', 'selected_rows'),
     Input('task-signal', 'data'),
     Input('task-status-filter', 'value'),
     Input('task-type-filter', 'value'),
     Input('task-priority-filter', 'value')],
    [State('task-datatable', 'data'),
     State('selected-tasks-store', 'data')],
    prevent_initial_call=True
)
def handle_task_selection(active_cell, selected_rows, task_signal,
                          status_filter, type_filter, priority_filter,
                          table_data, current_selected):
    tasks_df = GLOBAL_DATA['tasks_df']
    ctx_triggered = ctx.triggered_id

    selected_task_ids = []
    if selected_rows and table_data:
        for idx in selected_rows:
            if idx < len(table_data):
                selected_task_ids.append(table_data[idx].get('任务编号', ''))

    if tasks_df is None or tasks_df.empty:
        return (
            make_empty_msg('请先生成任务后再选择'),
            make_empty_fig('暂无任务数据'),
            selected_task_ids
        )

    tasks_df = refresh_all_task_status(tasks_df.copy())

    selected_task_id = None
    if active_cell and table_data:
        row_idx = active_cell.get('row', 0)
        if row_idx < len(table_data):
            selected_task_id = table_data[row_idx].get('任务编号', '')
            GLOBAL_DATA['selected_task_id'] = selected_task_id
    elif GLOBAL_DATA.get('selected_task_id'):
        selected_task_id = GLOBAL_DATA['selected_task_id']

    comparison_fig = make_empty_fig('请选择已完成的清淤/复检任务查看整改对比')
    if selected_task_id:
        comparison_data = compare_before_after(tasks_df, selected_task_id)
        comparison_fig = create_before_after_comparison_chart(comparison_data)

    if not selected_task_id:
        return (
            make_empty_msg('请从任务清单中点击选择一个任务查看详情并处理'),
            comparison_fig,
            selected_task_ids
        )

    task_row = tasks_df[tasks_df['任务编号'] == selected_task_id]
    if task_row.empty:
        return (
            make_empty_msg('未找到该任务数据'),
            comparison_fig,
            selected_task_ids
        )

    task = task_row.iloc[0]
    status = task['任务状态']
    status_code = task['任务状态编码']

    priority_color = task.get('优先级颜色', '#95a5a6')
    status_colors = {
        '待派发': '#95a5a6', '已派发': '#3498db', '处理中': '#f39c12',
        '已完成': '#27ae60', '已超期': '#e74c3c', '已闭环': '#8e44ad'
    }
    status_color = status_colors.get(status, '#95a5a6')

    def _fmt_date(val):
        if pd.isna(val) or val is None or val == '':
            return '-'
        if isinstance(val, str):
            return val
        return val.strftime('%Y-%m-%d %H:%M')

    def _fmt_rate(val):
        if pd.isna(val) or val is None or val == '':
            return '-'
        return f'{float(val) * 100:.1f}%'

    def _fmt_num(val):
        if pd.isna(val) or val is None or val == '':
            return '-'
        return str(val)

    detail_sections = []
    detail_sections.append(html.Div([
        dbc.Row([
            dbc.Col([
                html.H6(f"📋 任务 {task['任务编号']}",
                        style={'fontWeight': 'bold', 'marginBottom': '10px', 'color': '#2c3e50'})
            ], md=8),
            dbc.Col([
                html.Div([
                    html.Span(status, style={
                        'background': status_color, 'color': 'white',
                        'padding': '4px 12px', 'borderRadius': '12px',
                        'fontSize': '12px', 'fontWeight': 'bold'
                    }),
                    html.Span(f" {task['动态优先级']}", style={
                        'background': priority_color, 'color': 'white',
                        'padding': '4px 12px', 'borderRadius': '12px',
                        'fontSize': '12px', 'fontWeight': 'bold', 'marginLeft': '8px'
                    })
                ], style={'textAlign': 'right'})
            ], md=4)
        ]),
        html.Hr(style={'margin': '10px 0'})
    ]))

    detail_sections.append(html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label('管段/片区信息', style={'fontWeight': 'bold', 'fontSize': '13px', 'color': '#2980b9'}),
                    html.Ul([
                        html.Li(f"管段编号: {task['管段编号']}"),
                        html.Li(f"片区: {task['片区']}"),
                        html.Li(f"管径: {_fmt_num(task['管径(mm)'])} mm"),
                    ], style={'fontSize': '12px', 'paddingLeft': '20px'})
                ])
            ], md=4),
            dbc.Col([
                html.Div([
                    html.Label('淤积状态', style={'fontWeight': 'bold', 'fontSize': '13px', 'color': '#e67e22'}),
                    html.Ul([
                        html.Li(f"最新淤积率: {_fmt_rate(task['最新淤积率'])}"),
                        html.Li(f"淤积深度: {_fmt_num(task['最新淤积深度(mm)'])} mm"),
                        html.Li(f"增长率: {_fmt_rate(task['增长率'])} {'⚠️' if task['异常增长'] == '是' else ''}"),
                    ], style={'fontSize': '12px', 'paddingLeft': '20px'})
                ])
            ], md=4),
            dbc.Col([
                html.Div([
                    html.Label('任务信息', style={'fontWeight': 'bold', 'fontSize': '13px', 'color': '#8e44ad'}),
                    html.Ul([
                        html.Li(f"任务类型: {task['任务类型']}"),
                        html.Li(f"触发原因: {task['触发原因']}"),
                        html.Li(f"缺失巡检: {'是' if task['巡检缺失'] == '是' else '否'}" +
                                (f" ({task['缺失批次列表']})" if task['缺失批次列表'] else '')),
                    ], style={'fontSize': '12px', 'paddingLeft': '20px'})
                ])
            ], md=4),
        ], style={'marginBottom': '15px'})
    ]))

    detail_sections.append(html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label('处理进度', style={'fontWeight': 'bold', 'fontSize': '13px', 'color': '#27ae60'}),
                    html.Ul([
                        html.Li(f"派发人员: {task['派发人员'] if task['派发人员'] else '未派发'}"),
                        html.Li(f"派发时间: {_fmt_date(task['派发时间'])}"),
                        html.Li(f"处理人员: {task['处理人员'] if task['处理人员'] else '未指派'}"),
                        html.Li(f"开始处理: {_fmt_date(task['处理开始时间'])}"),
                        html.Li(f"处理完成: {_fmt_date(task['处理完成时间'])}"),
                        html.Li(f"闭环确认: {task['闭环确认人'] if task['闭环确认人'] else '未闭环'} ({_fmt_date(task['闭环时间'])})"),
                    ], style={'fontSize': '12px', 'paddingLeft': '20px'})
                ])
            ], md=6),
            dbc.Col([
                html.Div([
                    html.Label('时间与超期', style={'fontWeight': 'bold', 'fontSize': '13px', 'color': '#c0392b'}),
                    html.Ul([
                        html.Li(f"创建时间: {_fmt_date(task['创建时间'])}"),
                        html.Li([
                            '截止时间: ',
                            html.Span(_fmt_date(task['截止时间']), style={
                                'color': '#e74c3c', 'fontWeight': 'bold'
                            } if task['是否超期'] == '是' else {})
                        ]),
                        html.Li([
                            '是否超期: ',
                            html.Span(task['是否超期'] + (f" ({task['超期天数']}天)" if task['超期天数'] > 0 else ''),
                                      style={'color': '#e74c3c', 'fontWeight': 'bold'} if task['是否超期'] == '是' else {})
                        ]),
                    ], style={'fontSize': '12px', 'paddingLeft': '20px'})
                ])
            ], md=6),
        ], style={'marginBottom': '15px'})
    ]))

    if status_code in ['COMPLETED', 'CLOSED']:
        result_parts = [
            html.Li(f"处理结果: {task['处理结果'] if task['处理结果'] else '未填写'}"),
            html.Li(f"处理备注: {task['处理备注'] if task['处理备注'] else '无'}"),
        ]
        if task['任务类型编码'] == 'DREDGING':
            result_parts.append(html.Li(f"处理后淤积深度: {_fmt_num(task['处理后淤积深度(mm)'])} mm"))
            result_parts.append(html.Li(f"处理后淤积率: {_fmt_rate(task['处理后淤积率'])}"))
            effect = task['整改效果评级']
            effect_colors = {'显著有效': '#27ae60', '部分有效': '#f39c12',
                             '效果不明显': '#e67e22', '淤积加重': '#e74c3c'}
            if effect:
                result_parts.append(html.Li([
                    '整改效果评级: ',
                    html.Span(effect, style={
                        'color': effect_colors.get(effect, '#95a5a6'), 'fontWeight': 'bold'
                    })
                ]))
        detail_sections.append(html.Div([
            html.Label('处置结果', style={'fontWeight': 'bold', 'fontSize': '13px', 'color': '#16a085'}),
            html.Ul(result_parts, style={'fontSize': '12px', 'paddingLeft': '20px'})
        ], style={'marginBottom': '15px'}))

    action_forms = []

    if status_code == 'PENDING':
        action_forms.append(html.Div([
            html.Hr(style={'margin': '15px 0'}),
            html.Label('📤 派发任务', style={'fontWeight': 'bold', 'color': '#2980b9', 'marginBottom': '8px'}),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='assign-person-select',
                        options=[
                            {'label': '巡检组A - 张工', 'value': '张工(巡检组A)'},
                            {'label': '巡检组B - 李工', 'value': '李工(巡检组B)'},
                            {'label': '清淤组1 - 王队长', 'value': '王队长(清淤组1)'},
                            {'label': '清淤组2 - 赵队长', 'value': '赵队长(清淤组2)'},
                            {'label': '机动组 - 刘工', 'value': '刘工(机动组)'}
                        ],
                        placeholder='选择派发人员',
                        clearable=False,
                        value=task['处理人员'] if task['处理人员'] else None
                    )
                ], md=8),
                dbc.Col([
                    dbc.Button('确认派发', id='btn-assign-single',
                               color='primary', size='md', style={'width': '100%'})
                ], md=4)
            ])
        ]))
    elif status_code in ['ASSIGNED', 'OVERDUE']:
        action_forms.append(html.Div([
            html.Hr(style={'margin': '15px 0'}),
            html.Label('▶️ 开始处理任务',
                       style={'fontWeight': 'bold', 'color': '#f39c12', 'marginBottom': '8px'}),
            dbc.Button('开始处理此任务', id='btn-start-single',
                       color='warning', size='md')
        ]))
    elif status_code == 'IN_PROGRESS':
        is_dredging = task['任务类型编码'] == 'DREDGING'
        result_options = [
            {'label': '巡检正常/无异常', 'value': '巡检正常'},
            {'label': '发现问题已记录', 'value': '发现问题'},
            {'label': '已完成清淤作业', 'value': '清淤完成'},
            {'label': '复检合格', 'value': '复检合格'},
            {'label': '复检不合格', 'value': '复检不合格'},
        ]
        action_forms.append(html.Div([
            html.Hr(style={'margin': '15px 0'}),
            html.Label('✅ 完成任务处理 - 回填处置结果',
                       style={'fontWeight': 'bold', 'color': '#27ae60', 'marginBottom': '10px'}),
            dbc.Row([
                dbc.Col([
                    html.Label('处理结果:', style={'fontSize': '12px', 'marginBottom': '3px'}),
                    dcc.Dropdown(id='complete-result-select', options=result_options,
                                 placeholder='选择处理结果', clearable=False)
                ], md=4),
                dbc.Col([
                    html.Label('处理后淤积深度(mm):' if is_dredging else '备注淤积深度(mm):',
                               style={'fontSize': '12px', 'marginBottom': '3px'}),
                    dcc.Input(id='complete-post-depth', type='number',
                              placeholder='输入淤积深度mm', min=0,
                              style={'width': '100%', 'height': '36px'})
                ], md=4) if is_dredging else dbc.Col([
                    html.Label('处理后淤积率(%):', style={'fontSize': '12px', 'marginBottom': '3px'}),
                    dcc.Input(id='complete-post-rate', type='number',
                              placeholder='0-100%', min=0, max=100,
                              style={'width': '100%', 'height': '36px'})
                ], md=4),
                dbc.Col([
                    html.Label('处理备注:', style={'fontSize': '12px', 'marginBottom': '3px'}),
                    dbc.Input(id='complete-note', placeholder='输入备注信息', type='text')
                ], md=4),
            ], style={'marginBottom': '10px'}),
            dbc.Row([
                dbc.Col([
                    dbc.Button('提交处理结果', id='btn-complete-single',
                               color='success', size='md')
                ], md=4)
            ])
        ]))
    elif status_code in ['COMPLETED']:
        action_forms.append(html.Div([
            html.Hr(style={'margin': '15px 0'}),
            html.Label('🔒 闭环确认', style={'fontWeight': 'bold', 'color': '#8e44ad', 'marginBottom': '8px'}),
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='close-confirm-person', placeholder='输入闭环确认人',
                              type='text', value='系统管理员')
                ], md=6),
                dbc.Col([
                    dbc.Button('确认闭环此任务', id='btn-close-single',
                               color='secondary', size='md')
                ], md=6)
            ])
        ]))
    elif status_code == 'CLOSED':
        action_forms.append(html.Div([
            html.Hr(style={'margin': '15px 0'}),
            dbc.Alert([
                html.I(className='fas fa-check-circle', style={'marginRight': '8px'}),
                '此任务已完成闭环归档，处置流程结束'
            ], color='success')
        ]))

    detail_sections.append(html.Div(action_forms, id='task-action-forms'))

    detail_html = html.Div(detail_sections, style={
        'padding': '5px', 'fontSize': '13px',
        'maxHeight': '550px', 'overflowY': 'auto'
    })

    return detail_html, comparison_fig, selected_task_ids


@callback(
    Output('task-signal', 'data', allow_duplicate=True),
    [Input('btn-assign-single', 'n_clicks'),
     Input('btn-start-single', 'n_clicks'),
     Input('btn-complete-single', 'n_clicks'),
     Input('btn-close-single', 'n_clicks')],
    [State('assign-person-select', 'value'),
     State('complete-result-select', 'value'),
     State('complete-post-depth', 'value'),
     State('complete-post-rate', 'value'),
     State('complete-note', 'value'),
     State('close-confirm-person', 'value'),
     State('task-signal', 'data'),
     State('risk-rules-store', 'data')],
    prevent_initial_call=True
)
def handle_task_actions(assign_clicks, start_clicks, complete_clicks, close_clicks,
                        assignee, result, post_depth, post_rate, note,
                        confirmer, current_signal, rules_data):
    triggered = ctx.triggered_id
    selected_id = GLOBAL_DATA.get('selected_task_id')
    new_signal = (current_signal + 1) if current_signal else 1

    if not selected_id or GLOBAL_DATA['tasks_df'].empty:
        return dash.no_update

    tasks_df = GLOBAL_DATA['tasks_df'].copy()

    if triggered == 'btn-assign-single' and assign_clicks:
        if not assignee:
            return dash.no_update
        tasks_df = assign_task(tasks_df, selected_id, assignee)
    elif triggered == 'btn-start-single' and start_clicks:
        tasks_df = start_task(tasks_df, selected_id)
    elif triggered == 'btn-complete-single' and complete_clicks:
        if not result:
            return dash.no_update
        actual_post_rate = None
        if post_rate is not None:
            actual_post_rate = float(post_rate) / 100
        tasks_df = complete_task(
            tasks_df, selected_id, result, note or '',
            post_depth=post_depth, post_rate=actual_post_rate,
            rules=rules_data
        )
    elif triggered == 'btn-close-single' and close_clicks:
        tasks_df = close_task(tasks_df, selected_id, confirmer or '系统管理员')
    else:
        return dash.no_update

    GLOBAL_DATA['tasks_df'] = tasks_df
    return new_signal


@callback(
    [Output('task-generate-status', 'children', allow_duplicate=True),
     Output('task-signal', 'data', allow_duplicate=True)],
    [Input('btn-batch-assign', 'n_clicks'),
     Input('btn-batch-close', 'n_clicks'),
     Input('btn-batch-delete', 'n_clicks')],
    [State('selected-tasks-store', 'data'),
     State('task-assignee-select', 'value'),
     State('task-signal', 'data')],
    prevent_initial_call=True
)
def handle_batch_operations(assign_clicks, close_clicks, delete_clicks,
                            selected_ids, assignee, current_signal):
    triggered = ctx.triggered_id
    new_signal = (current_signal + 1) if current_signal else 1

    if not selected_ids:
        return dbc.Alert('请先从任务列表中勾选要操作的任务', color='warning', duration=4000), dash.no_update

    tasks_df = GLOBAL_DATA['tasks_df']
    if tasks_df.empty:
        return dbc.Alert('暂无任务数据', color='warning'), dash.no_update

    valid_ids = [tid for tid in selected_ids if tid and tid in tasks_df['任务编号'].values]
    if not valid_ids:
        return dbc.Alert('未找到有效的任务编号', color='warning', duration=4000), dash.no_update

    tasks_df = tasks_df.copy()
    count = 0

    if triggered == 'btn-batch-assign' and assign_clicks:
        if not assignee:
            return dbc.Alert('请先在上方选择派发人员', color='warning', duration=4000), dash.no_update
        pending_ids = tasks_df[
            (tasks_df['任务编号'].isin(valid_ids)) &
            (tasks_df['任务状态编码'].isin(['PENDING']))
        ]['任务编号'].tolist()
        if pending_ids:
            tasks_df = batch_assign_tasks(tasks_df, pending_ids, assignee)
            count = len(pending_ids)
        msg = f'✅ 已派发 {count} 个待派发任务给 {assignee}'
        if count == 0:
            msg = '所选任务中没有待派发状态的任务'

    elif triggered == 'btn-batch-close' and close_clicks:
        closable_ids = tasks_df[
            (tasks_df['任务编号'].isin(valid_ids)) &
            (tasks_df['任务状态编码'].isin(['COMPLETED', 'OVERDUE']))
        ]['任务编号'].tolist()
        for tid in closable_ids:
            tasks_df = close_task(tasks_df, tid, '批量闭环')
            count += 1
        msg = f'✅ 已批量闭环 {count} 个任务'
        if count == 0:
            msg = '所选任务中没有已完成或已超期状态的任务'

    elif triggered == 'btn-batch-delete' and delete_clicks:
        before = len(tasks_df)
        tasks_df = tasks_df[~tasks_df['任务编号'].isin(valid_ids)].reset_index(drop=True)
        count = before - len(tasks_df)
        msg = f'🗑️ 已删除 {count} 个任务'
        if count == 0:
            msg = '未找到可删除的任务'

    GLOBAL_DATA['tasks_df'] = tasks_df
    return dbc.Alert(msg, color='success' if count > 0 else 'info', duration=5000), new_signal


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
