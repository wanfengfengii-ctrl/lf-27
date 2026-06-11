import io
import base64
import pandas as pd
import numpy as np
from datetime import datetime

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc

from data_processor import (
    standardize_columns,
    validate_and_clean_data,
    get_districts,
    get_batches,
    get_pipe_ids,
    calculate_statistics,
    detect_abnormal_growth,
    get_high_risk_segments,
    detect_missing_inspections
)
from visualizations import (
    create_pipe_history_chart,
    create_pipes_comparison_chart,
    create_risk_heatmap,
    create_risk_distribution_chart,
    create_sediment_trend_chart
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
app.title = '城市雨水管网淤积巡检分析台'
server = app.server

GLOBAL_DATA = {
    'raw_df': None,
    'valid_df': pd.DataFrame(),
    'errors': [],
    'warnings': [],
    'has_district': False
}

HEADER_STYLE = {
    'background': 'linear-gradient(135deg, #2c3e50 0%, #34495e 100%)',
    'padding': '20px 30px',
    'marginBottom': '25px',
    'borderRadius': '8px',
    'boxShadow': '0 4px 12px rgba(0,0,0,0.15)'
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


app.layout = dbc.Container([
    html.Div([
        html.H1('🌧️ 城市雨水管网淤积巡检分析台', style={'color': 'white', 'margin': 0, 'fontSize': '28px'}),
        html.P('市政巡检数据管理与淤积风险分析系统', style={'color': '#bdc3c7', 'margin': '8px 0 0 0', 'fontSize': '14px'})
    ], style=HEADER_STYLE),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className='fas fa-upload', style={'marginRight': '8px'}),
                    '数据导入'
                ], style={'fontWeight': 'bold', 'fontSize': '16px', 'background': '#ecf0f1'}),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className='fas fa-cloud-upload-alt', style={'fontSize': '36px', 'color': '#3498db', 'marginBottom': '10px'}),
                            html.Br(),
                            '拖拽文件到此 或 ',
                            html.A('点击选择文件', style={'color': '#3498db', 'textDecoration': 'underline', 'cursor': 'pointer'}),
                            html.Br(),
                            html.Small('支持 CSV、Excel 格式', style={'color': '#7f8c8d'})
                        ], style={'textAlign': 'center', 'padding': '40px 20px'}),
                        style={
                            'width': '100%',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '8px',
                            'borderColor': '#bdc3c7',
                            'backgroundColor': '#fafafa',
                            'cursor': 'pointer'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-status', style={'marginTop': '15px'}),
                    html.Div(id='file-info', style={'marginTop': '10px', 'fontSize': '13px', 'color': '#7f8c8d'})
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className='fas fa-exclamation-triangle', style={'marginRight': '8px'}),
                    '导入问题报告'
                ], style={'fontWeight': 'bold', 'fontSize': '16px', 'background': '#fdf2e9'}),
                dbc.CardBody([
                    html.Div(id='import-report', children=[
                        html.Div('尚未导入数据', style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '30px'})
                    ])
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ]),
    
    html.Div(id='main-content', style={'display': 'none'}, children=[
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-filter', style={'marginRight': '8px'}),
                        '筛选条件'
                    ], style={'fontWeight': 'bold', 'fontSize': '16px', 'background': '#eaf2f8'}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label('选择片区:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Dropdown(id='district-filter', placeholder='全部片区', clearable=True)
                            ], md=3),
                            dbc.Col([
                                html.Label('巡检批次:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Dropdown(id='batch-filter', placeholder='全部批次', multi=True)
                            ], md=6),
                            dbc.Col([
                                html.Label('📊 数据概览:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                html.Div(id='data-summary', style={'fontSize': '13px', 'padding': '6px 10px', 'background': '#f8f9fa', 'borderRadius': '4px'})
                            ], md=3)
                        ])
                    ])
                ], style=CARD_STYLE)
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-chart-pie', style={'marginRight': '8px'}),
                        '统计概览'
                    ], style={'fontWeight': 'bold', 'fontSize': '16px', 'background': '#e8f8f5'}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Div('记录总数', style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-total', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': '#2c3e50'})
                                ], style={'textAlign': 'center', 'padding': '15px', 'background': '#f8f9fa', 'borderRadius': '8px'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.Div('管段数量', style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-pipes', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': '#2980b9'})
                                ], style={'textAlign': 'center', 'padding': '15px', 'background': '#ebf5fb', 'borderRadius': '8px'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.Div('最大淤积深度(mm)', style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-max-depth', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': '#8e44ad'})
                                ], style={'textAlign': 'center', 'padding': '15px', 'background': '#f5eef8', 'borderRadius': '8px'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.Div('平均淤积率', style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-avg-rate', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': '#d35400'})
                                ], style={'textAlign': 'center', 'padding': '15px', 'background': '#fef5e7', 'borderRadius': '8px'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.Div('高风险管段', style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-high-risk', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': '#c0392b'})
                                ], style={'textAlign': 'center', 'padding': '15px', 'background': '#fdedec', 'borderRadius': '8px'})
                            ], md=2),
                            dbc.Col([
                                html.Div([
                                    html.Div('异常增长管段', style={'fontSize': '13px', 'color': '#7f8c8d'}),
                                    html.Div(id='stat-abnormal', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': '#e67e22'})
                                ], style={'textAlign': 'center', 'padding': '15px', 'background': '#fef9e7', 'borderRadius': '8px'})
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
                                        html.Label('选择管段编号:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(id='pipe-select', placeholder='请选择管段编号...')
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
                                        html.Label('选择对比管段 (可多选):', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(id='pipes-compare-select', multi=True, placeholder='请选择多个管段进行对比...')
                                    ], md=6),
                                    dbc.Col([
                                        html.Label('对比指标:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id='compare-by',
                                            options=[
                                                {'label': '淤积率 (%)', 'value': '淤积率'},
                                                {'label': '淤积深度 (mm)', 'value': '淤积深度'}
                                            ],
                                            value='淤积率',
                                            clearable=False
                                        )
                                    ], md=3)
                                ], style={'marginBottom': '20px'}),
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
                            dbc.CardBody([
                                dcc.Graph(id='trend-chart')
                            ])
                        ], style=CARD_STYLE)
                    ], md=7),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='risk-distribution-chart')
                            ])
                        ], style=CARD_STYLE)
                    ], md=5)
                ])
            ]),
            
            dbc.Tab(label='⚠️ 风险与异常', tab_id='tab-risk', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.I(className='fas fa-exclamation-circle', style={'marginRight': '8px', 'color': '#c0392b'}),
                                '高风险区段列表（淤积率 ≥ 60%）'
                            ], style={'fontWeight': 'bold', 'background': '#fdedec'}),
                            dbc.CardBody([
                                html.Div(id='high-risk-table')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.I(className='fas fa-arrow-up', style={'marginRight': '8px', 'color': '#e67e22'}),
                                '异常增长管段（增长率 ≥ 20%）'
                            ], style={'fontWeight': 'bold', 'background': '#fef9e7'}),
                            dbc.CardBody([
                                html.Div(id='abnormal-growth-table')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.I(className='fas fa-calendar-times', style={'marginRight': '8px', 'color': '#7f8c8d'}),
                                '缺失巡检记录'
                            ], style={'fontWeight': 'bold', 'background': '#f2f3f4'}),
                            dbc.CardBody([
                                html.Div(id='missing-inspections-table')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ]),
            
            dbc.Tab(label='📋 数据明细', tab_id='tab-detail', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(id='detail-table')
                            ])
                        ], style=CARD_STYLE)
                    ], width=12)
                ])
            ])
        ], id='main-tabs', active_tab='tab-single', style={'marginBottom': '20px'})
    ]),
    
    html.Footer([
        html.Hr(),
        html.Div('城市雨水管网淤积巡检分析台 | Python + Dash', style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px'})
    ])
    
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
     Output('pipes-compare-select', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_upload(contents, filename):
    if contents is None:
        return (
            '',
            '',
            html.Div('尚未导入数据', style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '30px'}),
            {'display': 'none'},
            [], None, [], [], []
        )
    
    result = parse_contents(contents, filename)
    
    if len(result) == 2:
        _, msg = result
        return (
            dbc.Alert(msg, color='danger'),
            '',
            html.Div('尚未导入数据', style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '30px'}),
            {'display': 'none'},
            [], None, [], [], []
        )
    
    valid_df, errors, warnings, has_district, raw_df = result
    GLOBAL_DATA['valid_df'] = valid_df
    GLOBAL_DATA['errors'] = errors
    GLOBAL_DATA['warnings'] = warnings
    GLOBAL_DATA['has_district'] = has_district
    GLOBAL_DATA['raw_df'] = raw_df
    
    upload_status = dbc.Alert([
        html.I(className='fas fa-check-circle', style={'marginRight': '8px'}),
        f'成功导入 {len(valid_df)} 条有效记录'
    ], color='success')
    
    file_info = f'文件: {filename} | 原始记录: {len(raw_df)} 条 | 有效记录: {len(valid_df)} 条 | 错误: {len(errors)} 条'
    
    report_parts = []
    if warnings:
        report_parts.append(
            html.Div([
                html.H6('⚠️ 警告:', style={'color': '#e67e22', 'marginBottom': '10px'}),
                html.Ul([html.Li(w, style={'color': '#d35400'}) for w in warnings])
            ], style={'marginBottom': '15px'})
        )
    
    if errors:
        error_df = pd.DataFrame(errors)
        report_parts.append(
            html.Div([
                html.H6(f'❌ 共 {len(errors)} 条记录导入失败:', style={'color': '#c0392b', 'marginBottom': '10px'}),
                dash_table.DataTable(
                    data=error_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in error_df.columns],
                    style_table={'overflowX': 'auto', 'maxHeight': '300px', 'overflowY': 'auto'},
                    style_header={'backgroundColor': '#f8d7da', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                    style_data_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#fdf2f2'
                    }],
                    page_size=10
                )
            ])
        )
    
    if not errors and not warnings:
        report_parts.append(html.Div('✅ 所有数据校验通过，无异常！', style={'color': '#27ae60', 'textAlign': 'center', 'padding': '20px', 'fontSize': '16px'}))
    
    import_report = html.Div(report_parts) if report_parts else html.Div('✅ 所有数据校验通过', style={'textAlign': 'center', 'color': '#27ae60', 'padding': '20px'})
    
    districts = get_districts(valid_df)
    district_options = [{'label': d, 'value': d} for d in districts]
    
    batches = get_batches(valid_df)
    batch_options = [{'label': b, 'value': b} for b in batches]
    
    pipes = get_pipe_ids(valid_df)
    pipe_options = [{'label': p, 'value': p} for p in pipes]
    
    return (
        upload_status,
        file_info,
        import_report,
        {'display': 'block'},
        district_options,
        None,
        batch_options,
        pipe_options,
        pipe_options
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
    
    stats = calculate_statistics(df, district, batches)
    abnormal = detect_abnormal_growth(df, district)
    
    summary = f"共 {stats.get('记录总数', 0)} 条记录 / {stats.get('管段数量', 0)} 个管段 / {stats.get('巡检批次数量', 0)} 个批次"
    
    return (
        batches,
        summary,
        stats.get('记录总数', 0),
        stats.get('管段数量', 0),
        stats.get('最大淤积深度', '-'),
        f"{stats.get('平均淤积率', 0) * 100:.1f}%" if stats.get('平均淤积率') is not None else '-',
        stats.get('高风险管段数', 0),
        len(abnormal),
        None,
        None
    )


@callback(
    [Output('stat-total', 'children', allow_duplicate=True),
     Output('stat-pipes', 'children', allow_duplicate=True),
     Output('stat-max-depth', 'children', allow_duplicate=True),
     Output('stat-avg-rate', 'children', allow_duplicate=True),
     Output('stat-high-risk', 'children', allow_duplicate=True),
     Output('stat-abnormal', 'children', allow_duplicate=True),
     Output('data-summary', 'children', allow_duplicate=True)],
    [Input('batch-filter', 'value')],
    [State('district-filter', 'value')],
    prevent_initial_call=True
)
def update_on_batch_change(selected_batches, district):
    if GLOBAL_DATA['valid_df'].empty:
        return '-', '-', '-', '-', '-', '-', '无数据'
    
    df = GLOBAL_DATA['valid_df']
    batches = selected_batches if selected_batches else get_batches(df, district)
    
    stats = calculate_statistics(df, district, batches)
    abnormal = detect_abnormal_growth(df, district)
    
    summary = f"共 {stats.get('记录总数', 0)} 条记录 / {stats.get('管段数量', 0)} 个管段 / {len(batches)} 个批次"
    
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
     Input('district-filter', 'value')]
)
def update_pipe_history(pipe_id, district):
    if GLOBAL_DATA['valid_df'].empty or not pipe_id:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title='请选择管段编号查看淤积过程',
            template='plotly_white',
            height=600
        )
        return fig
    
    return create_pipe_history_chart(GLOBAL_DATA['valid_df'], pipe_id, district)


@callback(
    Output('pipes-comparison-chart', 'figure'),
    [Input('pipes-compare-select', 'value'),
     Input('compare-by', 'value'),
     Input('district-filter', 'value')]
)
def update_pipes_comparison(pipe_ids, compare_by, district):
    if GLOBAL_DATA['valid_df'].empty or not pipe_ids:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title='请选择至少一个管段进行对比分析',
            template='plotly_white',
            height=500
        )
        return fig
    
    return create_pipes_comparison_chart(GLOBAL_DATA['valid_df'], pipe_ids, district, compare_by)


@callback(
    Output('risk-heatmap', 'figure'),
    [Input('district-filter', 'value'),
     Input('batch-filter', 'value')]
)
def update_risk_heatmap(district, batches):
    if GLOBAL_DATA['valid_df'].empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(title='暂无数据', template='plotly_white')
        return fig
    
    return create_risk_heatmap(GLOBAL_DATA['valid_df'], district, batches)


@callback(
    [Output('trend-chart', 'figure'),
     Output('risk-distribution-chart', 'figure')],
    [Input('district-filter', 'value'),
     Input('batch-filter', 'value')]
)
def update_trend_and_distribution(district, batches):
    if GLOBAL_DATA['valid_df'].empty:
        import plotly.graph_objects as go
        empty_fig = go.Figure()
        empty_fig.update_layout(title='暂无数据', template='plotly_white')
        return empty_fig, empty_fig
    
    trend_fig = create_sediment_trend_chart(GLOBAL_DATA['valid_df'], district, batches)
    stats = calculate_statistics(GLOBAL_DATA['valid_df'], district, batches)
    dist_fig = create_risk_distribution_chart(stats)
    
    return trend_fig, dist_fig


@callback(
    [Output('high-risk-table', 'children'),
     Output('abnormal-growth-table', 'children'),
     Output('missing-inspections-table', 'children'),
     Output('detail-table', 'children')],
    [Input('district-filter', 'value'),
     Input('batch-filter', 'value')]
)
def update_tables(district, batches):
    if GLOBAL_DATA['valid_df'].empty:
        empty_msg = html.Div('暂无数据', style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '20px'})
        return empty_msg, empty_msg, empty_msg, empty_msg
    
    high_risk_df = get_high_risk_segments(GLOBAL_DATA['valid_df'], district, batches)
    abnormal_df = detect_abnormal_growth(GLOBAL_DATA['valid_df'], district)
    missing_df = detect_missing_inspections(GLOBAL_DATA['valid_df'], district)
    
    detail_df = GLOBAL_DATA['valid_df'].copy()
    if district:
        detail_df = detail_df[detail_df['片区'] == district]
    if batches:
        detail_df = detail_df[detail_df['巡检批次'].isin(batches)]
    detail_df = detail_df.sort_values(['片区', '管段编号', '检查时间'])
    
    def create_datatable(df, columns_config=None, title_prefix=''):
        if df.empty:
            return html.Div(f'暂无{title_prefix}数据', style={'textAlign': 'center', 'color': '#95a5a6', 'padding': '20px'})
        
        display_df = df.copy()
        for col in display_df.columns:
            if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')
        
        if columns_config:
            columns = columns_config
        else:
            columns = [{"name": i, "id": i} for i in display_df.columns]
        
        return dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=columns,
            style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'auto'},
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={'textAlign': 'left', 'padding': '8px 12px', 'fontSize': '13px'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                }
            ],
            page_size=15,
            sort_action='native',
            filter_action='native',
            export_format='csv'
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
    
    detail_cols = [
        {"name": "管段编号", "id": "管段编号"},
        {"name": "片区", "id": "片区"},
        {"name": "巡检批次", "id": "巡检批次"},
        {"name": "检查时间", "id": "检查时间"},
        {"name": "淤积深度(mm)", "id": "淤积深度"},
        {"name": "管径(mm)", "id": "管径"},
        {"name": "淤积率", "id": "淤积率", "type": "numeric", "format": {"specifier": ".1%"}},
        {"name": "备注", "id": "备注"}
    ]
    
    return (
        create_datatable(high_risk_df, high_risk_cols, '高风险'),
        create_datatable(abnormal_df, abnormal_cols, '异常增长'),
        create_datatable(missing_df, missing_cols, '缺失巡检'),
        create_datatable(detail_df, detail_cols, '明细')
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
