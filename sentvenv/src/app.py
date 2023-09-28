import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
from main import predict_sentiment, model_gru, tokenizer


app = dash.Dash(__name__)
# server = app.server

app.layout = html.Div(
    children=[
        html.H1("Modi-Meter"),
        dcc.Textarea(id='sentence-input', placeholder="Please enter your views on PM Modi. For reusing the tool press the Reset button", rows=4, cols=50),
        html.Button('Reset', id='reset-button', n_clicks=0),
        html.Button('Submit', id='submit-button', n_clicks=0),
        html.Div(id='prediction-output')
    ]
)

@app.callback(
    [Output("sentence-input", "value"),
     Output("prediction-output", "children")],
    [Input("submit-button", "n_clicks"),
     Input("reset-button", "n_clicks")],
    [State("sentence-input", "value")]
)
def sentiment_analysis(submit_clicks, reset_clicks, sentence):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'submit-button' and sentence is not None:
        sentiment_prediction = predict_sentiment(model_gru, tokenizer, sentence)
        return dash.no_update, html.Div(sentiment_prediction)
    elif trigger_id == 'reset-button':
        return "", ""
    else:
        return dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
