import logging
import queue
import threading
import time
from typing import List

import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv

from deal_agent_framework import DealAgentFramework
from log_utils import reformat


load_dotenv(override=True)


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def html_for(log_data: List[str]) -> str:
    output = "<br>".join(log_data[-18:])
    return (
        '<div style="height:400px;overflow-y:auto;border:1px solid #ccc;'
        'background:#222229;padding:10px;font-family:monospace">'
        f"{output}</div>"
    )


def setup_logging(log_queue) -> QueueHandler:
    handler = QueueHandler(log_queue)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return handler


class App:
    def __init__(self):
        self.agent_framework = None

    def get_agent_framework(self) -> DealAgentFramework:
        if self.agent_framework is None:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    @staticmethod
    def table_for(opportunities):
        return [
            [
                item.deal.product_description,
                f"${item.deal.price:.2f}",
                f"${item.estimate:.2f}",
                f"${item.discount:.2f}",
                item.deal.url,
            ]
            for item in opportunities
        ]

    def get_plot(self):
        _, vectors, colors = self.get_agent_framework().get_plot_data(max_datapoints=800)
        figure = go.Figure(
            data=[
                go.Scatter3d(
                    x=vectors[:, 0],
                    y=vectors[:, 1],
                    z=vectors[:, 2],
                    mode="markers",
                    marker={"size": 2, "color": colors, "opacity": 0.7},
                )
            ]
        )
        figure.update_layout(
            scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"},
            height=400,
            margin={"r": 5, "b": 1, "l": 5, "t": 20},
        )
        return figure

    def run_with_logging(self, initial_log_data):
        log_data = list(initial_log_data or [])
        log_queue = queue.Queue()
        result_queue = queue.Queue()
        handler = setup_logging(log_queue)
        framework = self.get_agent_framework()

        def worker():
            try:
                memory = framework.run()
                result_queue.put(("ok", self.table_for(memory)))
            except Exception as exc:
                logging.exception("Deal run failed")
                result_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        final_table = self.table_for(framework.memory)

        try:
            while True:
                while True:
                    try:
                        log_data.append(reformat(log_queue.get_nowait()))
                    except queue.Empty:
                        break
                try:
                    status, result = result_queue.get_nowait()
                    if status == "ok":
                        final_table = result
                    else:
                        log_data.append(reformat(f"Run stopped: {result}"))
                    yield log_data, html_for(log_data), final_table
                    break
                except queue.Empty:
                    yield log_data, html_for(log_data), final_table
                    time.sleep(0.15)
        finally:
            logging.getLogger().removeHandler(handler)

    def notify_selected(self, selected: gr.SelectData):
        framework = self.get_agent_framework()
        framework.init_agents_as_needed()
        row = selected.index[0] if isinstance(selected.index, (list, tuple)) else selected.index
        if not isinstance(row, int) or row < 0 or row >= len(framework.memory):
            return
        framework.planner.messenger.alert(framework.memory[row])

    def build(self):
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
            log_data = gr.State([])
            gr.Markdown(
                '<div style="text-align:center;font-size:24px"><strong>The Price is Right</strong> '
                "— Autonomous deal-hunting agents</div>"
            )
            gr.Markdown(
                '<div style="text-align:center;font-size:14px">Scanner, RAG, local neural network, '
                "Modal specialist, and messaging agents working together.</div>"
            )
            opportunities = gr.Dataframe(
                headers=["Deals found", "Price", "Estimate", "Discount", "URL"],
                wrap=True,
                column_widths=[6, 1, 1, 1, 3],
                row_count=10,
                col_count=5,
                height=400,
                interactive=False,
            )
            with gr.Row():
                with gr.Column():
                    logs = gr.HTML()
                with gr.Column():
                    plot = gr.Plot(value=self.get_plot(), show_label=False)

            ui.load(
                self.run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities],
            )
            timer = gr.Timer(value=300, active=True)
            timer.tick(
                self.run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities],
            )
            opportunities.select(self.notify_selected)
        return ui

    def run(self):
        self.build().launch(share=False, inbrowser=True)


if __name__ == "__main__":
    App().run()
