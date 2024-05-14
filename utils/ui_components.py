from functools import wraps

import gradio as gr


class FormComponent:
    webui_do_not_create_gradio_pyi_thank_you = True

    def get_expected_parent(self):
        return gr.components.Form


class ToolButton(gr.Button, FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    @wraps(gr.Button.__init__)
    def __init__(self, value="", *args, elem_classes=None, **kwargs):
        elem_classes = elem_classes or []
        super().__init__(*args, elem_classes=["tool", *elem_classes], value=value, **kwargs)

    def get_block_name(self):
        return "button"
