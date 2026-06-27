import gradio as gr

from Src.inference.generate import generate_text


demo = gr.Interface(
    fn=generate_text,

    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="Enter your prompt here...",
            lines=5,
        ),
    ],

    outputs=[
        gr.Textbox(
            label="Generated Text",
            lines=15,
        ),
    ],

    title="Mini GPT",

    description="""
Character-level GPT built completely from scratch using PyTorch.
Enter a prompt and let the model continue generating text.
""",
)

demo.launch()