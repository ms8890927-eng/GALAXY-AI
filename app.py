import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# استخدام نسخة أصغر (0.5B) لضمان العمل على سيرفرات Render المجانية
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32, # الـ CPU يحتاج float32
    low_cpu_mem_usage=True,
    device_map="cpu"
)

def predict(message, history, persona, age_group, job_title):
    system_prompt = f"أنت GALAXY AI. شخصيتك: {persona}. المستخدم عمره {age_group} ووظيفته {job_title}."
    
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]

# الواجهة
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align:center;'>🌌 GALAXY AI</h1>")
    with gr.Row():
        persona = gr.Dropdown(["مساعد عام", "خبير برمجة"], value="مساعد عام", label="الشخصية")
        age = gr.Radio(["طفل", "شاب", "كبير"], value="شاب", label="السن")
        job = gr.Textbox(label="الوظيفة", value="طالب")
    gr.ChatInterface(fn=predict, additional_inputs=[persona, age, job])

if __name__ == "__main__":
    # Render يحتاج تشغيل التطبيق على بورت 10000 أو المتغير المتاح
    import os
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
