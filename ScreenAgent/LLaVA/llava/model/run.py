import torch
import time
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from llava_cache_adapter import LlavaCacheAdapter

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def add_noise(image, noise_level=0.01):
    img_arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, noise_level * 255, img_arr.shape)
    img_noised = img_arr + noise
    img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)
    return Image.fromarray(img_noised)

def inference(model, tokenizer, image_processor, image, query, device='cuda'):
    conv = conv_templates["vicuna_v1"].copy()
    
    inp = DEFAULT_IMAGE_TOKEN + '\n' + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().to(device)
    input_token_len = input_ids.shape[1]

    torch.cuda.synchronize()
    start_t = time.time()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=100,
            use_cache=True,
            min_new_tokens=50
        )

    torch.cuda.synchronize()
    end_t = time.time()

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    
    if "ASSISTANT:" in full_text:
        answer = full_text.split("ASSISTANT:")[-1].strip()
    else:
        input_len = input_ids.shape[1]
        answer = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    return end_t - start_t, answer, output_ids[0]

def main():
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    device = "cuda"
    device_map = {"": 0}

    print(f"Loading Model: {model_name} (Native Float16)")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_4bit=False,
        load_8bit=False,
        device_map=device_map,
        device=device
    )
    model.to(dtype=torch.float16)
    
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)

    adapter = LlavaCacheAdapter(model, device=device)

    PNG = "png_data/track/"
    PNG_ORI = "track_ori.png"
    PNG_MOD1 = "track_mod_1.png"
    PNG_MOD2 = "track_mod_2.png"
    PNG_MOD3 = "track_mod_3.png"

    img_ori = load_image(PNG+PNG_ORI)

    images_sequence = [
        ("Original", img_ori),
        ("MOD1", load_image(PNG+PNG_MOD1)),
        ("MOD2", load_image(PNG+PNG_MOD2)),
        ("MOD3", load_image(PNG+PNG_MOD3))
    ]
    
    query = "Describe this image in detail."
    print("\nWarming up GPU.")
    inference(model, tokenizer, image_processor, img_ori, query, device)

    # Baseline
    print("\nBaseline:")
    adapter.disable()
    
    baseline_times = []
    for label, img in images_sequence:
        cost, text, tokens = inference(model, tokenizer, image_processor, img, query, device)
        baseline_times.append(cost)
        print(f"[{label}] Time: {cost:.4f}s | Len: {len(tokens)}")
    
    avg_base = sum(baseline_times)/len(baseline_times)
    print(f"Avg Baseline: {avg_base:.4f}s")

    print("\nCleaning GPU memory.")
    adapter.clear_cache()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("Warming up Adapter.")
    adapter.enable()
    inference(model, tokenizer, image_processor, img_ori, query, device)
    adapter.clear_cache() 
    
    # Adapter
    print("\nAdapter:")
    # adapter.enable()
    adapter.clear_cache()
    
    adapter_times = []
    for label, img in images_sequence:
        cost, text, tokens = inference(model, tokenizer, image_processor, img, query, device)
        adapter_times.append(cost)
        print(f"[{label}] Time: {cost:.4f}s | Len: {len(tokens)}")

    avg_adapt = sum(adapter_times)/len(adapter_times)
    print(f"Avg Adapter: {avg_adapt:.4f}s")
    
    if avg_adapt > 0:
        print(f"\nFinal Speedup: {avg_base / avg_adapt:.2f}x")

if __name__ == "__main__":
    main()