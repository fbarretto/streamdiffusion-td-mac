#MAIN.py td
import os
import sys
import time
import threading
import numpy as np
import cv2
# import NDIlib as ndi
import fire
import json
import argparse
import signal
from multiprocessing import Process, Queue, Manager, shared_memory
from queue import Empty
from typing import Dict, Literal, Optional, List
import torch
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
import uuid
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Update these import statements
from wrapper_td import StreamDiffusionWrapper
#from utils.wrapper import StreamDiffusionWrapper

from pipeline_td import StreamDiffusion

# from compel import Compel


# from utils.wrapper import StreamDiffusionWrapper
from streamdiffusion.image_utils import pil2tensor, pt_to_numpy, numpy_to_pil, postprocess_image

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# import SpoutGL
# from OpenGL import GL
import array
from itertools import repeat
from pythonosc import udp_client

# from ndi_spout_utils import spout_capture, spout_transmit, ndi_capture, ndi_transmit, select_ndi_source


# if not ndi.initialize():
#     raise Exception("NDI cannot initialize")

# # Debug Output Directory
# debug_output_dir = "debug-outputs"
# if not os.path.exists(debug_output_dir):
#     os.makedirs(debug_output_dir)

def select_input_type():
    print("\n==============================")
    print("Select the input type:")
    print("1. Spout")
    print("2. NDI")
    print("==============================\n")

    choice = input("Enter your choice (1 for Spout, 2 for NDI): ")
    if choice == "1":
        return "spout"
    elif choice == "2":
        return "ndi"
    else:
        raise Exception("Invalid input type selected")


class OSCClientFactory:
    _clients = {}
    @staticmethod
    def get_client(osc_out_port):
        if osc_out_port not in OSCClientFactory._clients:
            OSCClientFactory._clients[osc_out_port] = udp_client.SimpleUDPClient("127.0.0.1", osc_out_port)
        return OSCClientFactory._clients[osc_out_port]


def send_osc_message(address, value, osc_out_port):
    client = OSCClientFactory.get_client(osc_out_port)
    client.send_message(address, value)

def calculate_fps_and_send_osc(start_time, transmit_count, osc_out_port, sender_name, frame_created):
    if frame_created:
        transmit_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        if frame_created:
            fps = round(transmit_count / elapsed_time, 3)
            send_osc_message('/stream-info/fps', fps, osc_out_port)
            print(f"Streaming... Active | Sender: {sender_name} | FPS: {fps}\r", end='', flush=True)
        send_osc_message('/stream-info/output-name', sender_name, osc_out_port)
        start_time = time.time()
        if frame_created:
            transmit_count = 0
    return start_time, transmit_count


def osc_server(shared_data, ip='127.0.0.1', port=8247):

    def set_negative_prompt_handler(address, *args):
        shared_data["negative_prompt"] = args[0]

    def set_guidance_scale_handler(address, *args):
        shared_data["guidance_scale"] = args[0]

    def set_delta_handler(address, *args):
        shared_data["delta"] = args[0]

    def set_seed_handler(address, *args):
        shared_data["seed"] = args[0]

    def set_t_list_handler(address, *args):
        shared_data["t_list"] = list(args)  # Assuming args contains the t_list values

    def set_prompt_list_handler(address, *args):
        # Assuming args[0] is a string representation of a list
        prompt_list_str = args[0]
        # Convert the string back to a list
        prompt_list = json.loads(prompt_list_str)
        shared_data["prompt_list"] = prompt_list

    def set_seed_list_handler(address, *args):
        # Assuming args[0] is a string representation of a list of lists
        seed_list_str = args[0]
        # Convert the string back to a list of lists
        seed_list = json.loads(seed_list_str)
        # Convert each inner list to a list with an int and a float
        seed_list = [[int(seed_val), float(weight)] for seed_val, weight in seed_list]
        shared_data["seed_list"] = seed_list

    def set_sdmode_handler(address, *args):
        shared_data["sdmode"] = args[0]

    def stop_stream_handler(address, *args):
        print("Stop command received. Stopping the stream.")
        shared_data["stop_stream"] = True

    def set_gaussian_prompt_handler(address, *args):
        shared_data["gaussian_prompt"] = args[0]
        print(f"Gaussian Prompt: {shared_data['gaussian_prompt']}")


    def set_td_buffer_name_handler(address, *args):
        shared_data["input_mem_name"] = args[0]

    dispatcher = Dispatcher()

    dispatcher.map("/negative_prompt", set_negative_prompt_handler)
    dispatcher.map("/guidance_scale", set_guidance_scale_handler)
    dispatcher.map("/delta", set_delta_handler)
    dispatcher.map("/seed", set_seed_handler)
    dispatcher.map("/t_list", set_t_list_handler)

    dispatcher.map("/prompt_list", set_prompt_list_handler)
    dispatcher.map("/seed_list", set_seed_list_handler)

    dispatcher.map("/sdmode", set_sdmode_handler)

    dispatcher.map("/stop", stop_stream_handler)

    dispatcher.map("/gaussian_prompt", set_gaussian_prompt_handler)
    dispatcher.map("/td_buffer_name", set_td_buffer_name_handler)

    server = BlockingOSCUDPServer((ip, port), dispatcher)
    print(f"Starting OSC server on {ip}:{port}")
    server.serve_forever()

def print_sdtd_title():
    sdtd_title = """
\033[33m====================================\033[0m
\033[33mStreamDiffusionTD\033[0m
\033[33m====================================\033[0m
    """
    print(sdtd_title)

def terminate_processes(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()

def image_generation_process(
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    shared_data,
    t_index_list: List[int] ,
    mode:str,
    lcm_lora_id: Optional[str] = None,
    vae_id: Optional[str] = None,
    input_mem_name: str = "input_mem_name",
    osc_transmit_port: Optional[int] = None,
    scheduler_name: str = "EulerAncestral",
    use_karras_sigmas: bool = False,
    device = Literal["cpu","cuda", "mps"] = "cuda"

) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {"LoRA_1" : 0.5 , "LoRA_2" : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    frame_buffer_size : int, optional
        The frame buffer size for denoising batch, by default 1.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    do_add_noise : bool, optional
        Whether to add noise for following denoising steps or not,
        by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default False.
    similar_image_filter_threshold : float, optional
        The threshold for similar image filter, by default 0.98.
    similar_image_filter_max_skip_frame : int, optional
        The max skip frame for similar image filter, by default 10.
    """

    global inputs
    global stop_capture
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=t_index_list,
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
        lcm_lora_id=lcm_lora_id,  # Add this line
        vae_id=vae_id,
        scheduler_name=scheduler_name,
        use_karras_sigmas=use_karras_sigmas

    )

    current_prompt = prompt
    current_prompt_list = shared_data.get("prompt_list", [[prompt, 1.0]])
    current_seed_list = shared_data.get("seed_list", [[seed, 1.0]])
    
    noise_bank = {}
    prompt_cache = {}

    print('Preparing Stream...\n\n')

    stream.prepare(
        prompt=current_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )

    time.sleep(1)

    input_memory = None
    output_memory = None
    start_time = time.time()
    prompt_changed = False
    frame_count = 0
    transmit_count = 0
    output_mem_name = f"sd_output_{int(time.time())}"

    while True:
        try:
            if mode == "img2img":

                if 'input_mem_name' in shared_data and shared_data['input_mem_name'] != input_mem_name:
                    if input_memory is not None:
                        input_memory.close()  # Close the existing shared memory
                        input_memory = None
                    input_mem_name = shared_data['input_mem_name']

                if input_memory is None:
                    current_time = lambda: int(round(time.time() * 1000))
                    try:
                        input_memory = shared_memory.SharedMemory(name=input_mem_name)
                        print(f"{current_time()} - Shared memory '{input_mem_name}' found.")
                    except FileNotFoundError:
                        print(f"{current_time()} - Shared memory '{input_mem_name}' not found")
                        time.sleep(1)
                        continue

                total_size_bytes = input_memory.size
                buffer = np.ndarray(shape=(total_size_bytes,), dtype=np.uint8, buffer=input_memory.buf)
                image_data_size = width * height * 3

                try:
                    frame_np = buffer[:image_data_size].reshape((height, width, 3))
                except Exception as e:
                    print(f"Error in extracting image data: {str(e)}")
                    continue

                extra_data = np.frombuffer(buffer[image_data_size:], dtype=np.float32)
                input_tensor = to_tensor(frame_np)
                processed_tensor = stream.stream(input_tensor)
                processed_np = postprocess_image(processed_tensor, output_type="np")
                processed_np = (processed_np * 255).astype(np.uint8)
                if output_memory is None:
                    output_memory = shared_memory.SharedMemory(name=output_mem_name, create=True, size=processed_np.nbytes)
                output_array = np.ndarray(processed_np.shape, dtype=processed_np.dtype, buffer=output_memory.buf)
                output_array[:] = processed_np[:]

                send_osc_message('/framecount', frame_count, osc_transmit_port)
                start_time, transmit_count = calculate_fps_and_send_osc(start_time, transmit_count, osc_transmit_port, output_mem_name, True)

            elif mode == "txt2img":
                try:
                    processed_np = custom_txt2img_using_prepared_noise(stream_diffusion=stream.stream, expected_batch_size=1, output_type='np')
                    if processed_np.max() <= 1.0:
                        processed_np = (processed_np * 255).astype(np.uint8)

                    if output_memory is None:
                        output_memory = shared_memory.SharedMemory(name=output_mem_name, create=True, size=processed_np.nbytes)

                    output_array = np.ndarray(processed_np.shape, dtype=processed_np.dtype, buffer=output_memory.buf)
                    output_array[:] = processed_np[:]
                    send_osc_message('/framecount', frame_count, osc_transmit_port)
                    start_time, transmit_count = calculate_fps_and_send_osc(start_time, transmit_count, osc_transmit_port, output_mem_name, True)

                except Exception as e:
                    print(f"Error in txt2img mode: {str(e)}")


            frame_count += 1
            new_sdmode = shared_data.get("sdmode", mode)
            if new_sdmode != mode:
                mode = new_sdmode
            # PROMPT DICT + GUIDANCE SCALE + DELTA
            new_guidance_scale = float(shared_data.get("guidance_scale", guidance_scale))
            new_delta = float(shared_data.get("delta", delta))
            new_prompt_list = shared_data.get("prompt_list", {})
            new_negative_prompt = shared_data.get("negative_prompt", negative_prompt)
            gaussian_prompt = shared_data.get("gaussian_prompt", False)
            # Check if there is an actual change in parameters
            if (new_prompt_list != current_prompt_list or 
                new_guidance_scale != guidance_scale or 
                new_delta != delta or 
                new_negative_prompt != negative_prompt):
                # Update the current values
                current_prompt_list = new_prompt_list
                guidance_scale = new_guidance_scale
                delta = new_delta
                negative_prompt = new_negative_prompt
                update_combined_prompts_and_parameters(
                    stream.stream, 
                    current_prompt_list, 
                    guidance_scale, 
                    delta, 
                    negative_prompt,
                    prompt_cache,
                    # gaussian_prompt
                )
            ##SEED DICT
            new_seed_list = shared_data.get("seed_list", current_seed_list)
            if new_seed_list != current_seed_list:
                current_seed_list = new_seed_list
                # Check if all weights are zero
                if any(weight > 0 for _, weight in current_seed_list):
                    blended_noise = blend_noise_tensors(current_seed_list, noise_bank, stream.stream)
                    stream.stream.init_noise = blended_noise
            ##T_LIST
            new_t_list = shared_data.get("t_list", t_index_list)
            if new_t_list != stream.stream.t_list:
                update_t_list_attributes(stream.stream, new_t_list)
            ##STOP STREAM
            if shared_data.get("stop_stream", False):
                print("Stopping image generation process.")
                break
            else:
                time.sleep(0.001)
        except KeyboardInterrupt:
            break

# function to update t_list-related attributes
def update_t_list_attributes(stream_diffusion_instance, new_t_list):
    stream_diffusion_instance.t_list = new_t_list
    stream_diffusion_instance.sub_timesteps = [stream_diffusion_instance.timesteps[t] for t in new_t_list]
    sub_timesteps_tensor = torch.tensor(
        stream_diffusion_instance.sub_timesteps, dtype=torch.long, device=stream_diffusion_instance.device
    )
    stream_diffusion_instance.sub_timesteps_tensor = torch.repeat_interleave(
        sub_timesteps_tensor, 
        repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, 
        dim=0
    )
    c_skip_list = []
    c_out_list = []
    for timestep in stream_diffusion_instance.sub_timesteps:
        c_skip, c_out = stream_diffusion_instance.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
        c_skip_list.append(c_skip)
        c_out_list.append(c_out)
    stream_diffusion_instance.c_skip = torch.stack(c_skip_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    stream_diffusion_instance.c_out = torch.stack(c_out_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    # Calculate alpha_prod_t_sqrt and beta_prod_t_sqrt
    alpha_prod_t_sqrt_list = []
    beta_prod_t_sqrt_list = []
    for timestep in stream_diffusion_instance.sub_timesteps:
        alpha_prod_t_sqrt = stream_diffusion_instance.scheduler.alphas_cumprod[timestep].sqrt()
        beta_prod_t_sqrt = (1 - stream_diffusion_instance.scheduler.alphas_cumprod[timestep]).sqrt()
        alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
        beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
    alpha_prod_t_sqrt = torch.stack(alpha_prod_t_sqrt_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    beta_prod_t_sqrt = torch.stack(beta_prod_t_sqrt_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    stream_diffusion_instance.alpha_prod_t_sqrt = torch.repeat_interleave(alpha_prod_t_sqrt, repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, dim=0)
    stream_diffusion_instance.beta_prod_t_sqrt = torch.repeat_interleave(beta_prod_t_sqrt, repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, dim=0)

@torch.no_grad()
def update_combined_prompts_and_parameters(stream_diffusion, prompt_list, new_guidance_scale, new_delta, new_negative_prompt, prompt_cache):
    stream_diffusion.guidance_scale = new_guidance_scale
    stream_diffusion.delta = new_delta
    prompt_text = ''
    if stream_diffusion.guidance_scale > 1.0 and (stream_diffusion.cfg_type in ["self", "initialize"]):
        stream_diffusion.stock_noise *= stream_diffusion.delta
    combined_embeds = None
    current_prompts = set()
    for idx, prompt in enumerate(prompt_list):
        prompt_text, weight = prompt
        if weight == 0:
            continue
        current_prompts.add(idx)
        if idx not in prompt_cache or prompt_cache[idx]['text'] != prompt_text:
            encoder_output = stream_diffusion.pipe.encode_prompt(
                prompt=prompt_text,
                device=stream_diffusion.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=stream_diffusion.guidance_scale > 1.0,
                negative_prompt=new_negative_prompt,
            )
            prompt_cache[idx] = {'embed': encoder_output[0], 'text': prompt_text}
        weighted_embeds = prompt_cache[idx]['embed'] * weight
        if combined_embeds is None:
            combined_embeds = weighted_embeds
        else:
            combined_embeds += weighted_embeds
    if combined_embeds is not None:
        stream_diffusion.prompt_embeds = combined_embeds.repeat(stream_diffusion.batch_size, 1, 1)
    unused_prompts = set(prompt_cache.keys()) - current_prompts
    for prompt in unused_prompts:
        del prompt_cache[prompt]

def blend_noise_tensors(seed_list, noise_bank, stream_diffusion):
    blended_noise = None
    total_weight = 0
    for seed_val, weight in seed_list:
        if weight == 0:
            continue
        noise_tensor = noise_bank.get(seed_val)
        if noise_tensor is None:
            generator = torch.Generator().manual_seed(seed_val)
            noise_tensor = torch.randn(
                (stream_diffusion.batch_size, 4, stream_diffusion.latent_height, stream_diffusion.latent_width),
                generator=generator
            ).to(device=stream_diffusion.device, dtype=stream_diffusion.dtype)
            noise_bank[seed_val] = noise_tensor
        if blended_noise is None:
            blended_noise = noise_tensor * weight
        else:
            blended_noise += noise_tensor * weight
        total_weight += weight
    return blended_noise

def custom_txt2img_using_prepared_noise(stream_diffusion, expected_batch_size, output_type='np'):
    if stream_diffusion.init_noise.size(0) > expected_batch_size:
        adjusted_noise = stream_diffusion.init_noise[:expected_batch_size]
    elif stream_diffusion.init_noise.size(0) < expected_batch_size:
        repeats = [expected_batch_size // stream_diffusion.init_noise.size(0)] + [-1] * (stream_diffusion.init_noise.dim() - 1)
        adjusted_noise = stream_diffusion.init_noise.repeat(*repeats)[:expected_batch_size]
    else:
        adjusted_noise = stream_diffusion.init_noise

    x_0_pred_out = stream_diffusion.predict_x0_batch(adjusted_noise)
    x_output = stream_diffusion.decode_image(x_0_pred_out).detach().clone()

    if output_type == 'np':
        x_output = postprocess_image(x_output, output_type=output_type)

    return x_output


def main():
    def signal_handler(sig, frame):
        print('Exiting...')
        terminate_processes([osc_process, generation_process])
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description="StreamDiffusion NDI Stream Script")
    parser.add_argument('-c', '--config', type=str, default='stream_config.json', help='Path to the configuration file')
    args = parser.parse_args()
    print_sdtd_title()
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_script_dir, args.config)
    print(f"Config file path: {config_file_path}")
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    # Open and read the config file
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    osc_receive_port = config.get("osc_out_port", 8247)
    osc_transmit_port = config.get("osc_in_port", 8248)
    model_id_or_path = config["model_id_or_path"]
    lora_dict = config["lora_dict"]
    prompt = config["prompt"]
    negative_prompt = config["negative_prompt"]
    frame_buffer_size = config["frame_buffer_size"]
    width = config["width"]
    height = config["height"]
    acceleration = config["acceleration"]
    use_denoising_batch = config["use_denoising_batch"]
    seed = config["seed"]
    cfg_type = config["cfg_type"]
    guidance_scale = config["guidance_scale"]
    delta = config["delta"]
    do_add_noise = config["do_add_noise"]
    enable_similar_image_filter = config["enable_similar_image_filter"]
    similar_image_filter_threshold = config["similar_image_filter_threshold"]
    similar_image_filter_max_skip_frame = config["similar_image_filter_max_skip_frame"]
    t_index_list = config.get("t_index_list", [25, 40])
    mode=config.get("sdmode","img2img")
    lcm_lora_id = config.get("lcm_lora_id")
    vae_id = config.get("vae_id")
    scheduler_name = config.get("scheduler_name", "EulerAncestral")
    use_karras_sigmas = config.get("use_karras_sigmas", False)
    input_mem_name = config["input_mem_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"\n\ninput_mem_name: {input_mem_name}\n")


    print(f"Model ID or Path: {model_id_or_path}\n")
    if lcm_lora_id is not None:
        print(f"LCM LoRA ID: {lcm_lora_id}\n")
    else:
        print("LCM LoRA ID: None\n")
    if vae_id is not None:
        print(f"VAE ID: {vae_id}\n")
    else:
        print("VAE ID: None\n")
    if lora_dict is not None:
        for model_path, weight in lora_dict.items():
            print(f"LoRA Model: {model_path}, Weight: {weight}\n\n")
    print("====================================\n")


    if os.path.isfile(model_id_or_path):
        model_id_or_path = model_id_or_path.replace('/', '\\')


    with Manager() as manager:
        shared_data = manager.dict()
        shared_data["prompt"] = prompt

        osc_process = Process(target=osc_server, args=(shared_data, '127.0.0.1', osc_receive_port))
        osc_process.start()

        generation_process = Process(
            target=image_generation_process,
            args=(
                model_id_or_path,
                lora_dict,
                prompt,
                negative_prompt,
                frame_buffer_size,
                width,
                height,
                acceleration,
                use_denoising_batch,
                seed,
                cfg_type,
                guidance_scale,
                delta,
                do_add_noise,
                enable_similar_image_filter,
                similar_image_filter_threshold,
                similar_image_filter_max_skip_frame,
                shared_data,
                t_index_list,
                mode,
                lcm_lora_id,  # Added this line
                vae_id,       # And this line
                input_mem_name,
                osc_transmit_port,
                scheduler_name,
                use_karras_sigmas,
                device
                ),
        )
        generation_process.start()

        try:
            while True:
                if shared_data.get("stop_stream", False):
                    print("Stop command received. Initiating shutdown...")
                    break
                time.sleep(0.1)  # Short sleep to prevent high CPU usage

        except KeyboardInterrupt:
            print("KeyboardInterrupt received, signalling to stop subprocesses...")
            shared_data["stop_stream"] = True

        finally:
            terminate_processes([osc_process, generation_process])
            generation_process.join()
            osc_process.join()
            print("All subprocesses terminated. Exiting main process...")
            sys.exit(0)
if __name__ == "__main__":
    fire.Fire(main)