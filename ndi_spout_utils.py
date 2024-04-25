import time
import numpy as np
import cv2
import array
from multiprocessing import Queue
from OpenGL import GL
from itertools import repeat
import NDIlib as ndi
import SpoutGL
from pythonosc import udp_client
import os
import time
import random

ndi_send = None
ndi_name = None

class OSCClientFactory:
    _clients = {}
    @staticmethod
    def get_client(osc_out_port):
        if osc_out_port not in OSCClientFactory._clients:
            OSCClientFactory._clients[osc_out_port] = udp_client.SimpleUDPClient("127.0.0.1", osc_out_port)
        return OSCClientFactory._clients[osc_out_port]

def send_osc_message(address: str, message, osc_out_port):
    osc_client = OSCClientFactory.get_client(osc_out_port)
    osc_client.send_message(address, message)

def spout_capture(queue: Queue, sender_name: str = "SpoutGL-test"):
    with SpoutGL.SpoutReceiver() as receiver:
        receiver.setReceiverName(sender_name)
        buffer = None
        while True:
            result = receiver.receiveImage(buffer, GL.GL_RGBA, False, 0)
            if receiver.isUpdated():
                width = receiver.getSenderWidth()
                height = receiver.getSenderHeight()
                buffer = array.array('B', repeat(0, width * height * 4))
            if buffer and result and not SpoutGL.helpers.isBufferEmpty(buffer):
                frame_data = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
                if frame_data.dtype != np.uint8:
                    frame_data = frame_data.astype(np.uint8)
                frame_data = frame_data[..., :3]  
                frame_data = frame_data[..., ::-1]
                if queue.empty():
                    queue.put(frame_data)
            receiver.waitFrameSync(sender_name, 10000)


def spout_transmit(queue: Queue, sender_name, osc_out_port):
    print(f"\nInitializing Spout Streaming... Sender: {sender_name}")
    with SpoutGL.SpoutSender() as sender:
        sender.setSenderName(sender_name)
        start_time = time.time() 
        transmit_count = 0
        total_frame_count = 0 
        while True:
            if not queue.empty():
                frame_data = queue.get(block=False)
                if frame_data.ndim == 4 and frame_data.shape[0] == 1:
                    frame_data = frame_data.squeeze(0)  
                height, width, channels = frame_data.shape
                if channels == 3:  
                    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2RGBA)
                if frame_data.dtype != np.uint8:
                    frame_data = frame_data.astype(np.uint8)
                result = sender.sendImage(frame_data.tobytes(), width, height, GL.GL_RGBA, False, 0)
                send_osc_message('/framecount', total_frame_count, osc_out_port)
                sender.setFrameSync(sender_name)
                transmit_count += 1
                total_frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps = round(transmit_count / elapsed_time, 3)
                    send_osc_message('/stream-info/fps', fps, osc_out_port)
                    send_osc_message('/stream-info/output-name', sender_name, osc_out_port)
                    print(f"Streaming... Active | Sender: {sender_name} | FPS: {fps}\r", end='')
                    start_time = time.time()
                    transmit_count = 0
            time.sleep(0.005)


def ndi_capture(selected_source_name, queue: Queue):
    stop_capture = False 
    ndi_recv = ndi.recv_create_v3()
    if ndi_recv is None:
        raise Exception("Cannot initialize NDI receiver")
    ndi_source = ndi.Source()
    ndi_source.ndi_name = selected_source_name
    ndi.recv_connect(ndi_recv, ndi_source)
    frame_count = 0
    while not stop_capture:
        frame_type, video_frame, _, _ = ndi.recv_capture_v2(ndi_recv, 5000)
        if frame_type == ndi.FRAME_TYPE_VIDEO:
            frame_data = np.frombuffer(video_frame.data, dtype=np.uint8)
            if video_frame.FourCC == ndi.FOURCC_VIDEO_TYPE_UYVY:
                frame_data = frame_data.reshape((video_frame.yres, video_frame.xres, 2))
                frame_data = cv2.cvtColor(frame_data, cv2.COLOR_YUV2BGR_UYVY)
            elif video_frame.FourCC == ndi.FOURCC_VIDEO_TYPE_RGBA:
                frame_data = frame_data.reshape((video_frame.yres, video_frame.xres, 4))
            else:
                raise Exception("Unsupported video frame format")
            if queue.empty():
                queue.put(frame_data)
                frame_count += 1
            ndi.recv_free_video_v2(ndi_recv, video_frame) 
    ndi.recv_destroy(ndi_recv)

def generate_unique_ndi_name():
    minutes_since_epoch = int(time.time() // 60) 
    random_number = random.randint(100, 999)
    return f'SD_NDI_{minutes_since_epoch}_{random_number}'

def send_images_ndi(frame_data):
    global ndi_send, ndi_name
    if ndi_send is None:
        ndi_name = generate_unique_ndi_name()
        create_params = ndi.SendCreate()
        create_params.clock_video = False
        create_params.clock_audio = False
        create_params.ndi_name = ndi_name
        ndi_send = ndi.send_create(create_params)
        if ndi_send is None:
            raise Exception(f"Failed to create NDI send instance with name '{ndi_name}'")
    if frame_data.shape[-1] == 3:
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2RGBA)
    video_frame = ndi.VideoFrameV2()
    video_frame.data = frame_data
    video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
    ndi.send_send_video_v2(ndi_send, video_frame)

def ndi_transmit(queue: Queue, osc_out_port):
    global ndi_name
    print("\n====================================")
    print("Initializing NDI Streaming... Sender: 'StreamDiffusion_NDI_out'")
    print("====================================\n")

    transmit_count = 0
    total_frame_count = 0
    start_time = time.time()

    while True:
        try:
            if not queue.empty():
                processed_np = queue.get(block=False)

                # Ensure processed_np is in expected shape
                if processed_np.ndim == 3:  # Expected shape (H, W, C)
                    processed_np = np.expand_dims(processed_np, axis=0)  # Add batch dimension if missing
                if processed_np.ndim == 4 and processed_np.shape[0] == 1:
                    processed_np = processed_np.squeeze(0)  # Remove batch dimension for single image

                # Ensure processed_np is of type np.uint8
                if processed_np.dtype != np.uint8:
                    processed_np = processed_np.astype(np.uint8)

                if processed_np.shape[-1] == 3:
                    processed_np = cv2.cvtColor(processed_np, cv2.COLOR_RGB2RGBA)

                # Debug: Check final shape before sending
                # print(f"Final shape before NDI send: {processed_np.shape}")

                # Attempt to send the image through NDI
                send_images_ndi(processed_np)

                # OSC and FPS counting
                send_osc_message('/framecount', total_frame_count, osc_out_port)
                transmit_count += 1
                total_frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps = round(transmit_count / elapsed_time, 3)
                    send_osc_message('/stream-info/fps', fps, osc_out_port)
                    send_osc_message('/stream-info/output-name', ndi_name, osc_out_port)
                    print(f"Streaming... Active | Sender: {ndi_name} | FPS: {fps}\r", end='')
                    start_time = time.time()
                    transmit_count = 0
            else:
                time.sleep(0.005)
        except Exception as e:
            print(f"Error during NDI transmission: {e}")

def select_ndi_source(sender_name=None):
    ndi_find = ndi.find_create_v2()
    if ndi_find is None:
        print("\n====================================")
        print("ERROR: Cannot initialize NDI find")
        print("====================================\n")
        raise Exception("Cannot initialize NDI find")
    time.sleep(2)
    ndi.find_wait_for_sources(ndi_find, 1000)
    ndi_sources = ndi.find_get_current_sources(ndi_find)
    if sender_name:
        for source in ndi_sources:
            if sender_name.lower() in source.ndi_name.lower():
                print(f"Using NDI sender name: {source.ndi_name}\n")
                return source.ndi_name
        print("\n====================================")
        print(f"WARNING: NDI source containing '{sender_name}' not found.")
        print("====================================\n")
    print("\n====================================")
    print("Available NDI sources:")
    for i, source in enumerate(ndi_sources):
        print(f"{i + 1}. {source.ndi_name}")
    print("====================================\n")
    selected_index = int(input("Enter the number of the NDI source you want to use: ")) - 1
    if selected_index < 0 or selected_index >= len(ndi_sources):
        print("\n====================================")
        print("ERROR: Invalid NDI source selected")
        print("====================================\n")
        raise Exception("Invalid NDI source selected")
    return ndi_sources[selected_index].ndi_name
