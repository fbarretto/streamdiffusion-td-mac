"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""
import os
import subprocess
import socket
import json
import time
import traceback 
from functools import reduce
import ast
import datetime
import numpy as np
import shutil
import re
from TDStoreTools import StorageManager
import TDFunctions as TDF
import platform


param_json_map = {
    "Modelid": "model_id_or_path",
    "Tindexlist": "t_index_list",
    "Promptdict0concept": "prompt",
    "Negprompt": "negative_prompt",
    "Framesize": "frame_buffer_size",
    "Width": "width",
    "Height": "height",
    "Acceleration": "acceleration",
    "Denoisebatch": "use_denoising_batch",
    "Seeddict0seedval": "seed",
    "Cfgtype": "cfg_type",
    "Guidancescale": "guidance_scale",
    "Delta": "delta",
    "Addnoise": "do_add_noise",
    "Imagefilter": "enable_similar_image_filter",
    "Filterthresh": "similar_image_filter_threshold",
    "Maxskipframe": "similar_image_filter_max_skip_frame",
    "Oscinport": "osc_in_port",
    "Oscoutport": "osc_out_port",
    "Sdmode":"sdmode",
    "Customlcm" : "lcm_lora_id",
    "Customvae" : "vae_id",
    "Streamoutname" : "input_mem_name",
    # "Outputmemname" : "output_mem_name",
    # "Scheduler": "scheduler_name",
    # "Usekarrassigmas": "use_karras_sigmas"
}



class StreamDiffusionExt: 
    """
    StreamDiffusionExt description
    """
    def __init__(self, ownerComp):
        """
        Initializes the StarterExt class. Sets the owner component and calls the functions to create basic parameters and setup tables.
        """
        self.ownerComp = ownerComp

        self.setup_project_info_table()
        self.setup_par_details_table()


        self.logger = op('Logger').ext.Logger
        self.logger.log('StreamDiffusionExt initialized',
                        'StreamDiffusionExt started successfully.',
                        level='INFO') 

    def Updatesettings(self):
        print("update settings")
        """
        Sends the four settings (Prompt, Delta, Guidance Scale, and Negative Prompt) to the OSC Out DAT.
        """
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')

            # Sending the delta value
            delta_address = '/delta'
            delta_value = self.ownerComp.par.Delta.eval()
            osc_out.sendOSC(delta_address, [delta_value])

            # Sending the guidance scale value
            guidance_scale_address = '/guidance_scale'
            guidance_scale_value = self.ownerComp.par.Guidancescale.eval()
            osc_out.sendOSC(guidance_scale_address, [guidance_scale_value])

            # Sending the negative prompt
            negative_prompt_address = '/negative_prompt'
            negative_prompt_message = self.ownerComp.par.Negprompt.eval()
            osc_out.sendOSC(negative_prompt_address, [negative_prompt_message])

            t_index_list = []
            for block in self.ownerComp.par.Tindexblock.sequence:
                step_value = block.par.Step.eval()
                t_index_list.append(step_value)
            t_index_list_address = '/t_list'
            # Convert t_index_list to a format suitable for OSC
            osc_out.sendOSC(t_index_list_address, t_index_list)

    def send_osc_messages(self, message_dict):
        print("send osc messages")
        """
        Sends OSC messages to multiple addresses.

        Parameters:
        message_dict (dict): A dictionary where keys are OSC addresses and values are the messages to be sent.
        """
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')

            for address, message in message_dict.items():
                osc_out.sendOSC(address, [message])

    # def Gauspromptblend(self):
    #     """
    #     Updates the Gaussian mixing setting by sending an OSC message to the /gaussian_prompt channel.
    #     """
    #     if self.ownerComp.par.Streamactive:
    #         osc_out = op('oscout1')
    #         gaus_prompt_blend_value = self.ownerComp.par.Gauspromptblend.eval()
    #         osc_out.sendOSC('/gaussian_prompt', [gaus_prompt_blend_value])
    #         self.logger.log(f'Gaussian Mixing {"enabled" if gaus_prompt_blend_value else "disabled"}', level='INFO')

    def Generatesingle(self):
        print("generate single")
        run("parent.SDTD.op('numpy_share_out').par.Frameready = True", endFrame = True)

    def Promptblock(self):
        print("prompt block")
        """
        Sends the concept and weight from each block in the Promptdict sequence to the OSC Out DAT.
        If par.Normpweights is True, the weights are normalized so that their sum equals the value of par.Totalpweights.
        """
        prompt_list = []

        # Iterate over each block in the Promptdict sequence to build the list
        total_weight = 0
        for block in self.ownerComp.par.Promptdict.sequence:
            concept = block.par.Concept.eval()  
            weight = block.par.Weight.eval()  
            prompt = [concept, weight]  # Create a list
            prompt_list.append(prompt)  # Append the list to the list
            if self.ownerComp.par.Normpweights:
                total_weight += weight  

        # Normalize the weights if required
        if self.ownerComp.par.Normpweights and total_weight != 0:
            normalization_factor = self.ownerComp.par.Totalpweights / total_weight
            for i in range(len(prompt_list)):
                concept, weight = prompt_list[i]  # Unpack the list
                weight *= normalization_factor
                prompt_list[i] = [concept, weight]  # Create a new list

        # Convert the list to a JSON string
        prompt_list_str = json.dumps(prompt_list)

        # Send the JSON string to the OSC Out DAT
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')
            osc_out.sendOSC('/prompt_list', [prompt_list_str])
        # print(prompt_list_str)



    def Updatestreamout(self):
        print("update stream out")
        current_name = self.ownerComp.par.Streamoutname.eval()
        # Check if the current name ends with a number
        if re.search(r'\d+$', current_name):
            # Increment the final number
            new_name = re.sub(r'(\d+)$', lambda x: str(int(x.group(0)) + 1), current_name)
        else:
            # Append _1 if it doesn't end with a number
            new_name = f"{current_name}_1"

        # Update the Streamoutname parameter with the new name
        self.ownerComp.par.Streamoutname = new_name
        
        # Send the new stream out name via OSC
        osc_out = op('oscout1')
        osc_out.sendOSC('/td_buffer_name', [new_name])

    def Seed(self):
        print("seed")
        """
        Sends the seed settings to the OSC Out DAT as a list of lists, combining weights for identical seeds before normalization.
        """
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')

            # Use a dictionary to accumulate weights for each seed
            seed_weight_accumulator = {}

            # Iterate over each block in the Seeddict sequence to build the dictionary
            for block in self.ownerComp.par.Seeddict.sequence:
                seed_val = block.par.Seedval.eval()  # Extract the Seed value
                weight = block.par.Seedweight.eval()  # Extract the Weight value
                if weight != 0:  # Ignore pairs with a weight of 0
                    if seed_val in seed_weight_accumulator:
                        seed_weight_accumulator[seed_val] += weight  # Add weight to existing seed
                    else:
                        seed_weight_accumulator[seed_val] = weight  # Initialize new seed with its weight

            # Convert the accumulator dictionary to a list of [seed, weight] pairs
            seed_list = [[seed, weight] for seed, weight in seed_weight_accumulator.items()]

            # Calculate total weight for normalization
            total_weight = sum(weight for _, weight in seed_list)

            # Normalize the weights
            if total_weight != 0:
                for i in range(len(seed_list)):
                    seed_val, weight = seed_list[i]
                    weight /= total_weight
                    seed_list[i] = [seed_val, weight]

            # Calculate blend multipliers
            blend_multipliers = [1 + 0.5 * np.cos(np.pi * (weight - 0.5)) * (1 + 0.3 * (len(seed_list) - 2)) for _, weight in seed_list]

            avg_blend_multiplier = self.ownerComp.par.Noisemult.eval() * (sum(blend_multipliers) / len(blend_multipliers)) if blend_multipliers else 0

            # Apply the average blend multiplier to each weight
            for i in range(len(seed_list)):
                seed_val, weight = seed_list[i]
                weight *= avg_blend_multiplier
                seed_list[i] = [seed_val, weight]

            # Convert the list to a JSON string
            seed_list_str = json.dumps(seed_list)
            # Send the JSON string to the OSC Out DAT
            osc_out.sendOSC('/seed_list', [seed_list_str])

    def Sdmode(self,sdmode = None):
        print("sdmode")
        if self.ownerComp.par.Streamactive:
            osc_out = op('oscout1')
            if sdmode is None:
                sdmode = self.ownerComp.par.Sdmode.eval()
                if sdmode in ['txt2img', 'img2img']:
                    osc_out.sendOSC('/sdmode', [sdmode])

    def Startstream(self):
        print("start stream")
        """
        Starts the StreamDiffusion stream by executing a batch file.
        The batch file activates a Python virtual environment and starts the main script.
        """
        self.update_stream_config_dat()
        self.copy_ndi_code()

        base_folder = self.ownerComp.par.Basefolder.eval()
        venv_path = os.path.join(base_folder, 'venv')
        dot_venv_path = os.path.join(base_folder, '.venv')

        # Check if 'venv' or '.venv' directory exists and construct the activate script path
        if os.path.exists(venv_path):
            activate_script_path = os.path.join(venv_path, 'Scripts', 'activate.bat')
        elif os.path.exists(dot_venv_path):
            activate_script_path = os.path.join(dot_venv_path, 'Scripts', 'activate.bat')
        else:
            self.logger.log("Error: Basefolder parameter must be set to '/StreamDiffusion'. 'path/to/StreamDiffusion/venv' not found.", level="ERROR")
            return

        if not self.ownerComp.par.Visiblewindow:
            python_script_path = 'streamdiffusionTD\\main_sdtd.py'
            generation_command = f'{python_script_path}'

            full_command = f'cmd.exe /c "{activate_script_path} && python {generation_command}"'
            try:
                subprocess.Popen(full_command, cwd=base_folder, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
                self.logger.log('Started streaming process silently.', level='INFO')
            except Exception as e:
                self.logger.log(f'Error Startstream: Failed to start streaming process without command window. Error: {e}', level='ERROR')
            return
                    
        bat_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'Start_StreamDiffusion.bat')
        debug_cmd = 'pause' if self.ownerComp.par.Debugcmd.eval() else ''
        use_powershell = self.ownerComp.par.Powershell.eval()
        if use_powershell:
            batch_file_content = f"""
@echo off
cd /d %~dp0
if exist venv (
    PowerShell -Command "& {{& 'venv\\Scripts\\Activate.ps1'; & 'venv\\Scripts\\python.exe' 'streamdiffusionTD\\main_sdtd.py'}}"
) else (
    PowerShell -Command "& {{& '.venv\\Scripts\\Activate.ps1'; & '.venv\\Scripts\\python.exe' 'streamdiffusionTD\\main_sdtd.py'}}"
)
    {debug_cmd}
            """
        else:
            batch_file_content = f"""
@echo off
cd /d %~dp0
if exist venv (
    call venv\\Scripts\\activate.bat
    venv\\Scripts\\python.exe streamdiffusionTD\\main_sdtd.py
) else (
    call .venv\\Scripts\\activate.bat
    .venv\\Scripts\\python.exe streamdiffusionTD\\main_sdtd.py
)
    {debug_cmd}
            """
        with open(bat_file_path, 'w') as bat_file:
            bat_file.write(batch_file_content)
        subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=self.ownerComp.par.Basefolder.eval())
        self.logger.log(f'Started streaming using {("PowerShell" if use_powershell else "CMD")}.', level='INFO')

        
#     def Startstream(self, input_type=None, sender_name=None):
#         """
#         Starts the StreamDiffusion stream by executing a batch file with specified input type and sender name.
#         The batch file activates a Python virtual environment and starts the main script with the given parameters.

#         Parameters:
#         input_type (str, optional): The type of input to use (ndi or spout). Defaults to self.ownerComp.par.Streamtype.
#         sender_name (str, optional): The sender name, used if input_type is 'ndi'. Defaults to self.ownerComp.par.Streamoutname.
#         """
#         self.update_stream_config_dat()
#         self.copy_ndi_code()

#         # Use default values from component parameters if not provided
#         if input_type is None:
#             input_type = self.ownerComp.par.Streamtype.eval()
#         if sender_name is None:
#             sender_name = self.ownerComp.par.Streamoutname.eval()

#         base_folder = self.ownerComp.par.Basefolder.eval()
#         venv_path = os.path.join(base_folder, 'venv')
#         dot_venv_path = os.path.join(base_folder, '.venv')

#         # Check if 'venv' or '.venv' directory exists and construct the activate script path
#         if os.path.exists(venv_path):
#             activate_script_path = os.path.join(venv_path, 'Scripts', 'activate.bat')
#         elif os.path.exists(dot_venv_path):
#             activate_script_path = os.path.join(dot_venv_path, 'Scripts', 'activate.bat')
#         else:
#             self.logger.log("Error: Basefolder parameter must be set to '/StreamDiffusion'. 'path/to/StreamDiffusion/venv' not found.", level="ERROR")
#             return

#         if not self.ownerComp.par.Visiblewindow:

#             python_script_path = 'streamdiffusionTD\\main_sdtd.py'
#             sender_arg = f' --sender "{sender_name}"' if sender_name else ""
#             generation_command = f'{python_script_path} --input {input_type}{sender_arg}'

#             full_command = f'cmd.exe /c "{activate_script_path} && python {generation_command}"'
#             try:
#                 subprocess.Popen(full_command, cwd=base_folder, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
#                 self.logger.log(f'Started with {input_type} input silently.', level='INFO')
#             except Exception as e:
#                 self.logger.log(f'Error Startstream: Failed to start streaming process without command window. Error: {e}', level='ERROR')
#             return
                
#         bat_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'Start_StreamDiffusion.bat')
#         debug_cmd = 'pause' if self.ownerComp.par.Debugcmd.eval() else ''
#         use_powershell = self.ownerComp.par.Powershell.eval()
#         if use_powershell:
#             sender_arg = f" --sender '{sender_name}'" if sender_name else ""
#             batch_file_content = f"""
# @echo off
# cd /d %~dp0
# if exist venv (
#     PowerShell -Command "& {{& 'venv\\Scripts\\Activate.ps1'; & 'venv\\Scripts\\python.exe' 'streamdiffusionTD\\main_sdtd.py' --input {input_type}{sender_arg}}}"
# ) else (
#     PowerShell -Command "& {{& '.venv\\Scripts\\Activate.ps1'; & '.venv\\Scripts\\python.exe' 'streamdiffusionTD\\main_sdtd.py' --input {input_type}{sender_arg}}}"
# )
# {debug_cmd}
#     """
#         else:
#             sender_arg = f' --sender "{sender_name}"' if sender_name else ""
#             batch_file_content = f"""
# @echo off
# cd /d %~dp0
# if exist venv (
#     call venv\\Scripts\\activate.bat
#     venv\\Scripts\\python.exe streamdiffusionTD\\main_sdtd.py --input {input_type}{sender_arg}
# ) else (
#     call .venv\\Scripts\\activate.bat
#     .venv\\Scripts\\python.exe streamdiffusionTD\\main_sdtd.py --input {input_type}{sender_arg}
# )
# {debug_cmd}
#     """
#         with open(bat_file_path, 'w') as bat_file:
#             bat_file.write(batch_file_content)
#         # Execute bat file in a new command window
#         # subprocess.Popen('start /min cmd.exe /C call ' + bat_file_path, cwd=self.ownerComp.par.Basefolder.eval(), shell=True)
#         subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=self.ownerComp.par.Basefolder.eval())
#         self.logger.log(f'Started with {input_type} input using {("PowerShell" if use_powershell else "CMD")}.', level='INFO')

    def Stopstream(self):
        print("stop stream")
        """
        Sends a /stop command to the OSC server.
        """
        osc_out = op('oscout1')
        stop_address = '/stop'
        osc_out.sendOSC(stop_address, [])
        self.logger.log('Stopping stream...', level='INFO')
        run("me.par.Streamactive = False", delayFrames = 10, fromOP = self.ownerComp)

    # def Streamtype(self):
    #     if self.ownerComp.par.Streamtype.eval() == 'ndi':
    #         op('syphonspoutin').bypass = True
    #         op('ndiin').bypass = False
    #     elif self.ownerComp.par.Streamtype.eval() == 'spout':
    #         op('syphonspoutin').bypass = False
    #         op('ndiin').bypass = True

    def Streamactive(self):
        print("stream active")
        current_timestamp = str(datetime.datetime.now())
        callback_data = {
            'timestamp': current_timestamp,
        }

        if self.ownerComp.par.Streamactive:
            
            self.Updatestreamname()
            self.Updatesettings()
            self.Promptblock()
            # Call onStreamStart callback
            if self.ownerComp.par.Onstreamstart:
                self.ownerComp.DoCallback("onStreamStart", callback_data)
            self.logger.log("onStreamStart: StreamDiffusion has begun streaming", f"Called onStreamStart with data: {callback_data}", level='INFO')

        else:
            # Call onStreamEnd callback
            if self.ownerComp.par.Onstreamend:
                self.ownerComp.DoCallback("onStreamEnd", callback_data)
            self.logger.log("onStreamEnd: StreamDiffusion has ended streaming", f"Called onStreamEnd with data: {callback_data}", level='INFO')


    def Updatestreamname(self):
        print("update stream name")
        try:
            source_name = op('stream_osc_data')[1, 'output-name'].val.lower()
            # print(source_name)

            # Store the current value of par.Streaminname
            current_Streaminname = self.ownerComp.par.Streaminname.eval()

            # Check if any of the menu options in self.ownerComp.par.Streaminname contain the source_name
            for i, menu_label in enumerate(self.ownerComp.par.Streaminname.menuLabels):
                if source_name in menu_label.lower():
                    # If found, set the par.Streaminname to that
                    self.ownerComp.par.Streaminname = self.ownerComp.par.Streaminname.menuNames[i]
                    break
            # If par.Streaminname has changed, log the change
            if self.ownerComp.par.Streaminname.eval() != current_Streaminname:
                self.logger.log(f'Source Detected. Source name changed to {self.ownerComp.par.Streaminname.eval()}', level='INFO')
        except:
            pass

    def Clonestreamdiffusion(self):
        print("clone stream diffusion")
        repo_url = 'https://github.com/cumulo-autumn/StreamDiffusion.git'
        base_folder_param = self.ownerComp.par.Basefolder  # Adhering to PND
        print(base_folder_param)
        chosen_folder = base_folder_param.eval() if base_folder_param and base_folder_param.eval() else None

        # Check if the chosen_folder is already a git repository
        if chosen_folder and os.path.isdir(os.path.join(chosen_folder, '.git')):
            # If it is, ask the user what they want to do
            choice = ui.messageBox('Git Repository Detected',
                                "The selected folder is already a Git repository.\n"
                                "What would you like to do?",
                                buttons=['Update Repo', 'Choose New Download Location', 'Cancel'])
            if choice == 0:  # Update Repo
                try:
                    command = ['git', '-C', chosen_folder, 'pull']
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    stdout, stderr = process.communicate()

                    if process.returncode != 0:
                        ui.messageBox('Error', f'Error updating repository:\n{stderr}')
                        self.logger.log(f'rtLCM: Error updating repository:\n{stderr}', level='ERROR')
                        return False
                    else:
                        self.logger.log('Update successful.', level='INFO')
                        return True
                except Exception as e:
                    ui.messageBox('Error', f'Failed to execute the command: {e}')
                    self.logger.log(f'rtLCM: Failed to execute the command: {e}', level='ERROR')
                    return False
            elif choice == 1:  # Pick New Location for Fresh Download
                chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository')
                if chosen_folder is None:
                    return False
            else:  # Cancel or 'x' button
                return False

        # If it's not a git repository, or if the user chose to download to a new location, proceed with cloning
        success = self.clone_git_to_folder(repo_url, chosen_folder=chosen_folder, folder_parameter='Basefolder')
        
        if success:
            self.logger.log('Download successful.', level='INFO')
        else:
            self.logger.log('Download failed.', level='ERROR')

    def clone_git_to_folder(self, repo_url, chosen_folder=None, folder_parameter=None):
        print("clone git to folder")
        # Check if Git is installed
        if not self.is_git_installed():
            ui.messageBox('Git Not Found', 'Git is not installed on this system. Please install Git.')
            self.logger.log('rtLCM: Git Not Found', level='ERROR')
            return False

        # Check for internet connectivity
        if not self.is_connected():
            ui.messageBox('Error', 'No internet connection detected.')
            self.logger.log('rtLCM: No internet connection detected.', level='ERROR')
            return False

        # If no folder has been chosen, open a dialog for the user to select one
        if chosen_folder is None:
            chosen_folder = ui.chooseFolder(title='Select a folder to clone the repository')

        # If the user cancels the folder selection, chosen_folder will be None
        if chosen_folder is None:
            return False
        # print("chosen folder")
        # Prepare the command to clone the repository
        git_folder_name = repo_url.split('/')[-1].replace('.git', '')
        clone_destination = os.path.join(chosen_folder, git_folder_name)
        from platform import system

        

        print("clone destination")
        print(clone_destination)

        if (platform.system() == 'Windows'):
            clone_destination = clone_destination.replace('/', '\\')  # Ensure the path uses backslashes
        print("clone destination")
        print(clone_destination)
        # Execute the command to clone the repository
        try:
            command = ['git', 'clone', repo_url, clone_destination]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                ui.messageBox('Error', f'Error cloning repository:\n{stderr}')
                self.logger.log(f'rtLCM: Error cloning repository:\n{stderr}', level='ERROR')
                return False

        except Exception as e:
            ui.messageBox('Error', f'Failed to execute the command: {e}')
            self.logger.log(f'rtLCM: Failed to execute the command: {e}', level='ERROR')
            return False

        # If a folder_parameter is provided, set the corresponding TouchDesigner parameter
        if folder_parameter:
            try:
                setattr(self.ownerComp.par, folder_parameter, clone_destination)
            except Exception as e:
                ui.messageBox('Error', f'Failed to set the parameter: {folder_parameter}')
                self.logger.log(f'rtLCM: Failed to set the parameter: {folder_parameter}', level='ERROR')
                return False
        
        return True
    
    def is_git_installed(self):
        print("is git installed")
        try:
            subprocess.check_output(["git", "--version"])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def is_connected(self):
        print("is connected")
        try:
            socket.create_connection(("www.google.com", 80))
            return True
        except OSError:
            return False


    def is_git_lfs_installed(self):
        print("is git lfs installed")
        try:
            subprocess.check_output(["git", "lfs", "version"], creationflags=subprocess.CREATE_NO_WINDOW)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_git_lfs(self):
        print("install git lfs")
        try:
            subprocess.check_call(["git", "lfs", "install"])
            self.logger.log("Git LFS installed successfully.", level="INFO")
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Failed to install Git LFS. Error: {e}", level="ERROR")



    def Clonelocalmodels(self):
        print("clone local models")
        """
        Clones the specified repositories from Hugging Face into the local models directory.
        """
        if self.ownerComp.par.Basefolder.eval() in ['', None]:
            self.logger.log(f'Warning: Basefolder is not set. Please set the Basefolder parameter to proceed.', level='WARNING')
            return
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        models_folder = os.path.join(base_folder, 'models')

        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

        model_info = [
            ('https://huggingface.co/madebyollin/taesd', 'VAE/taesd', 'Customvae'),
            ('https://huggingface.co/latent-consistency/lcm-lora-sdv1-5', 'LCM_LoRA/lcm-lora-sdv1-5', 'Customlcm')
        ]

        for repo_url, model_subpath, param_name in model_info:
            target_folder = os.path.join(models_folder, model_subpath)
            was_cloned = self.clone_repo(repo_url, target_folder)
            self.update_local_model_par(was_cloned, target_folder, param_name)

    def clone_repo(self, repo_url, target_folder):
        print("clone repo")
        """
        Clones a git repository to a specified target folder.
        Returns True if cloning was successful, False otherwise.
        """
        if os.path.exists(os.path.join(target_folder, '.git')):
            self.logger.log(f'Warning: Model download skipped. {target_folder} is already a git repository. Cloning skipped.', level='WARNING')
            return True
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        clone_command = ['git', 'clone', repo_url, target_folder]
        try:
            subprocess.check_call(clone_command)
            self.logger.log(f'Successfully cloned {repo_url} into {target_folder}', level='INFO')
            # Remove the .git/lfs/objects directory to save space
            git_lfs_objects_path = os.path.join(target_folder, '.git', 'lfs', 'objects')
            if os.path.exists(git_lfs_objects_path):
                shutil.rmtree(git_lfs_objects_path)
                self.logger.log(f'Removed Git LFS objects from {target_folder}', level='INFO')
            return True
        except subprocess.CalledProcessError as e:
            self.logger.log(f'Failed to clone {repo_url}. Error: {e}', level='ERROR')
            return False
        except Exception as e:
            self.logger.log(f'Unexpected error occurred while cloning {repo_url}. Error: {e}', level='ERROR')
            return False


    def update_local_model_par(self, was_cloned, target_folder, parameter_name):
        print("update local model par")
        """
        Updates the component parameters if the model is downloaded successfully.
        """
        if was_cloned:
            use_parameter_name = 'Use' + parameter_name.lower()
            setattr(self.ownerComp.par, use_parameter_name, True)
            setattr(self.ownerComp.par, parameter_name, target_folder)
            self.logger.log(f'Parameters updated for {parameter_name} with {target_folder}', level='INFO')


    def Dlhfmodel(self):
        print("dlhfmodel")  
        self.Gethfmodelinfo(show_in_viewer=False)
        model_id = self.ownerComp.par.Dlhfmodelid.eval()
        model_type = self.ownerComp.par.Dlhfmodeltype.eval()  # This is a menu parameter

        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())

        # Check if Git LFS is installed
        if not self.is_git_lfs_installed():
            message = """
            Git LFS (Large File Storage) might be required to download and manage large files with git.

            Do you want to install Git LFS now?
            """
            choice = ui.messageBox('Git LFS Required', message, buttons=['Install', 'Cancel'])

            if choice == 0:  # Install Git LFS
                self.install_git_lfs()
            else:
                self.logger.log("Git LFS installation canceled. Model download aborted.", level="WARNING")
                return

        # Determine target folder based on model type
        if model_type == 'SD':
            target_folder = os.path.join(base_folder, 'models', 'Model', model_id.split('/')[-1])
        elif model_type == 'LoRA':
            target_folder = os.path.join(base_folder, 'models', 'LoRA', model_id.split('/')[-1])
        elif model_type == 'VAE':
            target_folder = os.path.join(base_folder, 'models', 'VAE', model_id.split('/')[-1])
        elif model_type == 'LCM':
            target_folder = os.path.join(base_folder, 'models', 'LCM', model_id.split('/')[-1])
        else:
            self.logger.log(f'Unknown model type: {model_type}.', level='ERROR')
            return

        # Check if the target folder exists
        if os.path.exists(target_folder):
            message = f"The folder '{target_folder}' already exists. What would you like to do?"
            choice = ui.messageBox('Folder Exists', message, buttons=['Update', 'Delete and Download Fresh', 'Cancel'])
            if choice == 0:  # Update
                # Perform git pull to update the existing repository
                try:
                    subprocess.check_call(['git', '-C', target_folder, 'pull'])
                    self.logger.log(f'Successfully updated the repository at {target_folder}', level='INFO')
                    return
                except subprocess.CalledProcessError as e:
                    self.logger.log(f'Failed to update the repository at {target_folder}. Error: {e}', level='ERROR')
                    return
            elif choice == 1:  # Delete and Download Fresh
                # Delete the existing folder and proceed with fresh download
                try:
                    shutil.rmtree(target_folder)
                except Exception as e:
                    self.logger.log(f'Failed to delete the folder at {target_folder}. Error: {e}', level='ERROR')
                    message = f"Failed to delete the folder at {target_folder}. Please manually remove the folder and try again."
                    ui.messageBox('Delete Failed', message)
                    return
            else:  # Cancel
                return

        # Create the target folder
        os.makedirs(target_folder, exist_ok=True)
        # # Ensure the target folder exists
        # if not os.path.exists(target_folder):
        #     os.makedirs(target_folder)
        # else:
        #     self.logger.log(f'{model_id} already exists at {target_folder}.', level='WARNING')
        #     # Update the models list only for SD models
        #     if model_type == 'SD':
        #         self.Savemodel(target_folder)
        #         # Ensure the model table is updated to reflect the new addition
        #         self.sync_model_table()
        #     return
        
        if "https://" in model_id:
            repo_url = model_id
        else:
            repo_url = f'https://huggingface.co/{model_id}'
        clone_command = ['git', 'clone', repo_url, target_folder]

        try:
            subprocess.check_call(clone_command)
            self.logger.log(f'Successfully cloned {model_id} into {target_folder}', level='INFO')
        except subprocess.CalledProcessError as e:
            self.logger.log(f'Failed to clone {model_id}. Error: {e}', level='ERROR')
            return

        # Remove the .git/lfs/objects directory to save space
        git_lfs_objects_path = os.path.join(target_folder, '.git', 'lfs', 'objects')
        if os.path.exists(git_lfs_objects_path):
            shutil.rmtree(git_lfs_objects_path)
            self.logger.log(f'Removed Git LFS objects from {target_folder}', level='INFO')

        # Update the models list only for SD models
        if model_type == 'SD':
            self.Savemodel(target_folder)
            # Ensure the model table is updated to reflect the new addition
            self.sync_model_table()

    def Gethfmodelinfo(self, show_in_viewer=True):
        print("get hf model info")
        model_id = self.ownerComp.par.Dlhfmodelid.eval()
        base_folder = self.ownerComp.par.Basefolder.eval()

        # Path to the virtual environment's Python executable
        venv_path = os.path.join(base_folder, 'venv', 'Scripts', 'python.exe')
        if not os.path.exists(venv_path):
            # Try alternative location for virtual environments
            venv_path = os.path.join(base_folder, '.venv', 'Scripts', 'python.exe')
            if not os.path.exists(venv_path):
                self.logger.log("Error: Python executable in virtual environment not found.", level="ERROR")
                return

        # Python script as a string
        python_script = f"""
import requests
import json
import subprocess
import platform


def get_repo_details(model_id):
    url = f"https://huggingface.co/api/models/{model_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {{'error': 'Failed to fetch details', 'status_code': response.status_code}}

model_details = get_repo_details("{model_id}")
print(json.dumps(model_details))

    """

        try:
            # Running the Python script with subprocess
            process = subprocess.Popen([venv_path, "-c", python_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, creationflags=subprocess.CREATE_NO_WINDOW)
            stdout, stderr = process.communicate()
            # # Running the Python script with subprocess
            # process = subprocess.Popen([venv_path, "-c", python_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            # stdout, stderr = process.communicate()
            if stderr:
                self.logger.log(f"Error fetching model details: {stderr}", level="ERROR")
            else:
                model_details = json.loads(stdout)
                formatted_text = f"Model Information:\n"
                formatted_text += f"ID: {model_details['id']}\n"
                formatted_text += f"Author: {model_details['author']}\n"
                formatted_text += f"Last Modified: {model_details['lastModified']}\n"
                formatted_text += f"Private: {'Yes' if model_details['private'] else 'No'}\n"
                formatted_text += f"Disabled: {'Yes' if model_details['disabled'] else 'No'}\n"
                formatted_text += f"Pipeline Tag: {model_details['pipeline_tag']}\n"
                formatted_text += f"Tags: {', '.join(model_details['tags'])}\n"
                formatted_text += f"Likes: {model_details['likes']}\n"
                formatted_text += f"Downloads: {model_details['downloads']}\n"
                op('model_data').par.text = formatted_text    
                if show_in_viewer:
                    op('model_data').openViewer()
                #if model_details['tags'] contains 'lora' as a tag set par.Dlhfmodeltype as #LoRA
                if 'lora' in model_details['tags']:
                    self.ownerComp.par.Dlhfmodeltype = 'LoRA'
                return
        except Exception as e:
            self.logger.log(f"Error fetching model details", f"Error fetching model details: {str(e)}", level="WARNING")
            return

    def create_folder_if_not_exists(self, folder_path):
        print("create folder if not exists")
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                self.logger.log(f"Created folder: {folder_path}", level="INFO")
            except OSError as e:
                self.logger.log(f"Error creating folder: {folder_path}. Error: {e}", level="ERROR")

    def Uiviewlora(self):
        print("ui view lora")
        lora_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models', 'LoRA')
        self.create_folder_if_not_exists(lora_folder)
        ui.viewFile(lora_folder, showInFolder=False)

    def Uiviewmodel(self):
        print("ui view model")
        model_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models', 'Model')
        self.create_folder_if_not_exists(model_folder)
        ui.viewFile(model_folder, showInFolder=False)

    def Uiviewlcm(self):
        print("ui view lcm")
        lcm_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models', 'LCM')
        self.create_folder_if_not_exists(lcm_folder)
        ui.viewFile(lcm_folder, showInFolder=False)

    def Uiviewvae(self):
        print("ui view vae")
        vae_folder = os.path.join(self.ownerComp.par.Basefolder.eval(), 'models', 'VAE')
        self.create_folder_if_not_exists(vae_folder)
        ui.viewFile(vae_folder, showInFolder=False)

    def find_python310_executable(self):
        print("find python310 executable")
        possible_paths = [
            # Common installation paths on Windows
            f"C:/Users/{os.getlogin()}/AppData/Local/Programs/Python/Python310/python.exe",
            "C:/Python310/python.exe",
            "C:/Program Files/Python310/python.exe",
            "C:/Program Files (x86)/Python310/python.exe",
        ]

        # Common installation paths on macOS
        possible_paths.extend([
            "/usr/local/bin/python",
            "/usr/local/bin/python3",
            "/usr/bin/python",
            "/usr/bin/python3",
            "/opt/homebrew/bin/python3",
            "/opt/homebrew/bin/python"
        ])

        # Adding Python executables from the system PATH
        path_env = os.environ.get('PATH', '')
        for path in path_env.split(os.pathsep):
            python_exe_path = os.path.join(path, 'python3')
            if os.path.isfile(python_exe_path):
                possible_paths.append(python_exe_path)

        # Check each path for Python 3.10
        for path in set(possible_paths):  # Use set to avoid duplicates
            if os.path.exists(path):
                try:
                    output = subprocess.check_output([path, '--version'], stderr=subprocess.STDOUT, universal_newlines=True)
                    print(output)
                    if "Python 3.1" in output:
                        return path  # Return the path if it's the correct version
                except Exception as e:
                    continue  # If an error occurs (e.g., subprocess call fails), continue to the next path

        return None  # Return None if no matching executable is found

    def Installstreamdiffusion(self):
        self.copy_ndi_code()
        python_exe = self.find_python310_executable()
        if not python_exe:
            choice = ui.messageBox('Installation Error ! No Python detected.', 
                                'Python 3.10 executable not found. Please ensure Python 3.10 is installed and accessible.\nDo you want to continue anyway?', 
                                buttons=['Continue Anyway', 'Cancel'])
            if choice != 0:  # Continue Anyway
                python_exe = 'Python Not Found'
                return False
        else:
            
            pass 

        try:
            cuda_version_output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, universal_newlines=True)
            if "Cuda compilation tools, release 11.8" in cuda_version_output:
                cuda_version = 'cu118'
            elif "Cuda compilation tools, release 12.1" in cuda_version_output:
                cuda_version = 'cu121'
            else:
                raise Exception(f"Detected CUDA version is not supported. Output: {cuda_version_output}")
        except Exception as e:
            cuda_version = 'CUDA Not Found'
            choice = ui.messageBox('Installation Error !', 
                                f'CUDA detection failed or unsupported CUDA version detected. The script requires CUDA 11.8 or 12.1 for installation. Error: {e}\nDo you want to continue anyway?', 
                                buttons=['Continue Anyway', 'Cancel'])
            if choice != 0:  # Continue Anyway
                cuda_version = 'CUDA Not Found'
                return False   
        if platform.system() == 'Windows':
            bat_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'Install_StreamDiffusion.bat')
        else:
            bat_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'Install_StreamDiffusion.sh')
            
        base_folder = tdu.expandPath(self.ownerComp.par.Basefolder.eval())
        set_base_folder = False
        if base_folder is None or base_folder == '':
            previous_base_folder = base_folder
            base_folder = project.folder
            self.ownerComp.par.Basefolder = base_folder
            set_base_folder = True
        installation_details = f"Python 3.10 executable: {python_exe}\nCUDA version: {cuda_version}\nBase folder for venv: {base_folder}\nInstalling packages for: StreamDiffusionTD"
        choice = ui.messageBox('Installation Details', 
                            f'{installation_details}\n\nDo you want to proceed with the installation?', 
                            buttons=['Install', 'Cancel'])
        if choice != 0: 
            if set_base_folder:
                self.ownerComp.par.Basefolder = previous_base_folder
            return False
        
        batch_file_content_win = f"""
            @echo off
            echo Current directory: %CD%
            cd /d "{base_folder}"
            echo Changed directory to: %CD%
            set "PIP_DISABLE_PIP_VERSION_CHECK=1"
            if not exist "venv" (
                echo Creating Python venv at: "{base_folder}\\venv"
                "{python_exe}" -m venv venv
            ) else (
                echo Virtual environment already exists at: "{base_folder}\\venv"
            )

            echo Attempting to activate virtual environment...
            call "venv\\Scripts\\activate.bat"

            rem Check if the virtual environment was activated successfully
            if "%VIRTUAL_ENV%" == "" (
                echo Failed to activate virtual environment. Please check the path and ensure the venv exists.
                echo Path to venv: "{base_folder}\\venv"
                echo VIRTUAL_ENV: "%VIRTUAL_ENV%"
                pause /b 1
            ) else (
                echo Virtual environment activated.
            )

            echo Installing 'wheel' to ensure successful building of packages...
            python -m pip install wheel

            echo Installing nvidia-pyindex to ensure access to NVIDIA-specific packages...
            python -m pip install nvidia-pyindex

            echo Installing dependencies with pip from the activated virtual environment...
            python -m pip install --upgrade pip
            python -m pip install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/{cuda_version}
            python -m pip install -e .
            python setup.py develop
            python -m pip install -r streamdiffusionTD/requirements.txt

            echo Installation Finished
            pause
        """

        batch_file_content_mac = f"""
            #!/bin/bash
            echo "Current directory: $PWD"
            cd "{base_folder}"
            echo "Changed directory to: $PWD"
            export PIP_DISABLE_PIP_VERSION_CHECK=1
            if [ ! -d "venv" ]; then
            echo "Creating Python venv at: {base_folder}/venv"
            python3 -m venv venv
            else
            echo "Virtual environment already exists at: {base_folder}/venv"
            fi

            echo "Attempting to activate virtual environment..."
            source venv/bin/activate

            if [ -z "$VIRTUAL_ENV" ]; then
                echo "Failed to activate virtual environment. Please check the path and ensure the venv exists."
                echo "Path to venv: {base_folder}/venv"
                echo "VIRTUAL_ENV: $VIRTUAL_ENV"
                exit 1
            else
                echo "Virtual environment activated."
            fi

            echo "Installing"

            
            pip install --upgrade pip --trusted-host pypi.org
            
            
            echo "Installing 'wheel' to ensure successful building of packages..."
            python -m pip install wheel --trusted-host pypi.org

            echo "Installing dependencies with pip from the activated virtual environment..."
            
            python -m pip install pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu --trusted-host download.pytorch.org
            python -m pip . --trusted-host pypi.org
            python -m pip install -r streamdiffusionTD/requirements.txt --trusted-host pypi.org

            python -m pip uninstall --yes numpy
            python -m pip install numpy --trusted-host pypi.org

            echo "Installation Finished"
            read -p "Press any key to continue..."
        """

        if platform.system() == 'Windows':
            batch_file_content = batch_file_content_win
        else:
            batch_file_content = batch_file_content_mac

        print("Writing batch file for installation...")
        # Write to the batch file
        with open(bat_file_path, 'w') as bat_file:
            bat_file.write(batch_file_content)

        print(f"Executing batch file: {bat_file_path}")
        # Execute the batch file in a new command window
        if platform.system() == 'Windows':
            subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=base_folder)
        else:
            print("running sh file")
            os.chmod(bat_file_path, 0o755)
            print(bat_file_path + base_folder)
            result = subprocess.run(['sh', bat_file_path], cwd=base_folder, stdout=subprocess.PIPE)
            print(result.stdout.decode('utf-8'))

            
        print("StreamDiffusion installation initiated with detected Python 3.10 and CUDA version.")


    def print_installation_details(self, python_exe, cuda_version, base_folder):
        print("\n====================================")
        print("Installation Details:")
        print("------------------------------------")
        print(f"Python 3.10 executable: {python_exe}")
        print(f"CUDA version: {cuda_version}")
        print(f"Base folder for venv: {base_folder}")
        print("Installing packages for: StreamDiffusionTD")
        print("====================================\n")


    def Installtensorrt(self):
        self.update_stream_config_dat()
        self.copy_ndi_code()
        
        if platform.system() == 'Windows':
            self.install_tensorrt()

    def install_tensorrt(self):
        """
        Creates and executes a batch file to install TensorRT.
        It activates the Python virtual environment and runs the command for TensorRT installation.
        """
        bat_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'Install_TensorRT.bat')

        batch_file_content = f"""
        @echo off
        echo Current directory: %CD%
        cd /d "{self.ownerComp.par.Basefolder.eval()}"

        echo Attempting to activate virtual environment...
        call "venv\\Scripts\\activate.bat"

        rem Check if the virtual environment was activated successfully
        if "%VIRTUAL_ENV%" == "" (
            echo Failed to activate virtual environment. Please check the path and ensure the venv exists.
            pause /b 1
        ) else (
            echo Virtual environment activated.
        )

        echo Installing TensorRT...
        python -m streamdiffusion.tools.install-tensorrt

        echo TensorRT installation finished
        pause
        """

        
        print("Writing batch file for TensorRT installation...")
        # Write the batch file content
        with open(bat_file_path, 'w') as bat_file:
            bat_file.write(batch_file_content)

        print(f"Executing batch file: {bat_file_path}")
        # Execute the batch file in a new command window
        subprocess.Popen(['cmd.exe', '/C', bat_file_path], cwd=self.ownerComp.par.Basefolder.eval())
        print("TensorRT installation initiated.")



    def copy_ndi_code(self):
        print("copy ndi code")
        """
        Copies the contents of Text DATs from the 'streamdiffusionTD' base component into corresponding files in the 'streamdiffusionTD' folder in the Basefolder.
        """
        base_folder = self.ownerComp.par.Basefolder.eval()

        # Ensure the 'streamdiffusionTD' folder exists
        streamdiffusionTD_folder = os.path.join(base_folder, 'streamdiffusionTD')
        if not os.path.exists(streamdiffusionTD_folder):
            os.makedirs(streamdiffusionTD_folder)

        # Paths to the Text DATs inside the 'streamdiffusionTD' base comp
        streamdiffusionTD_comp = self.ownerComp.op('streamdiffusionTD')
        text_dat_paths = {
            'main_sdtd': 'main_sdtd.py',
            'ndi_spout_utils': 'ndi_spout_utils.py',
            'requirements': 'requirements.txt',
            'stream_config': 'stream_config.json',
            'pipeline_td': 'pipeline_td.py',
            'wrapper_td': 'wrapper_td.py'
        }
        # Copy each Text DAT's content into a file in the 'streamdiffusionTD' folder
        for dat_name, filename in text_dat_paths.items():
            text_dat = streamdiffusionTD_comp.op(dat_name)
            if text_dat:
                file_path = os.path.join(streamdiffusionTD_folder, filename)
                with open(file_path, 'w') as file:
                    file.write(text_dat.text)
                self.logger.log(f'Copied {dat_name} to {file_path}', level='Debug')
            else:
                self.logger.log(f'{dat_name} DAT not found in streamdiffusionTD', level='ERROR')

    def Basefolder(self):
        print("base folder")
        self.sync_model_table()

    def update_stream_config_dat(self):
        print("update stream config dat")
        stream_config_dat = self.ownerComp.op('streamdiffusionTD/stream_config')
        try:
            config = json.loads(stream_config_dat.text)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return
        
        keys_to_remove = []

        for param_name, json_key in param_json_map.items():

            if param_name == "Customlcm" and not self.ownerComp.par.Usecustomlcm:
                keys_to_remove.append(json_key)
                continue
            if param_name == "Customvae" and not self.ownerComp.par.Usecustomvae:
                keys_to_remove.append(json_key)
                continue

            if json_key == "t_index_list":
                t_index_list = []
                # Iterate over each block in the Tindexblock sequence
                for block in self.ownerComp.par.Tindexblock.sequence:
                    step_value = block.par.Step.eval()  # Extract the Step value
                    t_index_list.append(step_value)     # Append the step value to the list
                config[json_key] = t_index_list

            else:
                param_value = getattr(self.ownerComp.par, param_name).eval()
                config[json_key] = param_value

        for key in keys_to_remove:
            if key in config:
                del config[key]
            # print(f"Deleted {key} from config")
        if self.ownerComp.par.Uselora:
            lora_dict = {}
            for block in self.ownerComp.par.Loradictblock.sequence:
                lora_path = block.par.Lorapath.eval()
                weight = block.par.Weight.eval()
                if lora_path and weight != 0:  # Check if lora_path is not None or an empty string and weight is not 0
                    lora_dict[lora_path] = weight
        else:
            lora_dict = None
        config["lora_dict"] = lora_dict

        stream_config_dat.text = json.dumps(config, indent=4)




    def Savemodel(self, model=None):
        print("save model")
        models_file_path = os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'working_models.json')
        os.makedirs(os.path.dirname(models_file_path), exist_ok=True)
        models_list = self.load_models_list(models_file_path)
        
        # Function to check if a string is a Windows absolute path
        def is_windows_abs_path(s):
            return len(s) > 2 and s[1] == ':' and (s[0].isalpha() and s[2] == '\\' or s[2] == '/')

        # Get the current model ID/PATH and normalize it if it's a full Windows path
        if not model:
            current_model_id = self.ownerComp.par.Modelid.eval()

            if is_windows_abs_path(current_model_id):
                current_model_id = os.path.normpath(current_model_id)
        else:
            current_model_id = model

        # Normalize all paths in the models_list for comparison, only if they are full Windows paths
        normalized_models_list = [os.path.normpath(model_id) if is_windows_abs_path(model_id) else model_id for model_id in models_list]
        
        # Check if the current model ID/PATH is not in the normalized list
        if current_model_id not in normalized_models_list:
            models_list.append(current_model_id)  # Append the original ID/path
            with open(models_file_path, 'w') as file:
                json.dump(models_list, file, indent=4)
            self.logger.log(f'Model ID {current_model_id} added to the working models list.', level='INFO')
        else:
            self.logger.log(f'Model ID {current_model_id} is already in the working models list.', level='INFO')
        self.sync_model_table()


    def sync_model_table(self):
        print("sync model table")
        """
        Synchronizes the model IDs from the working_models.json file with the 'model_table' DAT.
        Adds a second column 'Label' to the table, which contains only the file name if the model path is a full file path,
        or the base model path/id if it's a relative path or not an actual file.
        """
        models_list = self.load_models_list(os.path.join(self.ownerComp.par.Basefolder.eval(), 'StreamDiffusionTD', 'working_models.json'))
        # Get the 'model_table' DAT operator
        model_table = op('model_table')

        # Clear the table before adding new data
        model_table.clear()

        # Add headers
        model_table.appendRow(['Name', 'Label'])
        model_table.appendRow(['select_working_model_from_dropdown', 'Select Working Model from Dropdown'])
        if self.ownerComp.par.Basefolder != '':
            # Add each model ID as a new row in the table
            for model_id in models_list:
                # Determine the label based on whether the model_id is a full file path
                if os.path.isabs(model_id) and os.path.isfile(model_id):
                    # Extract the file name from the full file path
                    label = os.path.basename(model_id)
                else:
                    # Use the base model path/id as the label
                    label = os.path.basename(model_id)

                model_table.appendRow([model_id, label])

            self.logger.log('Model table synchronized with working models list.', level='INFO')

    def load_models_list(self, models_file_path):
        print("load models list")
        """
        Helper function that loads the list of models from the specified JSON file.

        Parameters:
        models_file_path (str): The file path to the JSON file containing the models list.

        Returns:
        list: The list of models loaded from the file, or an empty list if the file doesn't exist or an error occurs.
        """
        # Initialize an empty list for models
        models_list = []

        # Check if the file exists
        if os.path.exists(models_file_path):
            # If the file exists, read the current list of models
            with open(models_file_path, 'r') as file:
                try:
                    models_list = json.load(file)
                except json.JSONDecodeError:
                    # If there's a JSON decode error, keep the list empty
                    pass

        return models_list
    
    def Mymodels(self):
        print("my models")
        modelpreset = self.ownerComp.par.Mymodels.eval()
        if modelpreset != 'select_working_model_from_dropdown':
            self.ownerComp.par.Modelid = modelpreset
        # debug(modelpreset)
        
    def Updateloramodelpath(self, block, path):
        print("update lora model path")
        setattr(self.ownerComp.par, f'Loradictblock{block}lorapath', path)


    def Openvenv(self):
        print("open venv")
        base_folder = self.ownerComp.par.Basefolder.eval()
        
        # Check if 'venv' or '.venv' directory exists and construct the activate script path
        if os.path.exists(os.path.join(base_folder, 'venv')):
            venv_path = os.path.join(base_folder, 'venv')
        elif os.path.exists(os.path.join(base_folder, '.venv')):
            venv_path = os.path.join(base_folder, '.venv')
        else:
            self.logger.log("Error: Virtual environment not found in the specified base folder.", level="ERROR")
            return
        
        activate_script_path = os.path.join(venv_path, 'Scripts', 'activate.bat')
        
        if not os.path.exists(activate_script_path):
            self.logger.log("Error: activate.bat script not found in the virtual environment.", level="ERROR")
            return
        
        # Open a new CMD window and activate the virtual environment
        cmd_command = f'start cmd.exe /K "cd /d {base_folder} & {activate_script_path}"'
        
        try:
            subprocess.Popen(cmd_command, shell=True)
            self.logger.log("Virtual environment activated in a new CMD window.", level="INFO")
        except Exception as e:
            self.logger.log(f"Error opening CMD window and activating virtual environment: {e}", level="ERROR")









    def Createpars(self):
        print("create pars")
        # Configuration as per stream_config.json
        config = {
            "Modelid": {"type": "str", "default": "stabilityai/sd-turbo", "help": "The name of the model to use for image generation."},
            "Tindexlist": {"type": "str", "default": "[18, 25, 33, 41]", "help": "List of T-indices for the model."},
            "Loradict": {"type": "str", "default": "{\"F:\\ComfyUI\\ComfyUI_windows_portable\\ComfyUI\\models\\loras\\node-network-000013.safetensors\": 3.0}", "help": "Dictionary of LoRA names and scales."},
            "Prompt": {"type": "str", "default": "abstract artistic portal to surreal smoky void, colorful bright light god rays, a complex smoky billowing smoke tunnel, complex detailed ribbons and sparks, high quality stunning stark contrast, collage 3d and 2d digital art, ripped paper and folded collage paper cut", "help": "The prompt to generate images from."},
            "Negprompt": {"type": "str", "default": "text, words, black and white, shit, shit, shit, low quality, bad quality, blurry, low resolution", "help": "The negative prompt to avoid."},
            "Framesize": {"type": "int", "default": 1, "help": "The frame buffer size for denoising batch."},
            "Width": {"type": "int", "default": 512, "help": "The width of the image."},
            "Height": {"type": "int", "default": 512, "help": "The height of the image."},
            "Acceleration": {"type": "menu", "default": ["none", "xformers", "tensorrt"], "help": "The acceleration method."},
            "Denoisebatch": {"type": "bool", "default": True, "help": "Whether to use denoising batch or not."},
            "Seed": {"type": "int", "default": 3242346, "help": "The seed for the generation."},
            "Cfgtype": {"type": "menu", "default": ["none", "full", "self", "initialize"], "help": "The cfg_type for img2img mode."},
            "Guidancescale": {"type": "float", "default": 0.5, "help": "The CFG scale."},
            "Delta": {"type": "float", "default": 0.5, "help": "The delta multiplier of virtual residual noise."},
            "Addnoise": {"type": "bool", "default": True, "help": "Whether to add noise for following denoising steps or not."},
            "Imagefilter": {"type": "bool", "default": False, "help": "Whether to enable similar image filter or not."},
            "Filterthresh": {"type": "float", "default": 0.99, "help": "The threshold for similar image filter."},
            "Maxskipframe": {"type": "int", "default": 2, "help": "The max skip frame for similar image filter."}
        }

        for par_name, details in config.items():
            par_type = details["type"]
            default = details["default"]
            help_text = details["help"]

            # Use different approach for menu type to set menu names and labels
            if par_type == "menu":
                self.create_parameter(par_name, par_type, default=default, label=par_name, menu_items=default)
            else:
                self.create_parameter(par_name, par_type, default=default, label=par_name)

            # Set help text for the parameter
            getattr(self.ownerComp.par, par_name).help = help_text

    def setup_table(self, table_name, headers=None):
        print("setup table")
        # Check if the table exists, if not, create it
        table = self.ownerComp.op(table_name)
        if table is None:
            table = self.ownerComp.create(tableDAT, table_name)
            table.clear()
            if headers:
                table.appendRow(headers)
        return table

    def setup_project_info_table(self):
        print("setup project info table")
        """
        Sets up a table with project information.
        """
        # Define the headers for the project info table
        headers = ['Info', 'Value']
        project_info_table = self.setup_table('projectInfoTable', headers)
        project_info_table.clear(keepFirstRow = True)
        # Populate the table with project information
        project_data = [
            ('Project Folder', project.folder),
            ('Project Name', project.name),
            ('Last Save Time', project.saveTime),
            ('TDversion', str(app.build)),
            ('OS Name at Last Save', project.saveOSName),
            ('OS Version at Last Save', project.saveOSVersion),
            ('Real-Time State', str(project.realTime)),
            ('Cook Rate', str(project.cookRate)),
            ('Licenses', ', '.join([l.type for l in licenses])),
        ]

        for info, value in project_data:
            project_info_table.appendRow([info, value])


    def create_parameter(self, par_name, par_type, page='Custom', default=None, norm_min=None, norm_max=None, size=1, menu_items=None, label=None, order=None, replace=False, section=None):
        print("create parameter")
        '''
        Creates a custom parameter on a specified page in a TouchDesigner component, supporting a wide 
        range of parameter types. This function allows for creating basic types like 'float', 'int', 
        'str', 'bool', and 'menu', as well as node reference types such as 'op', 'comp', etc. Advanced 
        options like 'label', 'order', and 'replace' enable further customization. For 'float' and 'int' 
        types, 'size' determines the number of values associated with the parameter. The 'default' 
        parameter sets the initial value, with special handling for 'menu' types where it sets the menu 
        options. If 'replace' is set to False, the function errors if the parameter already exists.

        Parameters:
        par_name (str): The name of the parameter.
        par_type (str): The type of the parameter. Supported types are:
                        'float', 'int', 'str', 'bool', 'menu', 'op', 'comp', 'object', 'panelcomp', 
                        'top', 'chop', 'sop', 'mat', 'dat', 'xy', 'xyz', 'xyzw', 'wh', 'uv', 'uvw', 
                        'rgb', 'rgba', 'file', 'folder', 'pulse', 'momentary', 'python', 'par', 'header'.
        page (str): The page name where the parameter will be added. Defaults to 'Custom'.
        default: The default value for the parameter. For 'menu', it should be a list of strings.
        norm_min, norm_max (optional): Normalized minimum and maximum values for 'float' and 'int' types.
        size (int, optional): Number of values for 'float' and 'int' types. Defaults to 1.
        menu_items (list of str, optional): List of menu items for 'menu' type parameters.
        label (str, optional): Display label of the parameter. Defaults to par_name.
        order (int, optional): Display order of the parameter.
        replace (bool, optional): Determines whether to replace an existing parameter. Defaults to True.
                                If set to False, the function errors if the parameter already exists.
        section (bool, optional): If set to True, adds a visual separator above this parameter. Useful for organizing parameters into distinct sections on the UI.

        Returns:
        The created parameter object.
        '''
        # print(f"Creating parameter: Name: {par_name}, Type: {par_type}, Page: {page}")
        # Check if the parameter already exists
        if hasattr(self.ownerComp.par, par_name):
            if not replace:
                # If the parameter exists and 'replace' is False, skip creating the parameter
                return getattr(self.ownerComp.par, par_name)
        # Check if the page exists
        custom_page = next((p for p in self.ownerComp.customPages if p.name == page), None)
        if not custom_page:
            # If the page doesn't exist, create it
            custom_page = self.ownerComp.appendCustomPage(page)

        # Mapping for parameter creation based on type
        create_method = {
            'float': lambda: custom_page.appendFloat(par_name, label=label, size=size, order=order, replace=replace),
            'int': lambda: custom_page.appendInt(par_name, label=label, size=size, order=order, replace=replace),
            'str': lambda: custom_page.appendStr(par_name, label=label, order=order, replace=replace),
            'bool': lambda: custom_page.appendToggle(par_name, label=label, order=order, replace=replace),
            'menu': lambda: custom_page.appendMenu(par_name, label=label, order=order, replace=replace),
            'op': lambda: custom_page.appendOP(par_name, label=label, order=order, replace=replace),
            'comp': lambda: custom_page.appendCOMP(par_name, label=label, order=order, replace=replace),
            'object': lambda: custom_page.appendObject(par_name, label=label, order=order, replace=replace),
            'panelcomp': lambda: custom_page.appendPanelCOMP(par_name, label=label, order=order, replace=replace),
            'top': lambda: custom_page.appendTOP(par_name, label=label, order=order, replace=replace),
            'chop': lambda: custom_page.appendCHOP(par_name, label=label, order=order, replace=replace),
            'sop': lambda: custom_page.appendSOP(par_name, label=label, order=order, replace=replace),
            'mat': lambda: custom_page.appendMAT(par_name, label=label, order=order, replace=replace),
            'dat': lambda: custom_page.appendDAT(par_name, label=label, order=order, replace=replace),
            'xy': lambda: custom_page.appendXY(par_name, label=label, order=order, replace=replace),
            'xyz': lambda: custom_page.appendXYZ(par_name, label=label, order=order, replace=replace),
            'xyzw': lambda: custom_page.appendXYZW(par_name, label=label, order=order, replace=replace),
            'wh': lambda: custom_page.appendWH(par_name, label=label, order=order, replace=replace),
            'uv': lambda: custom_page.appendUV(par_name, label=label, order=order, replace=replace),
            'uvw': lambda: custom_page.appendUVW(par_name, label=label, order=order, replace=replace),
            'rgb': lambda: custom_page.appendRGB(par_name, label=label, order=order, replace=replace),
            'rgba': lambda: custom_page.appendRGBA(par_name, label=label, order=order, replace=replace),
            'file': lambda: custom_page.appendFile(par_name, label=label, order=order, replace=replace),
            'folder': lambda: custom_page.appendFolder(par_name, label=label, order=order, replace=replace),
            'pulse': lambda: custom_page.appendPulse(par_name, label=label, order=order, replace=replace),
            'momentary': lambda: custom_page.appendMomentary(par_name, label=label, order=order, replace=replace),
            'python': lambda: custom_page.appendPython(par_name, label=label, order=order, replace=replace),
            'par': lambda: custom_page.appendPar(par_name, label=label, order=order, replace=replace),
            'header': lambda: custom_page.appendHeader(par_name, label=label, order=order, replace=replace)
        }.get(par_type.lower())

        if create_method is None:
            raise ValueError(f"Unsupported parameter type: {par_type}")

        new_param_group = create_method()

        if new_param_group is None:
            raise Exception("Parameter group creation failed")

        if not hasattr(new_param_group, 'pars'):
            raise Exception(f"Expected ParGroup, got {type(new_param_group)}")

        new_param = new_param_group[0] if new_param_group.pars else None

        if new_param is None:
            raise Exception("Parameter creation failed")
        # Set default, norm_min, norm_max, and menu_items based on the parameter type
        if default is not None:
            if par_type == 'menu' and isinstance(default, list) and all(isinstance(item, str) for item in default):
                new_param.menuNames = default
                new_param.menuLabels = default
            else:
                setattr(self.ownerComp.par, par_name, default)

        if par_type in ['float', 'int']:
            if norm_min is not None:
                new_param.normMin = norm_min
                new_param.min = norm_min
                new_param.clampMin = True
            if norm_max is not None:
                new_param.normMax = norm_max
                new_param.max = norm_max
                new_param.clampMax = True

        if section:
            new_param.startSection = True
        return new_param 
    
    def Printpardetails(self):
        print("print par details")
        """
        This function prints the details of the parameters of the owner component.
        It prints the parameter name, label, value, default value, and the page name.
        """
        print(f"{'Parameter Name':<18}|{'Label':<18}|{'Value':<18}|{'Default':<18}|{'Page':<18}")
        print("-" * 90)
        for par in self.ownerComp.customPars:
            print(f"{par.name:<18}|{par.label:<18}|{str(par.eval()):<18}|{str(par.default):<18}|{par.page.name:<18}")


    def setup_par_details_table(self):
        print("setup par details table")
        """
        This function sets up a table with the details of the parameters of the owner component.
        It defines the headers for the table and populates it with the parameter details.
        """
        # Define the headers for the parameter details table
        headers = ['Parameter Name', 'Label', 'Value', 'Default', 'Page', 'Norm Min', 'Norm Max', 'Min', 'Max', 'Clamp Min', 'Clamp Max', 'Enabled', 'Menu Names', 'Menu Labels']
        
        par_details_table = self.setup_table('parDetailsTable', headers)
        par_details_table.clear(keepFirstRow = True)

        # Populate the table with parameter details
        for par in self.ownerComp.customPars:
            par_details = [
                par.name,
                par.label,
                str(par.eval()),
                str(par.default),
                par.page.name,
                str(par.normMin),
                str(par.normMax),
                str(par.min),
                str(par.max),
                str(par.clampMin),
                str(par.clampMax),
                str(par.enable),
                str(par.menuNames),
                str(par.menuLabels)
            ]
            par_details_table.appendRow(par_details)


    def Editcallbacksscript(self):
        print("edit callbacks script")
        viewop = self.ownerComp.par.Callbackdat.eval()
        viewop.openViewer(unique=False, borders=True)


    def Viewinstallguide(self):
        print("view install guide")
        viewop = op('how_to_install')
        op('how_to_install').par.Reloadsrc.pulse()
        viewop.openViewer(unique=False, borders=True)

