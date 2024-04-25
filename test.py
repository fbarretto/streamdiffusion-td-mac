result = subprocess.run(['sh', "Install_StreamDi"], cwd=base_folder, stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))