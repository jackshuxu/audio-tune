def get_custom_metadata(info, audio):

    prompt = info["relpath"]
    prompt = prompt.split("/", 1)[-1]  # Get the file name
    prompt = prompt.split(".")[0]      # Remove file extension
    prompt = prompt.split("[")[0]      # Remove everything after '['
    prompt = prompt.strip()            # Remove any leading/trailing spaces
    prompt = prompt.replace("_", " ")  # Replace underscores with spaces
    prompt = prompt.replace("/", " ")  # Replace slashes with spaces

    print(prompt)
    
    return {"prompt": prompt}