
'''
This script downloads all files (e.g. datasets and models) needed to demo the paper results. 
TODO: complete description.
'''


import requests
import os

#this code snippet is for downloading from gdrive, and is grabbed from 
#     https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url 

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk) #######################################################
                

def _setup_directories(dict_input, shared_pathfromroot = [None]):
    '''
    Sets up directories recursively. 
    Input has to be a dictionary like:
    dict_input = {
        "Material_PaperResults":{
            "Datasets":{
                "Cifar10":None,
                "MNIST":None,
                "Kather":None,
                "DogsWolves":None
            },
            "Models":{
                "ExplainAttention":None,
                "ExplainClassifier:None"
            }
        }
    }
    '''
    assert(len(list(dict_input.keys())) == 1)
    
    #handle the base case where input dictionary is like {"Cifar10": None}
    k0 = list(dict_input.keys())[0]
    if(dict_input[k0] is None):
        #create the path and return without making any change to shared_pathfromroot.
        path_tocreate = k0
        if(shared_pathfromroot[0] is not None):
            path_tocreate = os.path.join(
                shared_pathfromroot[0] + os.path.sep,
                path_tocreate
            )
        if(os.path.isdir(path_tocreate) == False):
            os.mkdir(path_tocreate)
        return
    
    #make the parent directory ===
    path_tocreate = k0
    if(shared_pathfromroot[0] is not None):
        path_tocreate = os.path.join(
            shared_pathfromroot[0] + os.path.sep,
            path_tocreate
        )
    if(os.path.isdir(path_tocreate) == False):
        os.mkdir(path_tocreate)
    shared_pathfromroot = [path_tocreate + ""]
    
    #call on children ====
    dict_children = dict_input[k0]
    for k, v in dict_children.items():
        _setup_directories({k:v}, shared_pathfromroot)
    

def setup_directories():
    '''
    Sets up the directory structure to which the required material will be saved. 
    '''
    dict_input = {
        "Material_PaperResults":{
            "Datasets":{
                "Cifar10":None,
                "MNIST":None,
                "Kather":None,
                "DogsWolves":None
            },
            "Models":{
                "ExplainAttention":None,
                "ExplainClassifier":None
            }
        }
    }
    _setup_directories(dict_input)
    

if __name__ == "__main__":
    setup_directories()



