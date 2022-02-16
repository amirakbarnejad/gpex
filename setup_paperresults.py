
'''
This script downloads all files (e.g. datasets and models) needed to demo the paper results. 
TODO: complete description.
'''


import requests
import os
import urllib
import tarfile



try:
    from google_drive_downloader import GoogleDriveDownloader as gdd
except Exception as e:
    print(str(e))
    print(" google_drive_downloader was not found \n ****************** you can install it via, e.g., pip *************************")
    
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
    
def untargz(str_fname, str_destination):
    file_file = tarfile.open(str_fname)
    file_file.extractall(str_destination)
    file_file.close()

def setup_cifar10material():
    '''
    Downloads cifar10 datasets and trained GPEX models.
    '''
    #download cifar10 dataset ====
    dest_path_dataset = os.path.join(
        "Material_PaperResults",
        "Datasets",
        "Cifar10",
        "cifar-10-python.tar.gz"
    )
    if(os.path.isfile(dest_path_dataset) == False):
        urllib.request.urlretrieve(
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            dest_path_dataset
        )
        untargz(
            dest_path_dataset,
            os.path.join(
                "Material_PaperResults",
                "Datasets",
                "Cifar10"
            )
        )
        path_currentroot_cifar10 = os.path.join(
            "Material_PaperResults",
            "Datasets",
            "Cifar10",
            "cifar-10-batches-py"
        )
        path_destinationroot_cifar10 = os.path.join(
            "Material_PaperResults",
            "Datasets",
            "Cifar10"
        )
        list_fname_tomove = [
            "batches.meta",
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
            "test_batch",
            "readme.html"
        ]
        for fname in list_fname_tomove:
            os.rename(
                os.path.join(path_currentroot_cifar10, fname),
                os.path.join(path_destinationroot_cifar10, fname)
            )
        os.remove(
            os.path.join(path_destinationroot_cifar10, "cifar-10-python.tar.gz")
        )
        
        

    #download the gpex models ====
    file_id_classifier = '1aMJ5KBClnv0sLIAuckMK5I1YIYi2Tc61'
    file_id_attention  = "1CUNmFgh_trvUvsqnhTYOqQ8geTQ7KSSd"
    dest_path_classifier = os.path.join(
        "Material_PaperResults",
        "Models",
        "ExplainClassifier",
        "cifar10.pt"
    )
    dest_path_attention = os.path.join(
        "Material_PaperResults",
        "Models",
        "ExplainAttention",
        "cifar10.pt"
    )
    if(os.path.isfile(dest_path_classifier) == False):
        gdd.download_file_from_google_drive(
            file_id = file_id_classifier,
            dest_path = dest_path_classifier
        )
    if(os.path.isfile(dest_path_attention) == False):
        gdd.download_file_from_google_drive(
            file_id = file_id_attention,
            dest_path = dest_path_attention
        )

    


if __name__ == "__main__":
    setup_directories()
    setup_cifar10material() #TODO:change



